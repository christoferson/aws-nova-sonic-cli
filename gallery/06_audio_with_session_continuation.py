"""Step 6: session continuation. Keep one conversation going past Nova Sonic's 8-minute connection limit.

Run:  uv run gallery/06_audio_with_session_continuation.py --switch-after 30    (switch every ~30 s to test)
      uv run gallery/06_audio_with_session_continuation.py                      (default: every ~6 minutes)

The problem
-----------
Each bidirectional stream is cut off by the service after about 8 minutes. A long conversation
therefore has to move to a fresh stream ("session") before that, without the user noticing.

How this step does it
---------------------
    1. Record the conversation as it happens (your speech transcript + the assistant's FINAL text),
       like step 04 does for its history file.
    2. Once the active session is older than --switch-after, wait for a quiet moment: the reply has
       fully finished, its audio has played, and the mic hears no speech. If no quiet moment comes
       before HARD_DEADLINE_SECONDS, switch anyway.
    3. Open a new session with the same setup, replay the recorded conversation as chat history
       (step 04), open its audio block, and hand the microphone over to it.
    4. Close the old session in the background.

While the new session is being opened (about a second), mic audio is held in a buffer instead of
being sent anywhere, then replayed to the new session together with the last
REPLAY_SECONDS of audio from before the switch. So speech that starts right at the switch isn't
lost.

Design: SessionSwitcher stands in for a single stream manager. AudioStreamer and run_conversation
from step 02 talk to it exactly as they would to one BedrockStreamManager, and it forwards to
whichever session is active. All sessions share one playback queue and barge-in counter, so audio
keeps playing smoothly across a switch.

Simplified from AWS's session-continuation sample (repeatable-patterns/session-continuation), which
additionally pre-opens the next session during a reply and recovers from sessions that fail to start.
Builds on step 02 (barge-in, no tools); reuses history helpers from step 04 and loudness from step 05.
"""
import asyncio
import collections
import importlib
import json
import math
import uuid

import boto3

from sonic.common import print_voices

# importlib because module names can't start with a digit in an import statement.
barge_in = importlib.import_module("02_audio_with_barge_in")
dual_mode = importlib.import_module("04_audio_with_text_dual_mode_input")
turn_taking = importlib.import_module("05_audio_with_turn_taking")
BedrockStreamManager = barge_in.BedrockStreamManager
build_parser = barge_in.build_parser
debug_print = barge_in.debug_print
run_conversation = barge_in.run_conversation
append_message = dual_mode.append_message
history_for_model = dual_mode.history_for_model
rms = turn_taking.rms
SPEECH_RMS_THRESHOLD = turn_taking.SPEECH_RMS_THRESHOLD

CONNECTION_LIMIT_SECONDS = 8 * 60
# Switch no later than this, even mid-reply, to stay clear of the 8-minute cut-off.
HARD_DEADLINE_SECONDS = 7 * 60 + 30
# Mic audio from just before the switch that is replayed to the new session.
REPLAY_SECONDS = 1.5
# How long the mic must be quiet before a moment counts as "nobody is speaking".
QUIET_SECONDS = 0.8
MONITOR_INTERVAL_SECONDS = 0.25
RETRY_AFTER_FAILURE_SECONDS = 10


def format_duration(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


class ContinuableStreamManager(BedrockStreamManager):
    """One session (stream). Shares playback state with the switcher and reports turn progress."""

    def __init__(self, *args, switcher, number, **kwargs):
        super().__init__(*args, **kwargs)
        self.switcher = switcher
        self.number = number
        self.started_at = None
        # Every session plays through the switcher's single queue, so audio a previous session
        # already delivered keeps playing after the switch.
        self.audio_output_queue = switcher.audio_output_queue
        # Turn tracking for "is now a quiet moment?". The assistant's text comes in pairs: a
        # SPECULATIVE block before each sentence is spoken and a FINAL block after. When every
        # speculative block has its final, the reply is complete and fully transcribed.
        self.speculative_blocks = 0
        self.final_blocks = 0
        self.awaiting_reply = False  # the user said something the model hasn't started answering
        self._block_role = None
        self._block_stage = None

    async def initialize_stream(self):
        self.started_at = asyncio.get_running_loop().time()
        await super().initialize_stream()

    def age(self):
        return asyncio.get_running_loop().time() - self.started_at

    def reply_complete(self):
        return not self.awaiting_reply and self.final_blocks >= self.speculative_blocks

    def _handle_barge_in(self):
        super()._handle_barge_in()
        # The shared counter is what AudioStreamer watches to stop mid-chunk.
        self.switcher.interrupt_count += 1
        # Sentences that were never spoken won't get a FINAL block; don't wait for them.
        self.final_blocks = self.speculative_blocks

    async def on_event(self, event_type, event):
        """Record the transcript and track whether a reply is in progress."""
        if event_type == "contentStart":
            content_start = event["contentStart"]
            self._block_role = content_start.get("role")
            fields = content_start.get("additionalModelFields")
            self._block_stage = json.loads(fields).get("generationStage") if fields else None
            if content_start.get("type") == "TEXT" and self._block_role == "ASSISTANT":
                self.awaiting_reply = False
                if self._block_stage == "SPECULATIVE":
                    self.speculative_blocks += 1
                elif self._block_stage == "FINAL":
                    self.final_blocks += 1
        elif event_type == "textOutput":
            text = event["textOutput"]["content"]
            if self._is_interruption_text(text):
                return
            if self._block_role == "USER":
                self.awaiting_reply = True
                append_message(self.switcher.transcript, "USER", text)
            elif self._block_role == "ASSISTANT" and self._block_stage == "FINAL":
                append_message(self.switcher.transcript, "ASSISTANT", text)

    async def send_history(self, messages):
        """Replay earlier turns as TEXT blocks; allowed once, before the audio block opens."""
        for message in messages:
            content_name = str(uuid.uuid4())
            await self.send_event({
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": message["role"],
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            })
            await self.send_event({
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": message["content"],
                }
            })
            await self.send_event({"contentEnd": {"promptName": self.prompt_name, "contentName": content_name}})


class SessionSwitcher:
    """Looks like one stream manager to AudioStreamer/run_conversation; swaps real sessions underneath."""

    def __init__(self, switch_after, **session_kwargs):
        self.switch_after = switch_after
        self.session_kwargs = session_kwargs  # passed to every new session (credentials, voice, ...)
        self.loop = asyncio.get_running_loop()

        # Shared by all sessions; AudioStreamer reads these.
        self.audio_output_queue = asyncio.Queue()
        self.interrupt_count = 0

        self.transcript = []  # the whole conversation across sessions, replayed on each switch
        self.active = None
        self.session_count = 0
        self.switching = False
        self.closing = False
        # Rolling window of the most recent mic chunks (512 frames = 32 ms each), and the chunks
        # captured while a switch is in progress.
        self.recent_audio = collections.deque(maxlen=math.ceil(REPLAY_SECONDS / 0.032))
        self.held_audio = collections.deque()
        self.last_speech_time = 0.0
        self.monitor_task = None
        self.background_closes = set()
        # Completes when the conversation really ends (not on a planned switch); run_conversation
        # waits on it the way it waits on a single session's response_task.
        self.response_task = self.loop.create_future()

    # ---- the stream-manager interface used by step 02 ----

    async def initialize_stream(self):
        self.active = await self._open_session()
        self.monitor_task = asyncio.create_task(self._monitor())

    async def send_audio_content_start(self):
        await self.active.send_audio_content_start()

    async def send_audio_chunk(self, audio_bytes):
        self.recent_audio.append(audio_bytes)
        if rms(audio_bytes) >= SPEECH_RMS_THRESHOLD:
            self.last_speech_time = self.loop.time()
        if self.switching:
            self.held_audio.append(audio_bytes)  # replayed to the new session once it's ready
            return
        await self.active.send_audio_chunk(audio_bytes)

    async def close(self):
        self.closing = True
        if self.monitor_task:
            self.monitor_task.cancel()
            await asyncio.gather(self.monitor_task, return_exceptions=True)
        await self.active.close()
        await asyncio.gather(*self.background_closes, return_exceptions=True)

    # ---- switching ----

    async def _open_session(self):
        self.session_count += 1
        session = ContinuableStreamManager(switcher=self, number=self.session_count, **self.session_kwargs)
        await session.initialize_stream()
        session.response_task.add_done_callback(lambda _task, s=session: self._on_session_ended(s))
        return session

    def _on_session_ended(self, session):
        # A session we closed on purpose is no longer active; only the active one ending is fatal.
        if session is self.active and not self.closing and not self.response_task.done():
            print(f"[session] session {session.number} closed unexpectedly; ending the conversation")
            self.response_task.set_result(None)

    def _quiet_moment(self):
        return (
            self.active.reply_complete()
            and self.audio_output_queue.empty()
            and self.loop.time() - self.last_speech_time >= QUIET_SECONDS
        )

    async def _monitor(self):
        """Check a few times per second whether it's time to move to a new session."""
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            age = self.active.age()
            if age < self.switch_after:
                continue
            if self._quiet_moment():
                reason = "quiet moment"
            elif age >= HARD_DEADLINE_SECONDS:
                reason = "deadline reached, switching mid-turn"
            else:
                continue
            try:
                await self._switch(reason)
            except Exception as e:
                # _switch already fell back to the old session; try again shortly.
                print(f"[session] switch failed ({e}); retrying in {RETRY_AFTER_FAILURE_SECONDS}s")
                await asyncio.sleep(RETRY_AFTER_FAILURE_SECONDS)

    def _history_for_new_session(self):
        messages = history_for_model(self.transcript)
        # Replayed history must start with a USER turn; trimming can leave an ASSISTANT turn first.
        while messages and messages[0]["role"] != "USER":
            messages.pop(0)
        return messages

    async def _switch(self, reason):
        old = self.active
        print(f"[session] session {old.number} is {format_duration(old.age())} old ({reason}); "
              f"opening session {self.session_count + 1}...")
        # From here on, mic audio is held instead of going to the old session.
        self.switching = True
        self.held_audio.extend(self.recent_audio)
        replayed_seconds = len(self.held_audio) * 0.032

        new = None
        try:
            new = await self._open_session()
            history = self._history_for_new_session()
            await new.send_history(history)
            await new.send_audio_content_start()

            # Replay held audio. More may arrive while we await, so drain until empty; the switch
            # below then happens with no await in between, so no chunk can slip in unsent.
            while self.held_audio:
                await new.send_audio_chunk(self.held_audio.popleft())
        except BaseException:
            # Failed, or cancelled because the user quit: drop the half-open session and fall back
            # to the old one, which stays usable until the 8-minute limit.
            self.held_audio.clear()
            self.switching = False
            if new is not None:
                await new.close()
            raise
        self.active = new
        self.switching = False

        print(f"[session] session {new.number} active: {len(history)} history messages, "
              f"{replayed_seconds:.1f}s of audio replayed")
        task = asyncio.create_task(old.close())
        self.background_closes.add(task)
        task.add_done_callback(self.background_closes.discard)


async def main():
    parser = build_parser("Amazon Nova 2 Sonic console client that continues past the 8-minute limit")
    parser.add_argument(
        "--switch-after",
        type=int,
        default=360,
        metavar="SECONDS",
        help="Move to a new session at the first quiet moment after this many seconds "
        f"(default 360; must be 20-{HARD_DEADLINE_SECONDS - 30}). Use e.g. 30 to see switches quickly",
    )
    args = parser.parse_args()
    barge_in.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return
    if not 20 <= args.switch_after <= HARD_DEADLINE_SECONDS - 30:
        parser.error(f"--switch-after must be between 20 and {HARD_DEADLINE_SECONDS - 30} seconds")

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")
    print(f"Sessions switch after {format_duration(args.switch_after)} at a quiet moment "
          f"(connection limit {format_duration(CONNECTION_LIMIT_SECONDS)})")

    await run_conversation(SessionSwitcher(
        switch_after=args.switch_after,
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
    ))


if __name__ == "__main__":
    asyncio.run(main())
