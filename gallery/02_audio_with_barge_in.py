"""Step 2: barge-in. Interrupt the assistant mid-answer; queued audio is dropped. Adds --debug event logging.

Run:  uv run gallery/02_audio_with_barge_in.py [--debug]     (use headphones)

Read gallery/01_audio_plain.py first for the event protocol; this step reuses it.

What barge-in means
-------------------
Nova Sonic generates speech much faster than real time, so when it answers, several seconds of
audio arrive within a moment and sit in our playback queue. If you start talking over it, the
server notices, stops generating, and tells us so. But the audio already queued here would keep
playing for seconds unless we throw it away. That client-side part is what this file adds:

    1. Detect the interruption. The server signals it two ways, and we handle both:
         - a textOutput whose content is the JSON  { "interrupted" : true }
         - a contentEnd with stopReason "INTERRUPTED"
    2. Drop every queued audio chunk (_handle_barge_in).
    3. Stop the chunk that is playing right now, mid-sentence (the interrupt_count check in
       AudioStreamer._play_output_audio).

Other changes from step 01
--------------------------
- Mic capture uses a PyAudio *callback* (PortAudio calls us from its own thread every 32 ms)
  instead of a blocking read loop, and hands chunks to asyncio through a queue so order is kept.
- The code is split into BedrockStreamManager (protocol) and AudioStreamer (devices).
- BedrockStreamManager has two extension points used by later steps:
  tool_configuration() (step 03 returns real tools) and on_event() (steps 03 and 04 react to
  extra events). run_conversation() lets step 04 run without a microphone and stop on /quit.
- The program also ends if the server closes the stream, e.g. at the 8-minute connection limit.
"""
import asyncio
import base64
import datetime
import json
import time
import uuid

import boto3
import pyaudio
from aws_sdk_bedrock_runtime.client import InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)

from sonic.common import (
    ASSISTANT_COLOR,
    RESET_COLOR,
    USER_COLOR,
    CHANNELS,
    DEFAULT_MODEL_ID,
    DEFAULT_VOICE,
    FORMAT,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    build_arg_parser,
    create_bedrock_client,
    default_system_prompt,
    print_voices,
    wait_for_enter,
)

# Frames per audio buffer: 512 frames at 16 kHz = 32 ms. Smaller than step 01 for snappier barge-in.
CHUNK_SIZE = 512

# How long close() waits for the server to end the stream before giving up.
CLOSE_TIMEOUT_SECONDS = 2

# Set by --debug. Later steps set it through the module (barge_in.DEBUG = ...).
DEBUG = False


def debug_print(message):
    """Print a timestamped line, only with --debug. Useful to see the event order and timing."""
    if DEBUG:
        print(f"{datetime.datetime.now():%H:%M:%S.%f}"[:-3] + f" [debug] {message}")


class BedrockStreamManager:
    """Owns the bidirectional Bedrock stream: sends events, parses responses, tracks barge-in."""

    def __init__(
        self,
        session,
        model_id=DEFAULT_MODEL_ID,
        region="us-east-1",
        voice_id=DEFAULT_VOICE,
        system_prompt=None,
    ):
        self.session = session
        self.model_id = model_id
        self.region = region
        self.voice_id = voice_id
        self.system_prompt = system_prompt or default_system_prompt(voice_id)

        self.client = None
        self.stream = None
        self.response_task = None  # background task running _process_responses()
        self.is_active = False
        # Several tasks send events (mic audio, tool results, typed text); the lock makes sure one
        # event is fully written to the stream before the next starts.
        self._send_lock = asyncio.Lock()

        self.audio_output_queue = asyncio.Queue()  # decoded model audio waiting to be played
        # Incremented on every barge-in so playback can drop audio dequeued before the interruption.
        self.interrupt_count = 0

        self.role = None  # role of the content block currently being received (USER / ASSISTANT / TOOL)
        self.display_assistant_text = False

        # promptName ties the whole conversation together; each contentName names one content block.
        self.prompt_name = str(uuid.uuid4())
        self.system_content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

    async def initialize_stream(self):
        """Open the stream, send session/prompt setup and the system prompt, start receiving.

        Same events as step 01, plus toolUseOutputConfiguration/toolConfiguration in promptStart.
        Here the tool list is empty; step 03 overrides tool_configuration() to add tools.
        """
        self.client = create_bedrock_client(self.session, self.region)

        start = time.perf_counter()
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        debug_print(f"Stream opened in {time.perf_counter() - start:.3f}s")
        self.is_active = True

        await self.send_event({
            "sessionStart": {
                "inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7}
            }
        })
        await self.send_event({
            "promptStart": {
                "promptName": self.prompt_name,
                "textOutputConfiguration": {"mediaType": "text/plain"},
                "audioOutputConfiguration": {
                    "mediaType": "audio/lpcm",
                    "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                    "voiceId": self.voice_id,
                    "encoding": "base64",
                    "audioType": "SPEECH",
                },
                "toolUseOutputConfiguration": {"mediaType": "application/json"},
                "toolConfiguration": self.tool_configuration(),
            }
        })
        await self.send_event({
            "contentStart": {
                "promptName": self.prompt_name,
                "contentName": self.system_content_name,
                "type": "TEXT",
                "interactive": True,
                "role": "SYSTEM",
                "textInputConfiguration": {"mediaType": "text/plain"},
            }
        })
        await self.send_event({
            "textInput": {
                "promptName": self.prompt_name,
                "contentName": self.system_content_name,
                "content": self.system_prompt,
            }
        })
        await self.send_event({
            "contentEnd": {"promptName": self.prompt_name, "contentName": self.system_content_name}
        })

        self.response_task = asyncio.create_task(self._process_responses())
        debug_print("Stream initialized")

    def tool_configuration(self):
        """Tools offered to the model in promptStart. None here; step 03 overrides this."""
        return {"tools": []}

    async def on_event(self, event_type, event):
        """Hook for subclasses; called after built-in handling of every response event."""

    async def send_event(self, event):
        """Send one JSON event over the stream (see step 01 for the SDK wrapper classes)."""
        if not self.is_active:
            return
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps({"event": event}).encode("utf-8"))
        )
        async with self._send_lock:
            await self.stream.input_stream.send(chunk)
        if DEBUG and "audioInput" not in event:
            debug_print(f"Sent {next(iter(event))}")

    async def send_audio_content_start(self):
        """Open the USER audio content block that all mic chunks belong to."""
        await self.send_event({
            "contentStart": {
                "promptName": self.prompt_name,
                "contentName": self.audio_content_name,
                "type": "AUDIO",
                "interactive": True,
                "role": "USER",
                "audioInputConfiguration": {
                    "mediaType": "audio/lpcm",
                    "sampleRateHertz": INPUT_SAMPLE_RATE,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                    "audioType": "SPEECH",
                    "encoding": "base64",
                },
            }
        })

    async def send_audio_chunk(self, audio_bytes):
        await self.send_event({
            "audioInput": {
                "promptName": self.prompt_name,
                "contentName": self.audio_content_name,
                "content": base64.b64encode(audio_bytes).decode("utf-8"),
            }
        })

    def _handle_barge_in(self):
        """The user interrupted: discard all queued audio and tell playback to stop its current chunk.

        Emptying the queue removes audio not yet started. The chunk currently being played was
        already taken out of the queue, so playback compares interrupt_count before and after
        each small write and stops when it changes.
        """
        self.interrupt_count += 1
        dropped = 0
        while not self.audio_output_queue.empty():
            self.audio_output_queue.get_nowait()
            dropped += 1
        debug_print(f"Barge-in: dropped {dropped} queued audio chunks")

    @staticmethod
    def _is_interruption_text(text):
        """True for the special assistant text '{ "interrupted" : true }' Nova Sonic sends on barge-in."""
        if not text.lstrip().startswith("{"):
            return False
        try:
            return bool(json.loads(text).get("interrupted"))
        except (json.JSONDecodeError, AttributeError):
            return False

    async def _process_responses(self):
        """Receive and dispatch the model's events until the stream ends.

        Each event is a JSON object with a single key naming its type, e.g. {"textOutput": {...}}.
        Events this method doesn't know are passed on to on_event() for subclasses.
        """
        try:
            while self.is_active:
                # (initial_response, event_receiver); receive() waits for the next event. See step 01.
                output = await self.stream.await_output()
                result = await output[1].receive()
                if not (result.value and result.value.bytes_):
                    continue

                event = json.loads(result.value.bytes_.decode("utf-8")).get("event", {})
                event_type = next(iter(event), None)  # the single top-level key, e.g. "audioOutput"

                if event_type == "contentStart":
                    content_start = event["contentStart"]
                    self.role = content_start.get("role")
                    stage = None
                    if "additionalModelFields" in content_start:
                        stage = json.loads(content_start["additionalModelFields"]).get("generationStage")
                    # Assistant text arrives twice (SPECULATIVE then FINAL); print only the speculative pass.
                    self.display_assistant_text = stage == "SPECULATIVE"
                    debug_print(f"contentStart type={content_start.get('type')} role={self.role} stage={stage}")

                elif event_type == "textOutput":
                    text = event["textOutput"]["content"]
                    if self._is_interruption_text(text):
                        print("(interrupted)")
                        self._handle_barge_in()
                    elif self.role == "ASSISTANT" and self.display_assistant_text:
                        print(f"{ASSISTANT_COLOR}Assistant: {text}{RESET_COLOR}")
                    elif self.role == "USER":
                        print(f"{USER_COLOR}User: {text}{RESET_COLOR}")

                elif event_type == "audioOutput":
                    await self.audio_output_queue.put(base64.b64decode(event["audioOutput"]["content"]))

                elif event_type == "contentEnd":
                    # stopReason: END_TURN (finished), PARTIAL_TURN, INTERRUPTED (barge-in), TOOL_USE.
                    stop_reason = event["contentEnd"].get("stopReason")
                    debug_print(f"contentEnd type={event['contentEnd'].get('type')} stopReason={stop_reason}")
                    if stop_reason == "INTERRUPTED":
                        self._handle_barge_in()

                elif event_type == "usageEvent":
                    # Running token totals for the session (speech and text); useful for cost tracking.
                    usage = event["usageEvent"]
                    debug_print(
                        f"usage: input={usage.get('totalInputTokens')} output={usage.get('totalOutputTokens')}"
                    )

                elif event_type is not None:
                    debug_print(f"Received {event_type}")

                await self.on_event(event_type, event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Typically the server ending the stream (8-minute limit, timeout, error event).
            # Returning ends this task, which run_conversation() treats as "session over".
            if self.is_active:
                print(f"Stream closed: {e}")

    async def close(self):
        """Close the audio block, prompt and session in protocol order, then stop receiving."""
        if not self.is_active:
            return
        try:
            await self.send_event({
                "contentEnd": {"promptName": self.prompt_name, "contentName": self.audio_content_name}
            })
            await self.send_event({"promptEnd": {"promptName": self.prompt_name}})
            await self.send_event({"sessionEnd": {}})
            await self.stream.input_stream.close()
        except Exception as e:
            print(f"Error while closing the session: {e}")
        finally:
            self.is_active = False
            if self.response_task and not self.response_task.done():
                # Let the server end the stream first: cancelling a receive that is still waiting
                # makes awscrt print a traceback when the server's last events arrive.
                try:
                    await asyncio.wait_for(asyncio.shield(self.response_task), CLOSE_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    self.response_task.cancel()
                    await asyncio.gather(self.response_task, return_exceptions=True)


class AudioStreamer:
    """Streams microphone audio (or silence) to Bedrock and plays responses, stopping playback on barge-in.

    Two tasks: _send_mic_audio (mic queue -> Bedrock) and _play_output_audio (Bedrock queue -> speaker).
    With use_microphone=False (step 04 text mode) the mic task is replaced by _send_silence.
    """

    def __init__(self, stream_manager, use_microphone=True):
        self.stream_manager = stream_manager
        # Saved so the PortAudio callback thread can schedule work on this event loop.
        self.loop = asyncio.get_running_loop()
        self.mic_queue = asyncio.Queue()
        self.is_streaming = False
        self.tasks = []

        self.p = pyaudio.PyAudio()
        self.input_stream = None
        if use_microphone:
            # Callback mode: PortAudio records in the background and calls _input_callback with each
            # 32 ms buffer. start=False so recording only begins in start(), after the audio block opens.
            self.input_stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=INPUT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._input_callback,
                start=False,
            )
        self.output_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )

    def _input_callback(self, in_data, frame_count, time_info, status):
        """Called by PortAudio on its own thread for every recorded buffer; must return quickly."""
        # Runs on PortAudio's thread; hand the chunk to the event loop in order.
        # asyncio objects aren't thread-safe, so call_soon_threadsafe runs put_nowait on the loop's
        # thread. A FIFO queue keeps chunks in recording order (separate tasks per chunk would not).
        if self.is_streaming and in_data:
            self.loop.call_soon_threadsafe(self.mic_queue.put_nowait, in_data)
        # (output data, flag): no output for an input stream; paContinue keeps recording.
        return (None, pyaudio.paContinue)

    async def _send_mic_audio(self):
        """Forward recorded mic chunks to Bedrock, in order, for the whole session."""
        while True:
            audio_bytes = await self.mic_queue.get()
            await self.stream_manager.send_audio_chunk(audio_bytes)

    async def _send_silence(self):
        # The session needs continuous audio even when the user only types; send silence in real time.
        silence = b"\x00" * (CHUNK_SIZE * 2)
        interval = CHUNK_SIZE / INPUT_SAMPLE_RATE
        while True:
            await self.stream_manager.send_audio_chunk(silence)
            await asyncio.sleep(interval)

    async def _play_output_audio(self):
        """Play model audio from the queue; abandon the current chunk as soon as a barge-in happens."""
        manager = self.stream_manager
        pending = None
        try:
            while True:
                audio_data = await manager.audio_output_queue.get()
                # Remember the barge-in counter for this chunk. Each audioOutput can be a second or
                # more of speech, so it is written in 512-byte slices (~10 ms) and we check between
                # slices: if the counter moved, the user interrupted, so stop mid-chunk.
                generation = manager.interrupt_count
                for i in range(0, len(audio_data), CHUNK_SIZE):
                    if manager.interrupt_count != generation:
                        break
                    # write() blocks until the device accepts the slice; run it in a worker thread.
                    pending = self.loop.run_in_executor(
                        None, self.output_stream.write, audio_data[i:i + CHUNK_SIZE]
                    )
                    await pending
        finally:
            # PortAudio must not be closed while a worker thread is still writing to it.
            if pending and not pending.done():
                await asyncio.wait([pending])

    async def start(self):
        """Open the audio content block, start recording, and launch the send and play tasks."""
        await self.stream_manager.send_audio_content_start()
        self.is_streaming = True
        if self.input_stream:
            self.input_stream.start_stream()
            sender = self._send_mic_audio()
        else:
            sender = self._send_silence()
        self.tasks = [
            asyncio.create_task(sender),
            asyncio.create_task(self._play_output_audio()),
        ]

    async def stop(self):
        """Stop recording first (no more callbacks), then the tasks, then the devices, then the stream."""
        self.is_streaming = False
        if self.input_stream and self.input_stream.is_active():
            self.input_stream.stop_stream()

        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

        if self.input_stream:
            self.input_stream.close()
        self.output_stream.stop_stream()
        self.output_stream.close()
        self.p.terminate()

        await self.stream_manager.close()


def build_parser(description):
    """Common options from sonic.common plus --debug."""
    parser = build_arg_parser(description)
    parser.add_argument("--debug", action="store_true", help="Print event-level debug logging")
    return parser


async def wait_for_enter_to_stop():
    """Default stop condition for run_conversation(): the user presses Enter."""
    print("Starting audio streaming. Speak into your microphone (you can interrupt the assistant)...")
    print("Press Enter to stop...")
    await wait_for_enter()


async def run_conversation(stream_manager, use_microphone=True, until=wait_for_enter_to_stop):
    """Stream audio and play responses until `until()` returns, Ctrl+C, or the server ends the session."""
    await stream_manager.initialize_stream()
    audio_streamer = AudioStreamer(stream_manager, use_microphone=use_microphone)
    until_task = None
    try:
        await audio_streamer.start()
        until_task = asyncio.create_task(until())
        # Whichever finishes first: the user asks to stop, or the response task ends because the
        # server closed the stream.
        await asyncio.wait([until_task, stream_manager.response_task], return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # Ctrl+C: asyncio.run() cancels main(); catching it here lets the cleanup below run.
        print("\nInterrupted, shutting down...")
    finally:
        if until_task and not until_task.done():
            until_task.cancel()
        await audio_streamer.stop()
        print("Session ended")


async def main():
    global DEBUG
    args = build_parser("Amazon Nova 2 Sonic speech-to-speech console client with barge-in").parse_args()
    DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")

    await run_conversation(BedrockStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
    ))


if __name__ == "__main__":
    asyncio.run(main())
