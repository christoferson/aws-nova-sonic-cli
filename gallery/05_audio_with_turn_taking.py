"""Step 5: turn-taking sensitivity. Control how long a pause ends your turn, and measure the result.

Run:  uv run gallery/05_audio_with_turn_taking.py --sensitivity HIGH     (or MEDIUM, LOW)
Try:  ask the same question with HIGH and with LOW and compare the "[turn]" timings.

Turn detection
--------------
You never tell Nova Sonic "I'm done talking". It listens to the audio stream and treats a long
enough pause as the end of your turn ("endpointing"). How long is configurable in sessionStart:

    "sessionStart": {
        "inferenceConfiguration": {...},
        "turnDetectionConfiguration": {"endpointingSensitivity": "HIGH" | "MEDIUM" | "LOW"}
    }

    HIGH    ~1.5 s pause   fastest replies; may cut in if you pause mid-sentence
    MEDIUM  ~1.75 s pause  balanced
    LOW     ~2.0 s pause   most patient; good when people pause to think

Measuring it
------------
To make the difference visible, this step estimates when you stopped speaking from the loudness
(RMS) of each mic chunk, and when the assistant's first audio of a reply arrives it prints:

    [turn] reply started 1.9s after you stopped speaking

That time is the endpointing pause plus model and network latency. The loudness check is a
rough heuristic: background noise or speaker echo can skew it, so use headphones in a quiet room.

Builds on step 02 (barge-in, no tools), so tool overhead doesn't affect the timings.
"""
import array
import asyncio
import importlib
import math

import boto3

from sonic.common import print_voices

# Builds on step 2; importlib because module names can't start with a digit in an import statement.
barge_in = importlib.import_module("02_audio_with_barge_in")
BedrockStreamManager = barge_in.BedrockStreamManager
build_parser = barge_in.build_parser
debug_print = barge_in.debug_print
run_conversation = barge_in.run_conversation

SENSITIVITY_PAUSES = {"HIGH": "~1.5 s", "MEDIUM": "~1.75 s", "LOW": "~2.0 s"}

# Mic chunks louder than this (RMS of 16-bit samples, 0-32767) count as speech. Typical speech
# is well above 1000 and a quiet room well below 300; raise it if noise triggers false timings.
SPEECH_RMS_THRESHOLD = 500


def rms(audio_bytes):
    """Loudness of a chunk of 16-bit PCM: the root mean square of its samples."""
    samples = array.array("h", audio_bytes)  # "h" = signed 16-bit, the mic's sample format
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class TurnTakingStreamManager(BedrockStreamManager):
    """Adds turnDetectionConfiguration to sessionStart and times each reply."""

    def __init__(self, *args, sensitivity="MEDIUM", **kwargs):
        super().__init__(*args, **kwargs)
        self.sensitivity = sensitivity
        self.loop = asyncio.get_running_loop()
        # When speech was last heard on the mic, and whether we're waiting for a reply's first audio.
        self.last_speech_time = None
        self.awaiting_reply_audio = False

    async def send_event(self, event):
        """Inject the turn-detection setting into sessionStart; send everything else unchanged.

        Step 02 builds sessionStart itself, so rather than adding another hook there, this
        intercepts the event on its way out.
        """
        if "sessionStart" in event:
            event = {
                "sessionStart": {
                    **event["sessionStart"],
                    "turnDetectionConfiguration": {"endpointingSensitivity": self.sensitivity},
                }
            }
            debug_print(f"sessionStart with endpointingSensitivity={self.sensitivity}")
        await super().send_event(event)

    async def send_audio_chunk(self, audio_bytes):
        """Track the last time the user was audible, then send the chunk as usual."""
        level = rms(audio_bytes)
        if level >= SPEECH_RMS_THRESHOLD:
            self.last_speech_time = self.loop.time()
        await super().send_audio_chunk(audio_bytes)

    async def on_event(self, event_type, event):
        """Time the gap between the end of the user's speech and the first audio of each reply."""
        if event_type == "contentStart":
            content_start = event["contentStart"]
            # Each spoken reply is an AUDIO content block with role ASSISTANT.
            if content_start.get("type") == "AUDIO" and content_start.get("role") == "ASSISTANT":
                self.awaiting_reply_audio = True
        elif event_type == "audioOutput" and self.awaiting_reply_audio:
            self.awaiting_reply_audio = False
            if self.last_speech_time is not None:
                gap = self.loop.time() - self.last_speech_time
                print(f"[turn] reply started {gap:.1f}s after you stopped speaking")
                # Don't reuse the same speech end for later replies (e.g. after a barge-in).
                self.last_speech_time = None


async def main():
    parser = build_parser("Amazon Nova 2 Sonic console client with configurable turn-taking sensitivity")
    parser.add_argument(
        "--sensitivity",
        type=str.upper,
        choices=tuple(SENSITIVITY_PAUSES),
        default="MEDIUM",
        help="How quickly a pause ends your turn: HIGH (~1.5 s), MEDIUM (~1.75 s, default), LOW (~2.0 s)",
    )
    args = parser.parse_args()
    barge_in.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")
    print(
        f"Turn-taking sensitivity: {args.sensitivity} "
        f"(answers after a {SENSITIVITY_PAUSES[args.sensitivity]} pause)"
    )

    await run_conversation(TurnTakingStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
        sensitivity=args.sensitivity,
    ))


if __name__ == "__main__":
    asyncio.run(main())
