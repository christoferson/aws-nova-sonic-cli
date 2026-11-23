"""Step 7: the assistant speaks first. It greets you as soon as the session connects, like a phone agent.

Run:  uv run gallery/07_audio_with_assistant_speaks_first.py
      uv run gallery/07_audio_with_assistant_speaks_first.py --system-prompt "You are Sunny, the
          receptionist at Kyoto Ramen. Greet callers, take reservations, keep answers short."

"Model starts first"
--------------------
Nova Sonic only speaks in response to input, and normally that input is your voice. To make it
open the conversation, we send a typed USER message right after the audio stream starts (the
same cross-modal TEXT block as step 04):

    contentStart (TEXT, role USER, interactive true) -> textInput "<instruction>" -> contentEnd

The message is an instruction ("the caller just connected, greet them"), not something the user
said. The model answers it out loud, so the user hears a greeting first; the instruction itself
is never spoken. With barge-in (step 02) the user can talk over the greeting.

The greeting follows the system prompt, so the persona decides how it sounds: give a
receptionist prompt and you get a receptionist's greeting.

It must be sent after the audio content block opens: cross-modal text needs an active audio stream.

Builds on step 02 (barge-in, no tools).
"""
import asyncio
import importlib
import uuid

import boto3

from sonic.common import print_voices

# Builds on step 2; importlib because module names can't start with a digit in an import statement.
barge_in = importlib.import_module("02_audio_with_barge_in")
BedrockStreamManager = barge_in.BedrockStreamManager
build_parser = barge_in.build_parser
debug_print = barge_in.debug_print
run_conversation = barge_in.run_conversation

DEFAULT_GREETING_INSTRUCTION = (
    "The user has just joined the conversation and hasn't said anything yet. "
    "Greet them warmly in one short sentence and ask how you can help."
)


class SpeaksFirstStreamManager(BedrockStreamManager):
    """Sends a greeting instruction as soon as the audio stream is open, and times the greeting."""

    def __init__(self, *args, greeting_instruction=DEFAULT_GREETING_INSTRUCTION, **kwargs):
        super().__init__(*args, **kwargs)
        self.greeting_instruction = greeting_instruction
        self.loop = asyncio.get_running_loop()
        self.greeting_sent_at = None  # set when the instruction goes out; cleared at the first audio

    async def send_audio_content_start(self):
        """Open the mic's audio block as usual, then immediately prompt the model to speak."""
        await super().send_audio_content_start()
        await self._send_greeting_instruction()

    async def _send_greeting_instruction(self):
        content_name = str(uuid.uuid4())
        await self.send_event({
            "contentStart": {
                "promptName": self.prompt_name,
                "contentName": content_name,
                "type": "TEXT",
                "interactive": True,
                "role": "USER",
                "textInputConfiguration": {"mediaType": "text/plain"},
            }
        })
        await self.send_event({
            "textInput": {
                "promptName": self.prompt_name,
                "contentName": content_name,
                "content": self.greeting_instruction,
            }
        })
        await self.send_event({"contentEnd": {"promptName": self.prompt_name, "contentName": content_name}})
        self.greeting_sent_at = self.loop.time()
        debug_print("Greeting instruction sent")

    async def on_event(self, event_type, event):
        """Report how quickly the greeting started playing (time to first audio)."""
        if event_type == "audioOutput" and self.greeting_sent_at is not None:
            print(f"[greeting] assistant started speaking {self.loop.time() - self.greeting_sent_at:.1f}s after connecting")
            self.greeting_sent_at = None


async def main():
    parser = build_parser("Amazon Nova 2 Sonic console client where the assistant greets you first")
    parser.add_argument(
        "--greeting-instruction",
        default=DEFAULT_GREETING_INSTRUCTION,
        metavar="TEXT",
        help="Hidden instruction sent at connect time that makes the assistant speak first",
    )
    args = parser.parse_args()
    barge_in.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")

    await run_conversation(SpeaksFirstStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
        greeting_instruction=args.greeting_instruction,
    ))


if __name__ == "__main__":
    asyncio.run(main())
