"""Step 11: phone-quality audio. Talk to Nova Sonic at 8 kHz, the sample rate of a phone call.

Run:  uv run gallery/11_audio_with_phone_quality.py               (8 kHz, like a phone line)
      uv run gallery/11_audio_with_phone_quality.py --rate 16000  (normal quality, to compare)

Sample rates
------------
Nova Sonic accepts audio at 8000, 16000 or 24000 Hz, set separately for each direction:

    promptStart.audioOutputConfiguration.sampleRateHertz    the assistant's voice
    contentStart.audioInputConfiguration.sampleRateHertz    your microphone

Phone networks carry 8 kHz audio, so a voice agent behind a phone number receives and sends 8 kHz.
Nova 2 Sonic's speech recognition was specifically improved for 8 kHz telephony input. At 8 kHz
only frequencies up to 4 kHz survive, which is why phone calls sound muffled: you'll hear that in
the assistant's voice, and it is what the model hears from you.

8 kHz is also half the data of 16 kHz: 16 KB per second of 16-bit mono audio instead of 32 KB.

In a real phone deployment, audio usually arrives as 8 kHz mu-law from a provider such as Amazon
Connect, Twilio or Vonage and is converted to 16-bit PCM before it is sent to Nova Sonic; key
presses (DTMF) can be sent as text with cross-modal input (step 04). Here we skip the phone network
and just record and play at 8 kHz.

How: step 02 reads the module-level INPUT_SAMPLE_RATE and OUTPUT_SAMPLE_RATE whenever it opens the
audio devices and builds the audio configuration events, so this step sets both before starting.
Builds on step 02 (barge-in, no tools).
"""
import asyncio
import importlib

import boto3

from sonic.common import print_voices

# Builds on step 2; importlib because module names can't start with a digit in an import statement.
barge_in = importlib.import_module("02_audio_with_barge_in")
BedrockStreamManager = barge_in.BedrockStreamManager
build_parser = barge_in.build_parser
run_conversation = barge_in.run_conversation

SUPPORTED_RATES = (8000, 16000, 24000)


async def main():
    parser = build_parser("Amazon Nova 2 Sonic console client with phone-quality (8 kHz) audio")
    parser.add_argument(
        "--rate",
        type=int,
        choices=SUPPORTED_RATES,
        default=8000,
        help="Sample rate in Hz for both your mic and the assistant's voice (default 8000, phone quality)",
    )
    args = parser.parse_args()
    barge_in.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    # Step 02 uses these for the PyAudio streams and the audio configuration it sends to Nova Sonic.
    barge_in.INPUT_SAMPLE_RATE = args.rate
    barge_in.OUTPUT_SAMPLE_RATE = args.rate

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")
    quality = "phone quality" if args.rate == 8000 else "normal quality"
    print(f"Audio: {args.rate // 1000} kHz in both directions ({quality})")

    await run_conversation(BedrockStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
    ))


if __name__ == "__main__":
    asyncio.run(main())
