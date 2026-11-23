"""Shared plumbing for every script: AWS client and credentials, audio format, voices, CLI options.

Why a custom credentials resolver: the Bedrock bidirectional-streaming client comes from the
experimental Smithy-based SDK (aws_sdk_bedrock_runtime), which only reads credentials from
environment variables out of the box. Boto3CredentialsResolver plugs boto3's credential chain
into it, so AWS profiles, SSO and assume-role work, with automatic refresh.
"""
import argparse
import asyncio
import os
import sys
import threading

import pyaudio
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.components import AWSCredentialsIdentity, AWSIdentityProperties
from smithy_core.aio.interfaces.identity import IdentityResolver
from smithy_core.exceptions import SmithyIdentityError

# Legacy console code pages (e.g. cp932) can't encode every character the model may output.
sys.stdout.reconfigure(errors="replace")

# Transcript colors (ANSI 256-color). os.system("") switches older Windows consoles into ANSI mode.
os.system("")
USER_COLOR = "\033[38;5;51m"  # neon cyan
ASSISTANT_COLOR = "\033[38;5;201m"  # neon magenta
RESET_COLOR = "\033[0m"

# Nova 2 Sonic (amazon.nova-2-sonic-v1:0) still works via --model-id.
DEFAULT_MODEL_ID = "amazon.nova-2-5-sonic"
DEFAULT_VOICE = "tiffany"

# Nova Sonic audio: 16-bit signed little-endian mono PCM ("LPCM"). Mic input at 16 kHz,
# model speech at 24 kHz. paInt16 makes PyAudio record/play exactly that sample format.
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16

# https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-language-support.html
VOICES = {
    "matthew": "en-US, masculine, polyglot (speaks all supported languages)",
    "tiffany": "en-US, feminine, polyglot (speaks all supported languages)",
    "amy": "en-GB, feminine",
    "olivia": "en-AU, feminine",
    "kiara": "en-IN / hi-IN, feminine",
    "arjun": "en-IN / hi-IN, masculine",
    "ambre": "fr-FR, feminine",
    "florian": "fr-FR, masculine",
    "beatrice": "it-IT, feminine",
    "lorenzo": "it-IT, masculine",
    "tina": "de-DE, feminine",
    "lennart": "de-DE, masculine",
    "lupe": "es-US, feminine",
    "carlos": "es-US, masculine",
    "carolina": "pt-BR, feminine",
    "leo": "pt-BR, masculine",
}


def default_system_prompt(voice_id=DEFAULT_VOICE):
    # Based on https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-system-prompts.html
    # Gender must match the voice for languages with gender agreement (French, Hindi, Spanish, ...).
    gender = "female" if "feminine" in VOICES.get(voice_id, "") else "male"
    return (
        f"You are a warm, professional, and helpful {gender} AI voice assistant. "
        "You are talking with the user out loud in real time: they hear your voice and you hear theirs. "
        "Give accurate answers that sound natural, direct, and human. Start by answering the user's "
        "question clearly in 1-2 sentences. Then, expand only enough to make the answer understandable, "
        "staying within 3-5 short sentences total. Avoid sounding like a lecture or essay. "
        "Always reply in the language the user is speaking, and do not mix in English unless the user "
        "speaks English."
    )


class Boto3CredentialsResolver(IdentityResolver[AWSCredentialsIdentity, AWSIdentityProperties]):
    """Resolves credentials via boto3 (profiles, SSO, assume-role, env vars), refreshing as needed."""

    def __init__(self, session):
        self._credentials = session.get_credentials()
        if self._credentials is None:
            raise SmithyIdentityError(
                f"No AWS credentials found for profile '{session.profile_name}'. "
                f"If it is an SSO profile, run: aws sso login --profile {session.profile_name}"
            )

    async def get_identity(self, *, properties: AWSIdentityProperties) -> AWSCredentialsIdentity:
        """Called by the SDK whenever it signs a request; returns the current key/secret/token."""
        # get_frozen_credentials() refreshes expiring temporary credentials (SSO, assume-role)
        # and returns a consistent snapshot, so key and token always belong together.
        frozen = self._credentials.get_frozen_credentials()
        return AWSCredentialsIdentity(
            access_key_id=frozen.access_key,
            secret_access_key=frozen.secret_key,
            session_token=frozen.token,
        )


def create_bedrock_client(session, region):
    """Bedrock Runtime client from the Smithy-based SDK (the one that supports bidirectional streams)."""
    config = Config(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=Boto3CredentialsResolver(session),
    )
    return BedrockRuntimeClient(config=config)


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--profile", help="AWS profile name (defaults to $AWS_PROFILE, then 'default')")
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="Bedrock region (defaults to $AWS_REGION, then us-east-1)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=VOICES,
        metavar="VOICE",
        help=f"Output voice ID (default: {DEFAULT_VOICE}). See --list-voices",
    )
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    parser.add_argument(
        "--system-prompt",
        help="Replace the default prompt (a voice assistant matching the voice's gender, replying in the user's language)",
    )
    return parser


def print_voices():
    for voice_id, description in VOICES.items():
        print(f"{voice_id:<10} {description}")


async def wait_for_enter():
    """Return when Enter is pressed; raises CancelledError on Ctrl+C."""
    loop = asyncio.get_running_loop()
    pressed = asyncio.Event()

    def worker():
        try:
            input()
        except EOFError:
            pass
        loop.call_soon_threadsafe(pressed.set)

    # Daemon thread, not the default executor: a blocked input() there would stall interpreter exit after Ctrl+C.
    threading.Thread(target=worker, daemon=True).start()
    await pressed.wait()
