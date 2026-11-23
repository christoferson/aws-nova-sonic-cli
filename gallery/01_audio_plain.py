"""Step 1: basic speech-to-speech. Mic in, voice out, transcripts printed; the full event protocol in one file.

Run:  uv run gallery/01_audio_plain.py      (speak, then press Enter to stop)

How Nova 2 Sonic works
----------------------
Unlike a normal request/response API, Nova Sonic uses ONE long-lived bidirectional stream
(InvokeModelWithBidirectionalStream). Both sides send JSON "events" over it at the same time:

    you  -> sessionStart                      inference settings (maxTokens, temperature, ...)
    you  -> promptStart                       output settings: voice, audio format, text format
    you  -> contentStart / textInput / contentEnd     the system prompt (role SYSTEM)
    you  -> contentStart (AUDIO, role USER)   opens the microphone channel
    you  -> audioInput, audioInput, ...       small chunks of mic audio, base64 encoded, nonstop
    model-> contentStart / textOutput / contentEnd    transcript of what you said (role USER)
    model-> contentStart / textOutput / contentEnd    what it plans to say (role ASSISTANT)
    model-> audioOutput, audioOutput, ...     the spoken answer, base64 encoded
    you  -> contentEnd / promptEnd / sessionEnd       shut down

Every piece of content is wrapped in contentStart ... contentEnd and identified by a
`contentName` (a UUID you choose). The whole conversation shares one `promptName`.
You never "send a question": you just keep streaming audio, and the model decides when you
have finished speaking (turn detection) and answers.

This file shows that protocol with the simplest possible structure: three asyncio tasks
(send mic audio, receive events, play audio) connected by a queue. Step 02 adds barge-in.
"""
import asyncio
import base64
import json
import uuid

import boto3
import pyaudio
# The experimental AWS SDK for Python (aws_sdk_bedrock_runtime, built on "smithy") is used instead
# of boto3 because boto3 does not support bidirectional streaming. boto3 is only used for credentials.
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

# Frames per PyAudio read/write. 1024 frames of 16 kHz mono 16-bit audio = 2048 bytes = 64 ms.
CHUNK_SIZE = 1024


class SimpleNovaSonic:
    """One Nova Sonic conversation: opens the stream, sends events, and receives the model's events."""

    def __init__(
        self,
        session,
        model_id=DEFAULT_MODEL_ID,
        region="us-east-1",
        voice_id=DEFAULT_VOICE,
        system_prompt=None,
    ):
        self.session = session  # boto3.Session, only used to resolve AWS credentials
        self.model_id = model_id
        self.region = region
        self.voice_id = voice_id
        self.system_prompt = system_prompt or default_system_prompt(voice_id)
        self.client = None
        self.stream = None  # the bidirectional stream, created in start_session()
        self.response = None  # background task running _process_responses()
        self.is_active = False
        # Identifiers you choose. promptName ties all events of this conversation together;
        # each contentName identifies one content block (system prompt, mic audio, ...).
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        # Decoded audio from the model waits here until play_audio() plays it.
        self.audio_queue = asyncio.Queue()
        # Role and generation stage of the content block currently being received.
        self.role = None
        self.display_assistant_text = False

    def _initialize_client(self):
        self.client = create_bedrock_client(self.session, self.region)

    async def send_event(self, event):
        """Send one JSON event, e.g. {"audioInput": {...}}, over the stream.

        The SDK expects raw bytes wrapped in two classes: BidirectionalInputPayloadPart holds the
        bytes, and InvokeModelWithBidirectionalStreamInputChunk marks them as a "chunk" event of
        the input stream. Nova Sonic itself only sees the JSON inside.
        """
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps({"event": event}).encode("utf-8"))
        )
        await self.stream.input_stream.send(chunk)

    async def start_session(self):
        """Open the stream and send the setup events: session, prompt, and system prompt."""
        if not self.client:
            self._initialize_client()

        # Opens the HTTP/2 connection. Returns as soon as we can send; responses arrive later.
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True

        # 1. Session-wide inference settings.
        await self.send_event({
            "sessionStart": {
                "inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7}
            }
        })

        # 2. How the model should answer: plain-text transcripts plus 24 kHz 16-bit mono PCM speech.
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
            }
        })

        # 3. The system prompt, as a TEXT content block with role SYSTEM. Every content block
        #    follows the same three-event pattern: contentStart, the content itself, contentEnd.
        await self.send_event({
            "contentStart": {
                "promptName": self.prompt_name,
                "contentName": self.content_name,
                "type": "TEXT",
                "interactive": False,
                "role": "SYSTEM",
                "textInputConfiguration": {"mediaType": "text/plain"},
            }
        })
        await self.send_event({
            "textInput": {
                "promptName": self.prompt_name,
                "contentName": self.content_name,
                "content": self.system_prompt,
            }
        })
        await self.send_event({
            "contentEnd": {"promptName": self.prompt_name, "contentName": self.content_name}
        })

        # Start reading the model's events in the background; they arrive while we keep sending audio.
        self.response = asyncio.create_task(self._process_responses())

    async def start_audio_input(self):
        """Open the microphone content block. It stays open for the whole conversation."""
        await self.send_event({
            "contentStart": {
                "promptName": self.prompt_name,
                "contentName": self.audio_content_name,
                "type": "AUDIO",
                # interactive=True: live user input the model should respond to.
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
        """Send raw 16 kHz 16-bit mono PCM mic audio, base64 encoded because events are JSON."""
        if not self.is_active:
            return
        await self.send_event({
            "audioInput": {
                "promptName": self.prompt_name,
                "contentName": self.audio_content_name,
                "content": base64.b64encode(audio_bytes).decode("utf-8"),
            }
        })

    async def end_audio_input(self):
        await self.send_event({
            "contentEnd": {"promptName": self.prompt_name, "contentName": self.audio_content_name}
        })

    async def end_session(self):
        """Close the conversation in protocol order (prompt, then session), then close the stream."""
        if not self.is_active:
            return
        await self.send_event({"promptEnd": {"promptName": self.prompt_name}})
        await self.send_event({"sessionEnd": {}})
        await self.stream.input_stream.close()

    async def _process_responses(self):
        """Receive the model's events until the stream ends: print transcripts, queue audio."""
        try:
            while self.is_active:
                # await_output() returns (initial_response, event_receiver); the SDK caches it,
                # so calling it every iteration is cheap. receive() waits for the next event.
                output = await self.stream.await_output()
                result = await output[1].receive()
                # Normal events carry JSON bytes in result.value.bytes_. Error events (throttling,
                # validation, timeout) have no bytes_, so they raise here and end the loop below.
                if not (result.value and result.value.bytes_):
                    continue

                event = json.loads(result.value.bytes_.decode("utf-8")).get("event", {})

                if "contentStart" in event:
                    # A new block begins; remember whose it is so textOutput can be labelled.
                    content_start = event["contentStart"]
                    self.role = content_start["role"]
                    # Assistant text arrives twice (SPECULATIVE then FINAL); print only the speculative pass.
                    # SPECULATIVE = what it is about to say (arrives before the audio, so it reads in sync).
                    # FINAL = what was actually spoken, sent afterwards; it would duplicate the output.
                    if "additionalModelFields" in content_start:
                        fields = json.loads(content_start["additionalModelFields"])
                        self.display_assistant_text = fields.get("generationStage") == "SPECULATIVE"
                    else:
                        self.display_assistant_text = False

                elif "textOutput" in event:
                    # Role USER: speech recognition of your voice. Role ASSISTANT: the reply text.
                    text = event["textOutput"]["content"]
                    if self.role == "ASSISTANT" and self.display_assistant_text:
                        print(f"{ASSISTANT_COLOR}Assistant: {text}{RESET_COLOR}")
                    elif self.role == "USER":
                        print(f"{USER_COLOR}User: {text}{RESET_COLOR}")

                elif "audioOutput" in event:
                    # Audio arrives faster than real time, so it is queued and played at its own pace.
                    audio_bytes = base64.b64decode(event["audioOutput"]["content"])
                    await self.audio_queue.put(audio_bytes)
        except Exception as e:
            if self.is_active:
                print(f"Error processing responses: {e}")

    async def play_audio(self):
        """Play queued model audio on the default speaker until the session ends."""
        # PyAudio wraps PortAudio, a C library for audio devices. p.open() returns a blocking stream.
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        loop = asyncio.get_running_loop()
        pending = None
        try:
            while self.is_active:
                audio_data = await self.audio_queue.get()
                for i in range(0, len(audio_data), CHUNK_SIZE):
                    if not self.is_active:
                        break
                    # stream.write() blocks until the device accepts the audio, so it runs in a
                    # worker thread (run_in_executor) to keep the event loop free for other tasks.
                    pending = loop.run_in_executor(None, stream.write, audio_data[i:i + CHUNK_SIZE])
                    await pending
                    await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            # PortAudio must not be closed while a worker thread is still writing to it.
            if pending and not pending.done():
                await asyncio.wait([pending])
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Audio playing stopped.")

    async def capture_audio(self):
        """Read the default microphone and stream it to Nova Sonic until the session ends."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        print("Starting audio capture. Speak into your microphone...")
        print("Press Enter to stop...")

        await self.start_audio_input()
        loop = asyncio.get_running_loop()
        pending = None
        try:
            while self.is_active:
                # stream.read() blocks for ~64 ms until a chunk is recorded; run it in a thread.
                # exception_on_overflow=False: if we fall behind, drop audio instead of crashing.
                pending = loop.run_in_executor(
                    None, lambda: stream.read(CHUNK_SIZE, exception_on_overflow=False)
                )
                await self.send_audio_chunk(await pending)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error capturing audio: {e}")
        finally:
            if pending and not pending.done():
                await asyncio.wait([pending])
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Audio capture stopped.")
            await self.end_audio_input()


async def main():
    args = build_arg_parser("Amazon Nova 2 Sonic speech-to-speech console client (simple)").parse_args()
    if args.list_voices:
        print_voices()
        return
    # boto3 resolves credentials from your AWS profile (SSO, assume-role, env vars, ...).
    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")
    nova_client = SimpleNovaSonic(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
    )

    await nova_client.start_session()

    # Three things now run concurrently on one event loop: receiving (started in start_session),
    # playing, and capturing. asyncio switches between them whenever one is waiting.
    playback_task = asyncio.create_task(nova_client.play_audio())
    capture_task = asyncio.create_task(nova_client.capture_audio())

    try:
        await wait_for_enter()
    except asyncio.CancelledError:
        # Ctrl+C: asyncio.run() cancels main(); catching it here lets the cleanup below run.
        print("\nInterrupted, shutting down...")
    finally:
        # Order matters: stop audio first, then close the conversation, then stop receiving.
        tasks = [t for t in (playback_task, capture_task) if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            await nova_client.end_session()
        except Exception as e:
            print(f"Error ending session: {e}")
        nova_client.is_active = False

        if nova_client.response and not nova_client.response.done():
            nova_client.response.cancel()
            await asyncio.gather(nova_client.response, return_exceptions=True)

        print("Session ended")


if __name__ == "__main__":
    asyncio.run(main())
