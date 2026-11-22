"""Step 1: basic speech-to-speech. Mic in, voice out, transcripts printed; the full event protocol in one file."""
import asyncio
import base64
import json
import uuid

import boto3
import pyaudio
from aws_sdk_bedrock_runtime.client import InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)

from sonic.common import (
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

CHUNK_SIZE = 1024


class SimpleNovaSonic:
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
        self.response = None
        self.is_active = False
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.audio_queue = asyncio.Queue()
        self.role = None
        self.display_assistant_text = False

    def _initialize_client(self):
        self.client = create_bedrock_client(self.session, self.region)

    async def send_event(self, event):
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=json.dumps({"event": event}).encode("utf-8"))
        )
        await self.stream.input_stream.send(chunk)

    async def start_session(self):
        if not self.client:
            self._initialize_client()

        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
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
            }
        })

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

        self.response = asyncio.create_task(self._process_responses())

    async def start_audio_input(self):
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
        if not self.is_active:
            return
        await self.send_event({"promptEnd": {"promptName": self.prompt_name}})
        await self.send_event({"sessionEnd": {}})
        await self.stream.input_stream.close()

    async def _process_responses(self):
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()
                if not (result.value and result.value.bytes_):
                    continue

                event = json.loads(result.value.bytes_.decode("utf-8")).get("event", {})

                if "contentStart" in event:
                    content_start = event["contentStart"]
                    self.role = content_start["role"]
                    # Assistant text arrives twice (SPECULATIVE then FINAL); print only the speculative pass.
                    if "additionalModelFields" in content_start:
                        fields = json.loads(content_start["additionalModelFields"])
                        self.display_assistant_text = fields.get("generationStage") == "SPECULATIVE"
                    else:
                        self.display_assistant_text = False

                elif "textOutput" in event:
                    text = event["textOutput"]["content"]
                    if self.role == "ASSISTANT" and self.display_assistant_text:
                        print(f"Assistant: {text}")
                    elif self.role == "USER":
                        print(f"User: {text}")

                elif "audioOutput" in event:
                    audio_bytes = base64.b64decode(event["audioOutput"]["content"])
                    await self.audio_queue.put(audio_bytes)
        except Exception as e:
            if self.is_active:
                print(f"Error processing responses: {e}")

    async def play_audio(self):
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

    playback_task = asyncio.create_task(nova_client.play_audio())
    capture_task = asyncio.create_task(nova_client.capture_audio())

    try:
        await wait_for_enter()
    except asyncio.CancelledError:
        print("\nInterrupted, shutting down...")
    finally:
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
