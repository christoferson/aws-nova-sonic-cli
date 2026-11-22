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

from sonic_common import (
    CHANNELS,
    DEFAULT_MODEL_ID,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VOICE,
    FORMAT,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    build_arg_parser,
    create_bedrock_client,
    print_voices,
    wait_for_enter,
)

CHUNK_SIZE = 512

DEBUG = False


def debug_print(message):
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
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ):
        self.session = session
        self.model_id = model_id
        self.region = region
        self.voice_id = voice_id
        self.system_prompt = system_prompt

        self.client = None
        self.stream = None
        self.response_task = None
        self.is_active = False
        self._send_lock = asyncio.Lock()

        self.audio_output_queue = asyncio.Queue()
        # Incremented on every barge-in so playback can drop audio dequeued before the interruption.
        self.interrupt_count = 0

        self.role = None
        self.display_assistant_text = False

        self.prompt_name = str(uuid.uuid4())
        self.system_content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

    async def initialize_stream(self):
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
                "toolConfiguration": {"tools": []},
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

    async def send_event(self, event):
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
        self.interrupt_count += 1
        dropped = 0
        while not self.audio_output_queue.empty():
            self.audio_output_queue.get_nowait()
            dropped += 1
        debug_print(f"Barge-in: dropped {dropped} queued audio chunks")

    @staticmethod
    def _is_interruption_text(text):
        if not text.lstrip().startswith("{"):
            return False
        try:
            return bool(json.loads(text).get("interrupted"))
        except (json.JSONDecodeError, AttributeError):
            return False

    async def _process_responses(self):
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()
                if not (result.value and result.value.bytes_):
                    continue

                event = json.loads(result.value.bytes_.decode("utf-8")).get("event", {})
                event_type = next(iter(event), None)

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
                        print(f"Assistant: {text}")
                    elif self.role == "USER":
                        print(f"User: {text}")

                elif event_type == "audioOutput":
                    await self.audio_output_queue.put(base64.b64decode(event["audioOutput"]["content"]))

                elif event_type == "contentEnd":
                    stop_reason = event["contentEnd"].get("stopReason")
                    debug_print(f"contentEnd type={event['contentEnd'].get('type')} stopReason={stop_reason}")
                    if stop_reason == "INTERRUPTED":
                        self._handle_barge_in()

                elif event_type == "usageEvent":
                    usage = event["usageEvent"]
                    debug_print(
                        f"usage: input={usage.get('totalInputTokens')} output={usage.get('totalOutputTokens')}"
                    )

                elif event_type is not None:
                    debug_print(f"Received {event_type}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.is_active:
                print(f"Stream closed: {e}")

    async def close(self):
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
            debug_print(f"Error while closing stream: {e}")
        finally:
            self.is_active = False
            if self.response_task and not self.response_task.done():
                self.response_task.cancel()
                await asyncio.gather(self.response_task, return_exceptions=True)


class AudioStreamer:
    """Streams microphone audio to Bedrock and plays responses, stopping playback on barge-in."""

    def __init__(self, stream_manager):
        self.stream_manager = stream_manager
        self.loop = asyncio.get_running_loop()
        self.mic_queue = asyncio.Queue()
        self.is_streaming = False
        self.tasks = []

        self.p = pyaudio.PyAudio()
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
        # Runs on PortAudio's thread; hand the chunk to the event loop in order.
        if self.is_streaming and in_data:
            self.loop.call_soon_threadsafe(self.mic_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    async def _send_mic_audio(self):
        while True:
            audio_bytes = await self.mic_queue.get()
            await self.stream_manager.send_audio_chunk(audio_bytes)

    async def _play_output_audio(self):
        manager = self.stream_manager
        pending = None
        try:
            while True:
                audio_data = await manager.audio_output_queue.get()
                generation = manager.interrupt_count
                for i in range(0, len(audio_data), CHUNK_SIZE):
                    if manager.interrupt_count != generation:
                        break
                    pending = self.loop.run_in_executor(
                        None, self.output_stream.write, audio_data[i:i + CHUNK_SIZE]
                    )
                    await pending
        finally:
            # PortAudio must not be closed while a worker thread is still writing to it.
            if pending and not pending.done():
                await asyncio.wait([pending])

    async def start(self):
        await self.stream_manager.send_audio_content_start()
        self.is_streaming = True
        self.input_stream.start_stream()
        self.tasks = [
            asyncio.create_task(self._send_mic_audio()),
            asyncio.create_task(self._play_output_audio()),
        ]
        print("Starting audio streaming. Speak into your microphone (you can interrupt the assistant)...")
        print("Press Enter to stop...")

    async def stop(self):
        self.is_streaming = False
        if self.input_stream.is_active():
            self.input_stream.stop_stream()

        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

        self.input_stream.close()
        self.output_stream.stop_stream()
        self.output_stream.close()
        self.p.terminate()

        await self.stream_manager.close()


async def main():
    global DEBUG
    parser = build_arg_parser("Amazon Nova 2 Sonic speech-to-speech console client with barge-in")
    parser.add_argument("--debug", action="store_true", help="Print event-level debug logging")
    args = parser.parse_args()
    DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")

    stream_manager = BedrockStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
    )
    await stream_manager.initialize_stream()

    audio_streamer = AudioStreamer(stream_manager)
    enter_task = None
    try:
        await audio_streamer.start()
        enter_task = asyncio.create_task(wait_for_enter())
        # Stop on Enter, or if the server closes the stream (e.g. session time limit).
        await asyncio.wait([enter_task, stream_manager.response_task], return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        print("\nInterrupted, shutting down...")
    finally:
        if enter_task and not enter_task.done():
            enter_task.cancel()
        await audio_streamer.stop()
        print("Session ended")


if __name__ == "__main__":
    asyncio.run(main())
