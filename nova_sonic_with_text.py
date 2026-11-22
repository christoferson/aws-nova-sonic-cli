import asyncio
import json
import pathlib
import threading
import uuid

import boto3

import nova_sonic
from nova_sonic import build_parser, debug_print, run_conversation, wait_for_enter_to_stop
from nova_sonic_tool_use import ToolUseStreamManager
from sonic_common import print_voices

# Limits from https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-chat-history.html
MAX_HISTORY_BYTES = 200_000
MAX_MESSAGE_BYTES = 50_000
HISTORY_ROLES = ("USER", "ASSISTANT")
QUIT_COMMANDS = ("/quit", "/exit", "quit", "exit")

HISTORY_PROMPT_SUFFIX = (
    " The conversation so far includes messages from your earlier conversations with this user. "
    "Treat them as things you remember, and use them when the user refers to past conversations."
)


def read_history_file(path):
    """Read [{"role": "USER"|"ASSISTANT", "content": "..."}]; a missing file means a new, empty history."""
    file = pathlib.Path(path)
    if not file.exists():
        return []
    messages = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(messages, list):
        raise ValueError(f"{path}: expected a JSON list of messages")
    cleaned = []
    for i, message in enumerate(messages):
        role = str(message.get("role", "")).upper()
        content = str(message.get("content", "")).strip()
        if role not in HISTORY_ROLES:
            raise ValueError(f"{path}: message {i} has role '{role}', expected USER or ASSISTANT")
        if content:
            append_message(cleaned, role, content)
    return cleaned


def history_for_model(messages):
    """Newest messages that fit Nova Sonic's size limits; the file itself keeps everything."""
    kept, total = [], 0
    for message in reversed(messages):
        content = message["content"].encode("utf-8")[:MAX_MESSAGE_BYTES].decode("utf-8", "ignore")
        total += len(content.encode("utf-8"))
        if total > MAX_HISTORY_BYTES:
            break
        kept.append({"role": message["role"], "content": content})
    kept.reverse()
    if len(kept) < len(messages):
        print(f"Sending the newest {len(kept)} of {len(messages)} history messages (200KB limit)")
    return kept


def append_message(messages, role, text):
    # History must alternate USER/ASSISTANT, so merge consecutive messages from the same role.
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] += " " + text
    else:
        messages.append({"role": role, "content": text})


class TextStreamManager(ToolUseStreamManager):
    """Adds typed (cross-modal) input, chat history replay, and transcript recording."""

    def __init__(self, *args, history=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = history or []
        if self.history:
            # Without this the model tends to insist it has no memory of earlier sessions.
            self.system_prompt += HISTORY_PROMPT_SUFFIX
        self.transcript = []
        self._transcript_role = None
        self._transcript_stage = None

    async def initialize_stream(self):
        await super().initialize_stream()
        # History is allowed once, after the system prompt and before audio streaming starts.
        for message in self.history:
            await self._send_text_block(message["role"], message["content"], interactive=True)

    async def _send_text_block(self, role, text, interactive):
        content_name = str(uuid.uuid4())
        await self.send_event({
            "contentStart": {
                "promptName": self.prompt_name,
                "contentName": content_name,
                "type": "TEXT",
                "interactive": interactive,
                "role": role,
                "textInputConfiguration": {"mediaType": "text/plain"},
            }
        })
        await self.send_event({
            "textInput": {"promptName": self.prompt_name, "contentName": content_name, "content": text}
        })
        await self.send_event({"contentEnd": {"promptName": self.prompt_name, "contentName": content_name}})

    async def send_user_text(self, text):
        await self._send_text_block("USER", text, interactive=True)
        append_message(self.transcript, "USER", text)
        debug_print(f"Sent typed text: {text}")

    async def on_event(self, event_type, event):
        await super().on_event(event_type, event)
        if event_type == "contentStart":
            content_start = event["contentStart"]
            self._transcript_role = content_start.get("role")
            fields = content_start.get("additionalModelFields")
            self._transcript_stage = json.loads(fields).get("generationStage") if fields else None
        elif event_type == "textOutput":
            text = event["textOutput"]["content"]
            if self._is_interruption_text(text):
                return
            # Record what was actually said: user ASR text and the FINAL (not speculative) assistant text.
            if self._transcript_role == "USER":
                append_message(self.transcript, "USER", text)
            elif self._transcript_role == "ASSISTANT" and self._transcript_stage == "FINAL":
                append_message(self.transcript, "ASSISTANT", text)


def start_line_reader(loop):
    """Read stdin lines on a daemon thread into an asyncio queue; None marks EOF."""
    lines = asyncio.Queue()

    def worker():
        while True:
            try:
                line = input()
            except EOFError:
                loop.call_soon_threadsafe(lines.put_nowait, None)
                return
            loop.call_soon_threadsafe(lines.put_nowait, line)

    threading.Thread(target=worker, daemon=True).start()
    return lines


def make_text_loop(stream_manager, mode):
    async def text_loop():
        if mode == "mixed":
            print("Speak into your microphone, or type a message and press Enter.")
        else:
            print("Type a message and press Enter. The assistant answers out loud.")
        print("Type /quit (or press Ctrl+C) to end the session.")
        lines = start_line_reader(asyncio.get_running_loop())
        while True:
            line = await lines.get()
            if line is None:
                return
            text = line.strip()
            if not text:
                continue
            if text.lower() in QUIT_COMMANDS:
                return
            await stream_manager.send_user_text(text)

    return text_loop


def save_history(path, history, transcript):
    if not transcript:
        print(f"Nothing new to save; {path} left unchanged")
        return
    messages = [dict(m) for m in history]
    for message in transcript:
        append_message(messages, message["role"], message["content"])
    pathlib.Path(path).write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(messages)} messages to {path}")


async def main():
    parser = build_parser("Amazon Nova 2 Sonic console client with text input, chat history and tools")
    parser.add_argument(
        "--mode",
        choices=("mixed", "text", "audio"),
        default="mixed",
        help="mixed: speak or type (default); text: type only (no microphone); audio: speak only",
    )
    parser.add_argument(
        "--history",
        metavar="FILE",
        help="JSON chat history file: loaded at start if it exists, and updated with this conversation on exit",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Load --history but don't write this conversation back to it",
    )
    args = parser.parse_args()
    nova_sonic.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    history = []
    if args.history:
        try:
            history = read_history_file(args.history)
        except (ValueError, OSError) as e:
            print(f"Can't read chat history: {e}")
            return
        if history:
            print(f"Loaded {len(history)} messages from {args.history}")
        else:
            print(f"No history in {args.history} yet; it will be created on exit")

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}', mode '{args.mode}'")

    stream_manager = TextStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
        history=history_for_model(history),
    )
    until = wait_for_enter_to_stop if args.mode == "audio" else make_text_loop(stream_manager, args.mode)
    try:
        await run_conversation(stream_manager, use_microphone=args.mode != "text", until=until)
    finally:
        if args.history and not args.no_save:
            save_history(args.history, history, stream_manager.transcript)


if __name__ == "__main__":
    asyncio.run(main())
