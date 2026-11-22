"""Step 4: dual-mode input. Speak or type (cross-modal input), plus chat history to resume conversations.

Run:  uv run gallery/04_audio_with_text_dual_mode_input.py --history chat.json
      --mode mixed|text|audio   speak and/or type (default mixed)
      --tools                   also enable the step 03 tools (off by default)
      --no-save                 load --history without writing to it

Cross-modal input
-----------------
A typed message is just another USER content block, sent while audio keeps streaming:
    contentStart (type TEXT, role USER, interactive true) -> textInput -> contentEnd
The model answers it out loud exactly as if you had said it. Use a new contentName per message.

The session still needs continuous audio input, even if you only type. Without it the
connection times out, so --mode text streams silence instead of the microphone.

Chat history
------------
Earlier turns can be replayed once, right after the system prompt and before audio starts,
as TEXT blocks with role USER or ASSISTANT (interactive true). They must alternate
USER/ASSISTANT, each message must be under 50 KB, and all of them together under 200 KB.

To build that history we record the conversation as it happens: your speech-recognition text,
your typed messages, and the assistant's FINAL text (what it actually said, not the SPECULATIVE
preview). On exit it is appended to the --history file, so the next run can resume.
"""
import asyncio
import importlib
import json
import pathlib
import threading
import uuid

import boto3

from sonic.common import default_system_prompt, print_voices

# Builds on steps 2 and 3; importlib because module names can't start with a digit in an import statement.
barge_in = importlib.import_module("02_audio_with_barge_in")
tool_use = importlib.import_module("03_audio_with_tools")
build_parser = barge_in.build_parser
debug_print = barge_in.debug_print
run_conversation = barge_in.run_conversation
wait_for_enter_to_stop = barge_in.wait_for_enter_to_stop
ToolUseStreamManager = tool_use.ToolUseStreamManager

# Limits from https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-chat-history.html
MAX_HISTORY_BYTES = 200_000
MAX_MESSAGE_BYTES = 50_000
HISTORY_ROLES = ("USER", "ASSISTANT")
QUIT_COMMANDS = ("/quit", "/exit", "quit", "exit")

# Replayed history looks like earlier turns of *this* conversation, so without this hint the model
# insists it "doesn't remember previous conversations" when asked about them.
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
    # Walk backwards from the newest message so the oldest ones are dropped first.
    kept, total = [], 0
    for message in reversed(messages):
        # Limits are in bytes, not characters (Japanese text is 3 bytes per character in UTF-8).
        # Cut at the byte limit; "ignore" drops a character split in half by the cut.
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
    """Add text to a message list, merging it into the last message if the role is the same."""
    # History must alternate USER/ASSISTANT, so merge consecutive messages from the same role.
    # This also joins the sentence-by-sentence transcript pieces into one message per turn.
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] += " " + text
    else:
        messages.append({"role": role, "content": text})


class TextStreamManager(ToolUseStreamManager):
    """Adds typed (cross-modal) input, chat history replay, and transcript recording.

    Inherits barge-in from step 02 and tool handling from step 03; tools are switched off unless
    tools_enabled, by sending an empty tool list.
    """

    def __init__(self, *args, history=None, tools_enabled=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools_enabled = tools_enabled
        if not tools_enabled and not kwargs.get("system_prompt"):
            # Drop the tool hint step 03 appended; the model shouldn't mention tools it doesn't have.
            self.system_prompt = default_system_prompt(self.voice_id)
        self.history = history or []
        if self.history:
            # Without this the model tends to insist it has no memory of earlier sessions.
            self.system_prompt += HISTORY_PROMPT_SUFFIX
        self.transcript = []  # this session's messages, appended to the history file on exit
        # Role and generation stage of the block being received, tracked separately from the
        # base class so recording doesn't depend on its printing logic.
        self._transcript_role = None
        self._transcript_stage = None

    def tool_configuration(self):
        return super().tool_configuration() if self.tools_enabled else {"tools": []}

    async def initialize_stream(self):
        """Run the normal setup (step 02), then replay the chat history."""
        await super().initialize_stream()
        # History is allowed once, after the system prompt and before audio streaming starts.
        for message in self.history:
            await self._send_text_block(message["role"], message["content"], interactive=True)

    async def _send_text_block(self, role, text, interactive):
        """Send one TEXT content block (contentStart, textInput, contentEnd) with a fresh contentName."""
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
        """Send a typed message (the model replies out loud) and record it for the history file."""
        await self._send_text_block("USER", text, interactive=True)
        append_message(self.transcript, "USER", text)
        debug_print(f"Sent typed text: {text}")

    async def on_event(self, event_type, event):
        """Keep step 03's tool handling, and record transcript text for the history file."""
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
    # input() blocks, and there's no portable async stdin on Windows, so a thread reads lines and
    # hands them to the event loop with call_soon_threadsafe. Daemon: it won't keep the program
    # alive at exit while still waiting for a line.
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
    """Build the stop condition for run_conversation(): send typed lines until /quit or EOF."""
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
    """Write the full loaded history plus this session's transcript; skip if nothing new was said."""
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
        "--tools",
        action="store_true",
        help="Enable the step 03 tools (date/time, weather, calculator). Off by default",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Load --history but don't write this conversation back to it",
    )
    args = parser.parse_args()
    barge_in.DEBUG = args.debug

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
    print(
        f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}', "
        f"mode '{args.mode}', tools {'on' if args.tools else 'off'}"
    )

    stream_manager = TextStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
        # The model gets the newest messages that fit; `history` (all of it) is what gets saved.
        history=history_for_model(history),
        tools_enabled=args.tools,
    )
    until = wait_for_enter_to_stop if args.mode == "audio" else make_text_loop(stream_manager, args.mode)
    try:
        await run_conversation(stream_manager, use_microphone=args.mode != "text", until=until)
    finally:
        if args.history and not args.no_save:
            save_history(args.history, history, stream_manager.transcript)


if __name__ == "__main__":
    asyncio.run(main())
