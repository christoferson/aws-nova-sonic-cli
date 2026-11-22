"""Step 3: tool use. The model calls Python functions (date/time, weather, calculator) and speaks the results."""
import asyncio
import importlib
import json
import time
import uuid

import boto3

from sonic.common import print_voices
from sonic.tools import run_tool, tool_specs

# Builds on step 2; importlib because module names can't start with a digit in an import statement.
barge_in = importlib.import_module("02_audio_with_barge_in")
BedrockStreamManager = barge_in.BedrockStreamManager
build_parser = barge_in.build_parser
debug_print = barge_in.debug_print
run_conversation = barge_in.run_conversation

TOOL_PROMPT_SUFFIX = (
    " You have tools for the current date and time, the weather, and exact calculations. "
    "Use them instead of guessing. While a tool is running, briefly tell the user what you are checking."
)


class ToolUseStreamManager(BedrockStreamManager):
    """Adds tool use: answers each toolUse event by running the tool in the background."""

    def __init__(self, *args, **kwargs):
        custom_prompt = kwargs.get("system_prompt")
        super().__init__(*args, **kwargs)
        if not custom_prompt:
            self.system_prompt += TOOL_PROMPT_SUFFIX
        self.pending_tool_uses = {}
        self.tool_tasks = set()
        # Each result is three events; results of concurrent tools must not interleave.
        self._tool_result_lock = asyncio.Lock()

    def tool_configuration(self):
        return {"tools": tool_specs()}

    async def on_event(self, event_type, event):
        if event_type == "toolUse":
            tool_use = event["toolUse"]
            self.pending_tool_uses[tool_use["toolUseId"]] = (tool_use["toolName"], tool_use.get("content") or "{}")
        elif event_type == "contentEnd" and event["contentEnd"].get("type") == "TOOL":
            # Run tools only once the model has finished emitting the tool-use block.
            for tool_use_id, (name, content) in self.pending_tool_uses.items():
                task = asyncio.create_task(self._run_tool_and_respond(tool_use_id, name, content))
                self.tool_tasks.add(task)
                task.add_done_callback(self.tool_tasks.discard)
            self.pending_tool_uses.clear()

    async def _run_tool_and_respond(self, tool_use_id, name, content):
        try:
            arguments = json.loads(content)
            if not isinstance(arguments, dict):
                raise ValueError("tool input must be a JSON object")
        except ValueError as e:
            arguments, result = {}, {"error": f"Invalid tool input: {e}"}
        else:
            result = None

        print(f"[tool] {name}({json.dumps(arguments, ensure_ascii=False)})")
        start = time.perf_counter()
        if result is None:
            result = await run_tool(name, arguments)
        status = "error" if "error" in result else "ok"
        print(f"[tool] {name} -> {status} in {time.perf_counter() - start:.1f}s")
        debug_print(f"tool result: {json.dumps(result, ensure_ascii=False)}")

        # The model waits for a result after every toolUse, so always send one, even on error.
        await self._send_tool_result(tool_use_id, result)

    async def _send_tool_result(self, tool_use_id, result):
        content_name = str(uuid.uuid4())
        async with self._tool_result_lock:
            await self.send_event({
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": tool_use_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    },
                }
            })
            await self.send_event({
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            })
            await self.send_event({
                "contentEnd": {"promptName": self.prompt_name, "contentName": content_name}
            })

    async def close(self):
        for task in list(self.tool_tasks):
            task.cancel()
        await asyncio.gather(*self.tool_tasks, return_exceptions=True)
        await super().close()


async def main():
    parser = build_parser("Amazon Nova 2 Sonic speech-to-speech console client with tool use")
    args = parser.parse_args()
    barge_in.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")
    print("Tools: get_date_and_time, get_weather, calculate")

    await run_conversation(ToolUseStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
    ))


if __name__ == "__main__":
    asyncio.run(main())
