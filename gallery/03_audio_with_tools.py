"""Step 3: tool use. The model calls Python functions (date/time, weather, calculator) and speaks the results.

Run:  uv run gallery/03_audio_with_tools.py [--debug]
Try:  "What time is it in New York?", "Will it rain in Osaka tomorrow?", "What's 15% of 2,400?"

How tool use works
------------------
    1. promptStart lists the tools: name, description, and a JSON schema for the arguments
       (sonic/tools.py builds these). The description is what makes the model pick a tool.
    2. When it needs one, the model sends a TOOL content block:
           contentStart (type TOOL) -> toolUse {toolName, toolUseId, content: '{"location": ...}'}
           -> contentEnd (type TOOL, stopReason TOOL_USE)
    3. We run the tool and reply with our own TOOL content block, tied to it by toolUseId:
           contentStart (role TOOL, toolResultInputConfiguration.toolUseId)
           -> toolResult {content: '{"temperature": ...}'} -> contentEnd
    4. The model turns the result into speech.

Nova 2 Sonic calls tools *asynchronously*: the conversation doesn't pause while a tool runs.
The model can keep talking and listening, and several tools can run at once. So each tool runs
in its own asyncio task here, and the mic and speaker keep going meanwhile.

Rule that matters: answer EVERY toolUse with a toolResult, even on errors. If one is missing,
the model waits for it and the conversation stalls.

Note: in testing, barge-in felt less responsive with tools configured than in step 02.
"""
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

# Appended to the default system prompt. The first sentence is needed: without it the model often
# answers from memory (e.g. guesses the time) instead of calling a tool. The second fills the
# silence while a slow tool (weather lookup) runs.
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
        # toolUseId -> (tool name, JSON arguments), collected until the TOOL block ends.
        self.pending_tool_uses = {}
        # Running tool tasks. asyncio only keeps weak references to tasks, so we hold them here
        # (otherwise they could be garbage-collected mid-run) and cancel them on close.
        self.tool_tasks = set()
        # Each result is three events; results of concurrent tools must not interleave.
        self._tool_result_lock = asyncio.Lock()

    def tool_configuration(self):
        """All tools from sonic/tools.py, as toolSpec entries for promptStart."""
        return {"tools": tool_specs()}

    async def on_event(self, event_type, event):
        """Collect toolUse requests, and start them when the model closes the TOOL block."""
        if event_type == "toolUse":
            # content is a JSON *string* with the arguments the model extracted from your speech.
            tool_use = event["toolUse"]
            self.pending_tool_uses[tool_use["toolUseId"]] = (tool_use["toolName"], tool_use.get("content") or "{}")
        elif event_type == "contentEnd" and event["contentEnd"].get("type") == "TOOL":
            # Run tools only once the model has finished emitting the tool-use block.
            # create_task (not await) so the receive loop keeps running while tools execute:
            # that is what keeps the conversation going during a slow tool.
            for tool_use_id, (name, content) in self.pending_tool_uses.items():
                task = asyncio.create_task(self._run_tool_and_respond(tool_use_id, name, content))
                self.tool_tasks.add(task)
                task.add_done_callback(self.tool_tasks.discard)
            self.pending_tool_uses.clear()

    async def _run_tool_and_respond(self, tool_use_id, name, content):
        """Parse the arguments, run the tool, and always send a result back (an error result if needed)."""
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
            # run_tool never raises: unknown tools, bad arguments and exceptions become {"error": ...}.
            result = await run_tool(name, arguments)
        status = "error" if "error" in result else "ok"
        print(f"[tool] {name} -> {status} in {time.perf_counter() - start:.1f}s")
        debug_print(f"tool result: {json.dumps(result, ensure_ascii=False)}")

        # The model waits for a result after every toolUse, so always send one, even on error.
        await self._send_tool_result(tool_use_id, result)

    async def _send_tool_result(self, tool_use_id, result):
        """Send the result as a TOOL content block; toolUseId tells the model which call it answers."""
        content_name = str(uuid.uuid4())
        # The lock keeps this block's three events together even if another tool finishes now.
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
                    # Like toolUse arguments, the result is a JSON string, not a nested object.
                    "content": json.dumps(result, ensure_ascii=False),
                }
            })
            await self.send_event({
                "contentEnd": {"promptName": self.prompt_name, "contentName": content_name}
            })

    async def close(self):
        """Cancel tools still running (their results can't be delivered anymore), then close the stream."""
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
