"""Step 10: forced tool use. Control whether the model may, must, or must use one specific tool.

Run:  uv run gallery/10_audio_with_forced_tool_use.py                                  (always search the docs)
      uv run gallery/10_audio_with_forced_tool_use.py --tool-choice auto               (model decides)
      uv run gallery/10_audio_with_forced_tool_use.py --tool-choice any                (must use some tool)
      uv run gallery/10_audio_with_forced_tool_use.py --tool-choice get_date_and_time  (always check the time)
      uv run gallery/10_audio_with_forced_tool_use.py --strict                         (docs-only answers)
Try:  ask the same things in each mode, e.g. "hello", "what time is it?", "what is barge-in?",
      and compare the "[choice]" lines.

toolChoice
----------
Next to the tool list in promptStart, toolConfiguration takes a toolChoice:

    "toolConfiguration": {
        "tools": [...],
        "toolChoice": {"auto": {}}                        model decides (default)
                      {"any": {}}                         must call at least one tool at the start
                                                          of every response; it picks which
                      {"tool": {"name": "<tool>"}}        must call exactly this tool, once, at the
                                                          start of every response
    }

Why force it: "always look it up" behaviour. A support agent that must search the knowledge base
before every answer can't answer from memory and make things up. The cost: every response waits
for a tool call (slower, and more input tokens), even for small talk like "hello", which will
trigger a search for "hello". Compare the modes and you'll see that trade-off directly.

Forcing the search makes the model *look*, but not stay within what it found: it can still add
general knowledge, and step 09's prompt only covers on-topic questions. --strict replaces the
prompt with a hard rule: answer only from the search results, say when the docs don't cover
something, and decline off-topic questions. Prompts can't make this a guarantee, but forced search
plus a strict prompt is the standard way to keep a docs assistant grounded.

Builds on step 09 (knowledge base) and adds the date/time tool from sonic/tools.py, so there is
a choice to make.
"""
import asyncio
import importlib

import boto3

from sonic.common import default_system_prompt, print_voices
from sonic.tools import run_tool, tool_specs

# Builds on steps 2 and 9; importlib because module names can't start with a digit in an import statement.
barge_in = importlib.import_module("02_audio_with_barge_in")
knowledge = importlib.import_module("09_audio_with_knowledge_base")
build_parser = barge_in.build_parser
run_conversation = barge_in.run_conversation
KnowledgeBaseStreamManager = knowledge.KnowledgeBaseStreamManager

EXTRA_TOOLS = ("get_date_and_time",)
TOOL_NAMES = (knowledge.TOOL_NAME,) + EXTRA_TOOLS
# --strict: answer only from the documents (the date/time tool stays available).
STRICT_PROMPT_SUFFIX = (
    " You are a documentation assistant for {topic}. Answer ONLY with information from the "
    "{tool} results; never add facts from your own knowledge, even if you are sure they are true. "
    "If the results don't contain the answer, say it isn't covered in the documentation. "
    "If the question is not about {topic}, politely say you can only help with that topic; "
    "the one exception is the current date and time, which you may answer with get_date_and_time. "
    "Summarize in your own words; don't read out URLs, code, or markdown."
)

# Shown at startup so the user knows how the assistant will behave.
TOOL_CHOICE_DESCRIPTIONS = {
    "auto": "Mode: the assistant decides when to search the docs or check the time.",
    "any": "Mode: the assistant always searches the docs or checks the time before answering.",
    knowledge.TOOL_NAME: "Mode: the assistant always searches the docs before answering.",
    "get_date_and_time": "Mode: the assistant always checks the date and time before answering.",
}


def tool_choice_config(choice):
    """The toolChoice object for a --tool-choice value: auto, any, or a tool name."""
    if choice in ("auto", "any"):
        return {choice: {}}
    return {"tool": {"name": choice}}


class ForcedToolStreamManager(KnowledgeBaseStreamManager):
    """Step 09's knowledge base plus a date/time tool, with a configurable toolChoice."""

    def __init__(self, *args, tool_choice=knowledge.TOOL_NAME, strict=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_choice = tool_choice
        if not kwargs.get("system_prompt"):
            if strict:
                # Replace step 09's "answer from the results" with a hard rule. Forcing the search
                # (toolChoice) makes the model look; this tells it not to go beyond what it found.
                self.system_prompt = default_system_prompt(self.voice_id) + STRICT_PROMPT_SUFFIX.format(
                    topic=self.topic, tool=knowledge.TOOL_NAME
                )
            else:
                self.system_prompt += " You can also check the current date and time with get_date_and_time."
        self.replies = 0
        self.tools_this_reply = []

    def tool_configuration(self):
        config = super().tool_configuration()  # step 09: the search_knowledge_base tool
        config["tools"] += tool_specs(EXTRA_TOOLS)
        config["toolChoice"] = tool_choice_config(self.tool_choice)
        return config

    async def execute_tool(self, name, arguments):
        if name in EXTRA_TOOLS:
            return await run_tool(name, arguments, EXTRA_TOOLS)
        return await super().execute_tool(name, arguments)

    async def on_event(self, event_type, event):
        """Keep step 03's tool handling; after each spoken reply, report which tools it used."""
        await super().on_event(event_type, event)
        if event_type == "toolUse":
            self.tools_this_reply.append(event["toolUse"]["toolName"])
        elif (
            event_type == "contentEnd"
            and event["contentEnd"].get("type") == "AUDIO"
            and event["contentEnd"].get("stopReason") != "PARTIAL_TURN"
        ):
            self.replies += 1
            used = ", ".join(self.tools_this_reply) if self.tools_this_reply else "no tool"
            print(f"[choice] reply {self.replies} ({self.tool_choice}): {used}")
            self.tools_this_reply = []


async def main():
    parser = build_parser("Amazon Nova 2 Sonic console client demonstrating toolChoice")
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "any") + TOOL_NAMES,
        default=knowledge.TOOL_NAME,
        help=f"auto: model decides; any: must use some tool; or a tool name to force it "
        f"(default {knowledge.TOOL_NAME})",
    )
    parser.add_argument("--kb-dir", default=str(knowledge.DEFAULT_KB_DIR), metavar="DIR",
                        help="Folder of .md/.txt files to search (default: the saved Nova 2 Sonic docs)")
    parser.add_argument("--strict", action="store_true",
                        help="Answer only from the documents; decline general-knowledge and off-topic questions")
    parser.add_argument("--kb-topic", default=knowledge.DEFAULT_KB_TOPIC, metavar="TEXT",
                        help="What the documents are about; tells the model when to search")
    args = parser.parse_args()
    barge_in.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    knowledge_base = knowledge.KnowledgeBase(args.kb_dir)
    if not knowledge_base.passages:
        parser.error(f"no .md or .txt content found in {args.kb_dir}")
    knowledge.print_knowledge_base(args.kb_dir, args.kb_topic)

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")
    print(TOOL_CHOICE_DESCRIPTIONS.get(args.tool_choice, f"Mode: always uses {args.tool_choice} before answering."))
    if args.strict:
        print("Strict: answers only from the documentation; other questions are declined.")

    await run_conversation(ForcedToolStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
        knowledge_base=knowledge_base,
        topic=args.kb_topic,
        tool_choice=args.tool_choice,
        strict=args.strict,
    ))


if __name__ == "__main__":
    asyncio.run(main())
