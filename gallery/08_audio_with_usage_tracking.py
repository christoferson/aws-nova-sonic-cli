"""Step 8: usage and cost tracking. See tokens and estimated cost per reply and per session.

Run:  uv run gallery/08_audio_with_usage_tracking.py

The usageEvent
--------------
While a session runs, Nova Sonic keeps sending usageEvent events with token counts:

    "usageEvent": {
        "details": {
            "delta": {"input": {"speechTokens", "textTokens"}, "output": {...}},   since the last event
            "total": {"input": {"speechTokens", "textTokens"}, "output": {...}}    since session start
        },
        "totalInputTokens", "totalOutputTokens", "totalTokens"
    }

Four kinds of tokens are billed at different rates:
    input speech   your microphone audio
    output speech  the assistant's voice (the most expensive)
    input text     system prompt, chat history, typed messages, tool results
    output text    the assistant's text and tool calls

What this step prints
---------------------
After each reply, the tokens and cost since the previous reply, plus the running session total:

    [usage] reply 2: in 310 speech / 12 text, out 455 speech / 61 text = $0.0065 | session $0.0158

and a breakdown when the session ends. The per-reply figure is the difference between two
"total" snapshots, so a usage event that arrives late is counted in the next reply. The session
total is exact. Try staying silent for a while and watch whether input speech tokens keep rising:
the mic streams continuously.

Prices are for us-east-1 and change over time; check https://aws.amazon.com/bedrock/pricing/.
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
debug_print = barge_in.debug_print
run_conversation = barge_in.run_conversation

# USD per 1,000 tokens, on-demand, us-east-1. From the AWS Price List API (offer "AmazonBedrock",
# product "Nova Sonic 2.0"), retrieved 2026-11-23.
PRICE_PER_1K_TOKENS = {
    ("input", "speechTokens"): 0.003,
    ("output", "speechTokens"): 0.012,
    ("input", "textTokens"): 0.00033,
    ("output", "textTokens"): 0.00275,
}
TOKEN_KINDS = tuple(PRICE_PER_1K_TOKENS)


def read_totals(usage_event):
    """Cumulative token counts from a usageEvent, as {(direction, kind): count}."""
    total = usage_event.get("details", {}).get("total", {})
    return {(direction, kind): total.get(direction, {}).get(kind, 0) for direction, kind in TOKEN_KINDS}


def cost(counts):
    return sum(counts[key] * PRICE_PER_1K_TOKENS[key] / 1000 for key in TOKEN_KINDS)


def describe(counts):
    return (
        f"in {counts[('input', 'speechTokens')]} speech / {counts[('input', 'textTokens')]} text, "
        f"out {counts[('output', 'speechTokens')]} speech / {counts[('output', 'textTokens')]} text"
    )


class UsageTrackingStreamManager(BedrockStreamManager):
    """Keeps the latest cumulative usage and prints a cost line after every spoken reply."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop = asyncio.get_running_loop()
        self.started_at = None
        self.totals = {key: 0 for key in TOKEN_KINDS}  # latest cumulative counts from the server
        self.totals_at_last_reply = dict(self.totals)
        self.replies = 0

    async def initialize_stream(self):
        self.started_at = self.loop.time()
        await super().initialize_stream()

    async def on_event(self, event_type, event):
        if event_type == "usageEvent":
            # "total" is cumulative, so keeping the latest snapshot is enough; no summing needed.
            self.totals = read_totals(event["usageEvent"])
        elif (
            event_type == "contentEnd"
            and event["contentEnd"].get("type") == "AUDIO"
            # A reply can be split into several AUDIO blocks; all but the last end with PARTIAL_TURN.
            and event["contentEnd"].get("stopReason") != "PARTIAL_TURN"
        ):
            # A spoken reply just finished: report what it cost. (Only the assistant's audio blocks
            # are received; our mic block is something we send.)
            self.replies += 1
            delta = {key: self.totals[key] - self.totals_at_last_reply[key] for key in TOKEN_KINDS}
            self.totals_at_last_reply = dict(self.totals)
            print(f"[usage] reply {self.replies}: {describe(delta)} = ${cost(delta):.4f} "
                  f"| session ${cost(self.totals):.4f}")

    async def close(self):
        await super().close()
        self.print_summary()

    def print_summary(self):
        minutes = (self.loop.time() - self.started_at) / 60 if self.started_at else 0
        print("\n[usage] session summary")
        for direction, kind in TOKEN_KINDS:
            count = self.totals[(direction, kind)]
            line_cost = count * PRICE_PER_1K_TOKENS[(direction, kind)] / 1000
            label = f"{direction} {kind.replace('Tokens', '')} tokens"
            print(f"  {label:<22} {count:>8}   ${line_cost:.4f}")
        total_cost = cost(self.totals)
        per_minute = f", ${total_cost / minutes:.4f}/min" if minutes >= 0.1 else ""
        print(f"  {'total':<22} {sum(self.totals.values()):>8}   ${total_cost:.4f} "
              f"over {minutes:.1f} min{per_minute}, {self.replies} replies")


async def main():
    args = build_parser("Amazon Nova 2 Sonic console client with token usage and cost tracking").parse_args()
    barge_in.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")
    if args.region != "us-east-1":
        print("Note: cost estimates use us-east-1 prices")

    await run_conversation(UsageTrackingStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
    ))


if __name__ == "__main__":
    asyncio.run(main())
