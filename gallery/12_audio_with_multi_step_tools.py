"""Step 12: multi-step tool workflow. The assistant chains tools to handle a product return.

Run:  uv run gallery/12_audio_with_multi_step_tools.py
Try:  "I'd like to return the socks from order 1001.", "Can I return my rain jacket? Order 1002.",
      "Return the lantern from order 1003.", "Where is order 1004?"
Compare with Nova 2 Sonic:
      uv run gallery/12_audio_with_multi_step_tools.py --model-id amazon.nova-2-sonic-v1:0

Agentic voice workflows
-----------------------
Steps 03 and 09 answer a question with one tool call. Real voice agents usually have to do several
dependent steps in one conversation, which is what Nova 2.5 Sonic's improved reasoning and tool
calling are aimed at. The model card's own example: "look up an order, check eligibility, and take
action in one session". Here the store "Nova Outfitters" has three tools:

    lookup_order(order_id)                         what was ordered, when, and its status
    check_return_eligibility(order_id, item)       can it be returned, and for how much
    create_return(order_id, item, reason)          the action: creates the return and refund

A typical return takes several turns and three tool calls:

    user:  "I want to return the socks from order 1001"
    model: toolUse lookup_order {"order_id": "1001"}                    -> toolResult (items, dates)
    model: toolUse check_return_eligibility {"order_id": "1001", ...}  -> toolResult (eligible, $18)
    model: "The wool socks qualify for an $18 refund. Shall I start the return?"
    user:  "yes"
    model: toolUse create_return {...}                                  -> toolResult (return ID)
    model: "Done. Your return number is R-1001-1..."

Each toolUse is answered by step 03's code, so nothing new is needed at the event level. What
makes it work is the system prompt and the tool design:

- The prompt sets the order of steps and says to confirm before acting. Without that rule the
  model may call create_return straight after the eligibility check.
- Tool results carry the facts (prices, dates, reasons), so the model never has to guess them, and
  is told never to invent order details.
- The action tool is defensive: create_return checks eligibility again itself, so a skipped or
  misremembered step can't create a return that shouldn't exist. The Nova docs recommend this:
  validate tool input, and return a clear error the model can explain to the user.

Sample orders (made-up data, in memory; returns you create last until the program exits):
    1001  delivered 3 days ago     trail running shoes, wool socks        returnable
    1002  delivered 48 days ago    rain jacket                             past the 30-day window
    1003  delivered 7 days ago     camping lantern (final sale), bottle    lantern not returnable
    1004  shipped, not delivered   hiking backpack                         not returnable yet

Builds on step 03 (tool use) by overriding tool_configuration() and execute_tool().
"""
import asyncio
import datetime
import importlib
import json
import re

import boto3

from sonic.common import default_system_prompt, print_voices

# Builds on steps 2 and 3; importlib because module names can't start with a digit in an import statement.
barge_in = importlib.import_module("02_audio_with_barge_in")
tool_use = importlib.import_module("03_audio_with_tools")
build_parser = barge_in.build_parser
run_conversation = barge_in.run_conversation
ToolUseStreamManager = tool_use.ToolUseStreamManager

RETURN_WINDOW_DAYS = 30

# Shown at startup, so the user knows what to try.
SAMPLE_ORDERS_SUMMARY = (
    "I'm the returns assistant for Nova Outfitters. Sample orders to try: 1001 (recent), "
    "1002 (over 30 days old), 1003 (has a final-sale item), 1004 (not delivered yet)."
)

WORKFLOW_PROMPT_SUFFIX = (
    " You are the returns assistant for Nova Outfitters, an online outdoor store. "
    "To handle a return: first ask for the order number if you don't have it, then call "
    "lookup_order, then call check_return_eligibility for the item the customer means. "
    "If the order has several items and it's unclear which one, ask. "
    "Before calling create_return, tell the customer the item and refund amount and wait for a clear yes. "
    "Never invent order details, prices, dates or policies; use only what the tools return. "
    "If a tool returns an error or says the item is not eligible, explain why in plain words. "
    "While a tool is running, briefly tell the customer what you are checking."
)


def _days_ago(days):
    return datetime.date.today() - datetime.timedelta(days=days)


# Dates are relative to today so the sample orders behave the same whenever you run this.
ORDERS = {
    "1001": {
        "status": "delivered",
        "ordered": _days_ago(6),
        "delivered": _days_ago(3),
        "items": [
            {"name": "Trail running shoes", "price": 89.00},
            {"name": "Wool socks (3-pack)", "price": 18.00},
        ],
    },
    "1002": {
        "status": "delivered",
        "ordered": _days_ago(52),
        "delivered": _days_ago(48),
        "items": [{"name": "Rain jacket", "price": 120.00}],
    },
    "1003": {
        "status": "delivered",
        "ordered": _days_ago(10),
        "delivered": _days_ago(7),
        "items": [
            {"name": "Camping lantern", "price": 35.00, "final_sale": True},
            {"name": "Insulated water bottle", "price": 15.00},
        ],
    },
    "1004": {
        "status": "shipped",
        "ordered": _days_ago(2),
        "delivered": None,
        "items": [{"name": "Hiking backpack", "price": 140.00}],
    },
}

# Returns created this run: (order_id, item name) -> return ID.
RETURNS = {}


def normalize_order_id(value):
    """Keep the digits, so "order #1001" or "1 0 0 1" from speech still match."""
    return re.sub(r"\D", "", str(value))


def find_order(order_id):
    order_id = normalize_order_id(order_id)
    order = ORDERS.get(order_id)
    if order is None:
        return order_id, None, {"error": f"No order found with number '{order_id}'. Ask the customer to check it."}
    return order_id, order, None


def find_item(order, item):
    """Match the item the customer named against the order, case-insensitively and by partial name."""
    wanted = str(item).lower().strip()
    matches = [i for i in order["items"] if wanted and (wanted in i["name"].lower() or i["name"].lower() in wanted)]
    if len(matches) == 1:
        return matches[0], None
    names = [i["name"] for i in order["items"]]
    if not matches:
        return None, {"error": f"No item matching '{item}' in this order.", "items_in_order": names}
    return None, {"error": f"'{item}' matches several items; ask which one.", "items_in_order": names}


def lookup_order(order_id):
    order_id, order, error = find_order(order_id)
    if error:
        return error
    return {
        "order_id": order_id,
        "status": order["status"],
        "order_date": order["ordered"].isoformat(),
        "delivery_date": order["delivered"].isoformat() if order["delivered"] else None,
        "items": [
            {
                "name": i["name"],
                "price_usd": i["price"],
                "final_sale": i.get("final_sale", False),
                "returned": (order_id, i["name"]) in RETURNS,
            }
            for i in order["items"]
        ],
        "total_usd": sum(i["price"] for i in order["items"]),
    }


def check_return_eligibility(order_id, item):
    order_id, order, error = find_order(order_id)
    if error:
        return error
    match, error = find_item(order, item)
    if error:
        return error

    result = {"order_id": order_id, "item": match["name"], "refund_usd": match["price"]}
    if (order_id, match["name"]) in RETURNS:
        return {**result, "eligible": False, "reason": f"A return already exists: {RETURNS[(order_id, match['name'])]}."}
    if order["delivered"] is None:
        return {**result, "eligible": False, "reason": "The order hasn't been delivered yet; returns open after delivery."}
    if match.get("final_sale"):
        return {**result, "eligible": False, "reason": "Final-sale items can't be returned."}
    deadline = order["delivered"] + datetime.timedelta(days=RETURN_WINDOW_DAYS)
    if datetime.date.today() > deadline:
        return {**result, "eligible": False,
                "reason": f"The {RETURN_WINDOW_DAYS}-day return window ended on {deadline.isoformat()}."}
    return {**result, "eligible": True, "return_deadline": deadline.isoformat()}


def create_return(order_id, item, reason):
    # Defensive: re-check eligibility here instead of trusting that the model did it first.
    eligibility = check_return_eligibility(order_id, item)
    if "error" in eligibility:
        return eligibility
    if not eligibility["eligible"]:
        return {"error": f"Return not created: {eligibility['reason']}"}
    order_id, name = eligibility["order_id"], eligibility["item"]
    return_id = f"R-{order_id}-{len(RETURNS) + 1}"
    RETURNS[(order_id, name)] = return_id
    return {
        "return_id": return_id,
        "order_id": order_id,
        "item": name,
        "reason": reason,
        "refund_usd": eligibility["refund_usd"],
        "refund_method": "original payment method, within 5 business days of receiving the item",
        "shipping_label": "emailed to the customer",
    }


# name -> (handler, description, JSON schema for the arguments)
WORKFLOW_TOOLS = {
    "lookup_order": (
        lookup_order,
        "Look up an order by its order number: status, dates, items with prices, and which items "
        "were already returned. Call this first for any question about an order.",
        {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order number, e.g. '1001'"}},
            "required": ["order_id"],
        },
    ),
    "check_return_eligibility": (
        check_return_eligibility,
        "Check whether one item from an order can be returned, why not if it can't, and the refund amount. "
        "Call this after lookup_order and before offering a return.",
        {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order number"},
                "item": {"type": "string", "description": "The item name as listed in the order"},
            },
            "required": ["order_id", "item"],
        },
    ),
    "create_return": (
        create_return,
        "Create a return and refund for one item. Only call this after the customer has clearly "
        "confirmed the item and refund amount.",
        {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order number"},
                "item": {"type": "string", "description": "The item name as listed in the order"},
                "reason": {"type": "string", "description": "Why the customer is returning it, in a few words"},
            },
            "required": ["order_id", "item", "reason"],
        },
    ),
}


class WorkflowStreamManager(ToolUseStreamManager):
    """Offers the three returns tools, with a prompt that sets the order of steps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not kwargs.get("system_prompt"):
            # Replace step 03's hint (date/weather/calculator) with the returns workflow.
            self.system_prompt = default_system_prompt(self.voice_id) + WORKFLOW_PROMPT_SUFFIX

    def tool_configuration(self):
        return {"tools": [
            {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    # As in sonic/tools.py: inputSchema.json is a JSON string, not an object.
                    "inputSchema": {"json": json.dumps(schema)},
                }
            }
            for name, (_handler, description, schema) in WORKFLOW_TOOLS.items()
        ]}

    async def execute_tool(self, name, arguments):
        if name not in WORKFLOW_TOOLS:
            return {"error": f"Unknown tool '{name}'"}
        handler, _description, schema = WORKFLOW_TOOLS[name]
        missing = [p for p in schema["required"] if not str(arguments.get(p, "")).strip()]
        if missing:
            return {"error": f"Missing required arguments: {', '.join(missing)}"}
        return handler(**{p: arguments[p] for p in schema["properties"] if p in arguments})


async def main():
    parser = build_parser("Amazon Nova 2.5 Sonic console client with a multi-step tool workflow")
    args = parser.parse_args()
    barge_in.DEBUG = args.debug

    if args.list_voices:
        print_voices()
        return

    print(SAMPLE_ORDERS_SUMMARY)
    session = boto3.Session(profile_name=args.profile)
    print(f"Using AWS profile '{session.profile_name}' in region {args.region}, voice '{args.voice}'")

    await run_conversation(WorkflowStreamManager(
        session=session,
        model_id=args.model_id,
        region=args.region,
        voice_id=args.voice,
        system_prompt=args.system_prompt,
    ))


if __name__ == "__main__":
    asyncio.run(main())
