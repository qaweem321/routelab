"""One prompt format for training, evaluation and serving, so they can never drift apart.

The untuned base model needs the full instructions. The fine-tuned model has learned the task, so it
gets a one-line prompt: about 4x fewer tokens per request, which is what makes CPU training practical.
"""

# Few-shot, with real values: a first version that wrote the format as "orders|tickets|..." was
# copied literally by the base model (0% valid outputs), which would have been an unfair baseline.
SYSTEM = """Route a support-desk request to an agent and the first tool it should call.
Reply with one line of JSON and nothing else.

Agents and their tools:
- "orders": "get_order", "search_orders", "cancel_order", "refund_order"
- "tickets": "get_ticket", "search_tickets", "create_ticket", "update_ticket"
- "knowledge": store policy questions, tool null
- "other": greetings and anything unrelated, tool null

Examples:
Where is ORD-1004? -> {"route":"orders","tool":"get_order"}
Refund $20 on ORD-1002 -> {"route":"orders","tool":"refund_order"}
Close TCK-2004 -> {"route":"tickets","tool":"update_ticket"}
How long do refunds take? -> {"route":"knowledge","tool":null}
Hello! -> {"route":"other","tool":null}"""
COMPACT_SYSTEM = "Route the support-desk request."


def messages(text: str, compact: bool = False) -> list[dict]:
    system = COMPACT_SYSTEM if compact else SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": text.strip()[:600]}]
