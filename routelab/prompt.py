"""One prompt format for training, evaluation and serving, so they can never drift apart."""

SYSTEM = (
    "Route a support-desk request. Reply with JSON only: "
    '{"route": orders|tickets|knowledge|other, "tool": <first tool or null>}. '
    "Tools: orders: get_order, search_orders, cancel_order, refund_order; "
    "tickets: get_ticket, search_tickets, create_ticket, update_ticket. "
    "knowledge = store policy questions (tool null). other = greetings and unrelated (tool null)."
)


def messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text.strip()[:600]}]
