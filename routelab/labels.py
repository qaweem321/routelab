"""The label space: which Deskpilot agent handles a request, and which tool it should call first."""

import json
from typing import NamedTuple

ROUTES = ("orders", "tickets", "knowledge", "other")
TOOLS = {
    "orders": ("get_order", "search_orders", "cancel_order", "refund_order"),
    "tickets": ("get_ticket", "search_tickets", "create_ticket", "update_ticket"),
    "knowledge": (),
    "other": (),
}
ALL_TOOLS = tuple(t for tools in TOOLS.values() for t in tools)


class Label(NamedTuple):
    route: str
    tool: str | None

    def to_json(self) -> str:
        return json.dumps({"route": self.route, "tool": self.tool}, separators=(",", ":"))

    @property
    def key(self) -> str:
        return f"{self.route}/{self.tool or '-'}"


def parse_label(text: str) -> Label | None:
    """Read a model's output. Anything that isn't a valid label for the label space is None."""
    start, end = text.find("{"), text.find("}")
    if start == -1 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    route, tool = data.get("route"), data.get("tool")
    if route not in ROUTES:
        return None
    if tool is not None and tool not in TOOLS[route]:
        return None
    if tool is None and TOOLS[route]:
        return None
    return Label(route, tool)
