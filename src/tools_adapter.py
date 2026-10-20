"""Adapter around the dataset's provided tools/ module (lookup_order, create_ticket).

We have NOT yet seen tools/tools.py, so the adapter inspects the real function signatures and
only passes the arguments they accept. Revisit after downloading the dataset.
"""
import inspect
import json
import sys

from .config import DATASET_DIR


def _load():
    sys.path.insert(0, str(DATASET_DIR))
    from tools import lookup_order, create_ticket  # provided by the organisers
    return lookup_order, create_ticket


def _call(fn, **kwargs):
    accepted = set(inspect.signature(fn).parameters)
    return fn(**{k: v for k, v in kwargs.items() if k in accepted and v not in (None, "")})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": "Look up a customer's order. Needs the order ID plus ONE verification detail: "
                       "last 4 digits of the phone number on the account, or the delivery pincode.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "phone_last4": {"type": "string", "description": "last 4 digits of customer phone"},
            "pincode": {"type": "string", "description": "delivery pincode"}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Hand over to a human agent. Only call when policy says a person must decide "
                       "or the customer's request cannot be granted by you. Give a short summary.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "order_id": {"type": "string"}},
            "required": ["summary"]}}},
]


class Tools:
    def __init__(self):
        self._lookup, self._ticket = _load()
        self.log: list[dict] = []       # everything the code really did, for honesty checks
        self.ticket_created = False

    def run(self, name: str, args: dict) -> str:
        try:
            if name == "lookup_order":
                res = _call(self._lookup, **args)
            elif name == "create_ticket":
                res = _call(self._ticket, **args)
                self.ticket_created = bool(isinstance(res, dict) and res.get("created"))
            else:
                res = {"error": f"unknown tool {name}"}
        except Exception as e:
            res = {"error": str(e)}
        self.log.append({"tool": name, "args": args, "result": res})
        return json.dumps(res, default=str)
