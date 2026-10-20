"""Thin Groq chat wrapper with simple retry on rate limits."""
import os
import time

from groq import Groq

from .config import GROQ_MODEL

_client = None


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def chat(messages, tools=None, temperature=0.1, max_tokens=700):
    """Return the raw assistant message object from Groq."""
    kwargs = dict(model=GROQ_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    for attempt in range(4):
        try:
            return client().chat.completions.create(**kwargs).choices[0].message
        except Exception as e:  # rate limit / transient errors
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
