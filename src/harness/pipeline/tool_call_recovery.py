"""Recover tool calls the server's parser missed.

vLLM's `hermes` parser only recognizes calls wrapped in <tool_call>...</tool_call>. Qwen2.5-Coder
often writes the same JSON inside a ```json fence, or as bare JSON, so the server returns it as
plain text and the agent loop would end without doing anything. This module finds such calls
and turns them into real tool calls.

The recovered message is what gets recorded for training, rendered by the chat template as a
proper <tool_call>, so fine-tuning teaches the model the format the server expects.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_CODE_FENCE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)


def _candidate_json_texts(text: str) -> list[str]:
    """Places a JSON tool call may hide: <tool_call> tags, code fences, or the whole reply."""
    candidates = [match.strip() for match in _TOOL_CALL_TAG.findall(text)]
    candidates += [match.strip() for match in _CODE_FENCE.findall(text)]
    candidates.append(text.strip())
    return [candidate for candidate in candidates if candidate.startswith(("{", "["))]


def _as_tool_call(data: Any, tool_names: set[str]) -> dict | None:
    if not isinstance(data, dict) or data.get("name") not in tool_names:
        return None
    arguments = data.get("arguments", data.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return {"name": data["name"], "args": arguments, "id": f"recovered-{uuid.uuid4().hex[:12]}",
            "type": "tool_call"}


def recover_tool_calls(text: str, tool_names: set[str]) -> list[dict]:
    """Tool calls (LangChain format) written as JSON in `text`; [] if there are none.

    Only names of real tools are accepted, so ordinary JSON in a reply is never mistaken for a
    call. The first place that yields valid calls wins.
    """
    for candidate in _candidate_json_texts(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        calls = [_as_tool_call(item, tool_names) for item in items]
        if calls and all(call is not None for call in calls):
            return calls
    return []


def with_recovered_tool_calls(reply: AIMessage, tool_names: set[str]) -> tuple[AIMessage, int]:
    """(reply with recovered tool calls, number recovered). Replies that already carry tool
    calls, or contain none, are returned unchanged."""
    if reply.tool_calls:
        return reply, 0
    content = reply.content if isinstance(reply.content, str) else ""
    calls = recover_tool_calls(content, tool_names)
    if not calls:
        return reply, 0
    return AIMessage(content="", tool_calls=calls, response_metadata=reply.response_metadata), len(calls)
