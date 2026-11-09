"""LangChain message helpers: conversion to chat-template dicts for training rows, and
compaction of a tool-using conversation into the prompt budget."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from harness.monitoring import Tracer
from harness.prompt_context import TokenCounter, truncate_text

ELIDED_TOOL_OUTPUT = "[earlier tool output elided]"
MIN_LAST_MESSAGE_TOKENS = 50


def message_text(content: Any) -> str:
    """Plain text of a message's content (which may be a string or a list of content parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
    return str(content)


def to_chat_dict(message: BaseMessage) -> dict:
    """Chat-template message dict. Tool-call arguments stay dicts: Qwen's template double-encodes
    string arguments."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message_text(message.content)}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message_text(message.content)}
    if isinstance(message, AIMessage):
        chat_dict: dict = {"role": "assistant", "content": message_text(message.content)}
        if message.tool_calls:
            chat_dict["tool_calls"] = [
                {"type": "function", "function": {"name": call["name"], "arguments": call["args"]}}
                for call in message.tool_calls
            ]
        return chat_dict
    if isinstance(message, ToolMessage):
        return {"role": "tool", "content": message_text(message.content), "name": message.name or ""}
    return {"role": "user", "content": message_text(message.content)}


def message_token_count(message: BaseMessage, count_tokens: TokenCounter) -> int:
    tokens = count_tokens(message_text(message.content))
    if isinstance(message, AIMessage) and message.tool_calls:
        calls = [{"name": call["name"], "arguments": call["args"]} for call in message.tool_calls]
        tokens += count_tokens(json.dumps(calls))
    return tokens


def _with_content(message: BaseMessage, content: str) -> BaseMessage:
    return message.model_copy(update={"content": content})


def compact_messages(messages: list[BaseMessage], budget_tokens: int, count_tokens: TokenCounter,
                     tracer: Tracer | None = None) -> list[BaseMessage]:
    """Fit a tool-using conversation into `budget_tokens`; returns a new list.

    Works on whole turns (an AI message plus its tool results) so tool-call ids are never
    orphaned. Keeps the system prompt and the task. Shrinks in this order: elide old tool
    output, drop the oldest turns, truncate the last message.
    """
    def total_tokens(conversation: list[BaseMessage]) -> int:
        return sum(message_token_count(message, count_tokens) for message in conversation)

    if total_tokens(messages) <= budget_tokens:
        return list(messages)
    header, rest = list(messages[:2]), messages[2:]  # system prompt + task
    turns: list[list[BaseMessage]] = []
    for message in rest:
        if isinstance(message, AIMessage) or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)

    def flattened() -> list[BaseMessage]:
        return header + [message for turn in turns for message in turn]

    events: list[tuple[str, int, int]] = []  # (section, from_tokens, to_tokens)
    for turn in turns[:-1]:  # 1. elide old tool output, oldest first; the newest turn is kept
        if total_tokens(flattened()) <= budget_tokens:
            break
        for position, message in enumerate(turn):
            if not isinstance(message, AIMessage) and message_text(message.content) != ELIDED_TOOL_OUTPUT:
                turn[position] = _with_content(message, ELIDED_TOOL_OUTPUT)
                events.append(("conversation:tool_output", 1, 0))
    dropped_turns = 0
    while total_tokens(flattened()) > budget_tokens and len(turns) > 1:  # 2. oldest whole turns
        turns.pop(0)
        dropped_turns += 1
    if dropped_turns:
        task_text = message_text(header[1].content) + f"\n[{dropped_turns} earlier steps omitted]"
        header[1] = _with_content(header[1], task_text)
        events.append(("conversation:dropped_steps", dropped_turns, 0))
    compacted = flattened()
    if total_tokens(compacted) > budget_tokens:  # 3. last resort
        room = max(budget_tokens - total_tokens(compacted[:-1]), MIN_LAST_MESSAGE_TOKENS)
        last_text = truncate_text(message_text(compacted[-1].content), room, count_tokens, "middle")
        compacted[-1] = _with_content(compacted[-1], last_text)
        events.append(("conversation:last_message", 0, room))
    if tracer:
        for section, from_tokens, to_tokens in events:
            tracer.emit("context_truncated", section=section, from_tokens=from_tokens, to_tokens=to_tokens)
    return compacted
