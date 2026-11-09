"""Render SFT rows with the model's real chat template and tokenize them.

Rows are rendered here (not by TRL) so training text is byte-identical to what vLLM renders at
inference: tool schemas go through the template's `tools=` block, tool calls become
<tool_call> JSON, and tool results become <tool_response> turns. The result is pre-tokenized
`input_ids` plus a `completion_mask` (1 only on tokens the model must generate).
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_END_OF_TURN = "<|im_end|>"


class RenderError(ValueError):
    """A row cannot be rendered into a clean (prompt, completion) pair."""


@dataclass
class RenderedRow:
    prompt_text: str
    completion_text: str
    input_ids: list[int]
    completion_mask: list[int]

    @property
    def token_count(self) -> int:
        return len(self.input_ids)

    @property
    def completion_token_count(self) -> int:
        return sum(self.completion_mask)


def render_row(tokenizer, row: dict) -> RenderedRow:
    prompt_messages, completion_messages = row["prompt"], row["completion"]
    tool_schemas = row.get("tools") or None
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tools=tool_schemas, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(
            prompt_messages + completion_messages, tools=tool_schemas, tokenize=False)
    except Exception as error:
        raise RenderError(f"chat template failed: {type(error).__name__}: {error}") from error
    if not full_text.startswith(prompt_text):
        raise RenderError("the rendered prompt is not a prefix of the rendered conversation")
    completion_text = full_text[len(prompt_text):]
    # Qwen's template ends an assistant turn with "<|im_end|>\n". The model stops at <|im_end|>
    # (its end-of-sequence token), so the trailing newline is never generated or trained on.
    completion_text = completion_text.removesuffix("\n")
    # An empty assistant message still renders as a bare end-of-turn token; training on that
    # would teach the model to stop immediately.
    end_of_turn = getattr(tokenizer, "eos_token", None) or DEFAULT_END_OF_TURN
    if not completion_text.replace(end_of_turn, "").strip():
        raise RenderError("empty completion")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    return RenderedRow(prompt_text, completion_text, prompt_ids + completion_ids,
                       [0] * len(prompt_ids) + [1] * len(completion_ids))
