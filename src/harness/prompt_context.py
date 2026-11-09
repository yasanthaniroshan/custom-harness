"""Prompt construction within a token budget: counting, deliberate truncation, budgeted
assembly of prompt sections, and few-shot example injection.

Every stage prompt goes through PromptBuilder.build_prompt, so budgets and truncation are
applied identically when generating training data and at inference.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from harness.config import ContextConfig

if TYPE_CHECKING:
    from harness.monitoring import Tracer
    from harness.pipeline.stages import Stage

TokenCounter = Callable[[str], int]
CODE_ARTIFACTS = {"code", "tests"}  # rendered inside ```python fences


def approximate_token_count(text: str) -> int:
    """Cheap fallback (~4 characters per token). Prefer a real tokenizer when one is loaded."""
    return (len(text) + 3) // 4


def tokenizer_token_counter(tokenizer) -> TokenCounter:
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


# ---- truncation -------------------------------------------------------------------------

def truncate_text(text: str, max_tokens: int, count_tokens: TokenCounter, mode: str = "middle") -> str:
    """Shrink text to at most max_tokens, leaving an explicit marker where content was cut.

    mode: "head" keeps the start, "tail" keeps the end, "middle" keeps both ends.
    """
    total_tokens = count_tokens(text)
    if total_tokens <= max_tokens:
        return text
    if max_tokens <= 0:
        return ""
    marker = f"\n[... ~{total_tokens - max_tokens} tokens truncated ...]\n"

    def keep_characters(kept: int) -> str:
        if mode == "head":
            return text[:kept] + marker
        if mode == "tail":
            return marker + (text[len(text) - kept:] if kept else "")
        head_length = int(kept * 0.6)
        tail_length = kept - head_length
        return text[:head_length] + marker + (text[len(text) - tail_length:] if tail_length else "")

    low, high = 0, len(text)
    while low < high:  # binary search for the most characters whose result still fits
        middle = (low + high + 1) // 2
        if count_tokens(keep_characters(middle)) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return keep_characters(low)


# ---- budgeted assembly ------------------------------------------------------------------

@dataclass
class PromptSection:
    """One part of a prompt. Only `body` is ever truncated; prefix and suffix (labels, code
    fences) are always kept.

    priority: 0 = never truncated; otherwise higher numbers are shrunk first when over budget.
    mode: "head" | "middle" | "tail" truncation, or "drop" (keep whole or remove entirely).
    """

    name: str
    body: str
    prefix: str = ""
    suffix: str = ""
    priority: int = 1
    max_tokens: int | None = None  # per-section cap (counted on the whole section text)
    mode: str = "middle"

    @property
    def text(self) -> str:
        return f"{self.prefix}{self.body}{self.suffix}" if self.body else ""


def _shrink_section(section: PromptSection, target_tokens: int, count_tokens: TokenCounter) -> None:
    if section.mode == "drop":
        section.body = ""
        return
    overhead = count_tokens(section.prefix + section.suffix)
    section.body = truncate_text(section.body, max(target_tokens - overhead, 0), count_tokens, section.mode)


def assemble_sections(sections: list[PromptSection], budget_tokens: int,
                      count_tokens: TokenCounter) -> tuple[str, list[dict]]:
    """Join sections with newlines, truncating by priority to fit `budget_tokens`.

    Returns (prompt, events). Events describe every truncation and any remaining overflow.
    """
    events: list[dict] = []

    def total_tokens() -> int:
        return sum(count_tokens(section.text) for section in sections if section.text)

    def record_truncation(section: PromptSection, tokens_before: int) -> None:
        tokens_after = count_tokens(section.text)
        if tokens_after < tokens_before:
            events.append({"kind": "context_truncated", "section": section.name,
                           "from_tokens": tokens_before, "to_tokens": tokens_after,
                           "dropped": not section.body})

    for section in sections:  # per-section caps first
        if section.priority and section.max_tokens and count_tokens(section.text) > section.max_tokens:
            tokens_before = count_tokens(section.text)
            _shrink_section(section, section.max_tokens, count_tokens)
            record_truncation(section, tokens_before)

    shrinkable = sorted((section for section in sections if section.priority),
                        key=lambda section: -section.priority)
    for section in shrinkable:
        excess_tokens = total_tokens() - budget_tokens
        if excess_tokens <= 0:
            break
        tokens_before = count_tokens(section.text)
        _shrink_section(section, tokens_before - excess_tokens, count_tokens)
        record_truncation(section, tokens_before)

    if (final_tokens := total_tokens()) > budget_tokens:  # only never-truncated parts are left
        events.append({"kind": "context_overflow", "tokens": final_tokens, "budget": budget_tokens})
    return "\n".join(section.text for section in sections if section.text), events


# ---- few-shot injection -----------------------------------------------------------------

_WORD_PATTERN = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


@dataclass
class _FewShotDocument:
    term_counts: Counter
    length: int
    prompt: str
    completion: str


class FewShotStore:
    """BM25 retrieval of earlier (prompt, completion) pairs, per stage, from filtered SFT rows."""

    def __init__(self, rows: list[dict], term_saturation: float = 1.5, length_normalization: float = 0.75):
        self.term_saturation = term_saturation  # BM25 k1
        self.length_normalization = length_normalization  # BM25 b
        self.documents_by_stage: dict[str, list[_FewShotDocument]] = defaultdict(list)
        for row in rows:
            prompt, completion = row["prompt"][-1]["content"], row["completion"][0]["content"]
            words = _words(prompt)
            self.documents_by_stage[row["stage"]].append(
                _FewShotDocument(Counter(words), len(words), prompt, completion))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> FewShotStore:
        lines = Path(path).read_text().splitlines()
        return cls([json.loads(line) for line in lines if line.strip()])

    def retrieve(self, stage: str, query: str, count: int) -> list[tuple[str, str]]:
        """Up to `count` (prompt, completion) pairs most similar to `query`; never zero-score ones."""
        documents = self.documents_by_stage.get(stage, [])
        if not documents or count <= 0:
            return []
        query_words = set(_words(query))
        document_total = len(documents)
        average_length = sum(document.length for document in documents) / document_total or 1
        document_frequency = {word: sum(1 for document in documents if word in document.term_counts)
                              for word in query_words}
        scored = []
        for document in documents:
            score = 0.0
            for word in query_words:
                term_frequency = document.term_counts[word]
                if not term_frequency:
                    continue
                inverse_frequency = math.log(
                    1 + (document_total - document_frequency[word] + 0.5) / (document_frequency[word] + 0.5))
                length_factor = 1 - self.length_normalization + (
                    self.length_normalization * document.length / average_length)
                score += inverse_frequency * term_frequency * (self.term_saturation + 1) / (
                    term_frequency + self.term_saturation * length_factor)
            scored.append((score, document.prompt, document.completion))
        scored.sort(key=lambda item: -item[0])
        return [(prompt, completion) for score, prompt, completion in scored[:count] if score > 0]


# ---- the builder ------------------------------------------------------------------------

class PromptBuilder:
    """Builds every stage prompt within the token budget, with optional context injection."""

    def __init__(self, config: ContextConfig | None = None,
                 count_tokens: TokenCounter = approximate_token_count,
                 tracer: Tracer | None = None, few_shot_store: FewShotStore | None = None):
        self.config = config or ContextConfig()
        self.count_tokens = count_tokens
        self.tracer = tracer
        self.few_shot_store = few_shot_store
        if few_shot_store is None and self.config.few_shot_count > 0 and self.config.few_shot_path:
            self.few_shot_store = FewShotStore.from_jsonl(self.config.few_shot_path)

    def _artifact_section(self, artifact_name: str, value: str) -> PromptSection:
        is_test_output = artifact_name == "test_output"
        cap = self.config.test_output_max_tokens if is_test_output else self.config.artifact_max_tokens
        label = f"{artifact_name.replace('_', ' ').capitalize()}:\n"
        if artifact_name in CODE_ARTIFACTS:
            return PromptSection(f"artifact:{artifact_name}", value, label + "```python\n", "\n```",
                                 priority=2, max_tokens=cap, mode="middle")
        return PromptSection(f"artifact:{artifact_name}", value, label, "", priority=2, max_tokens=cap,
                             mode="head" if is_test_output else "middle")

    def build_prompt(self, stage: Stage, artifacts: dict, instruction: str | None = None) -> str:
        """Prompt for a stage: injected context, then its declared input artifacts, then the
        instruction (the stage's own unless `instruction` overrides it)."""
        config = self.config
        sections: list[PromptSection] = []
        conventions = "\n".join(
            text for text in (config.conventions.get("*"), config.conventions.get(stage.name)) if text)
        if conventions:
            sections.append(PromptSection("conventions", conventions, "Project conventions:\n", "",
                                          priority=3, mode="head"))

        if self.few_shot_store and config.few_shot_count > 0:
            query = "\n".join(str(artifacts[name]) for name in stage.inputs)
            examples = self.few_shot_store.retrieve(stage.name, query, config.few_shot_count)
            for example_index, (example_prompt, example_completion) in enumerate(examples):
                example_text = f"Example input:\n{example_prompt}\nExample output:\n{example_completion}"
                if self.count_tokens(example_text) <= config.few_shot_max_tokens:
                    sections.append(PromptSection(f"example:{example_index}", example_text,
                                                  priority=4, mode="drop"))

        sections += [self._artifact_section(name, artifacts[name]) for name in stage.inputs]
        sections.append(PromptSection("instruction", instruction or stage.instruction, priority=0))

        prompt, events = assemble_sections(sections, config.max_prompt_tokens, self.count_tokens)
        if self.tracer:
            for event in events:
                self.tracer.emit(event.pop("kind"), stage_name=stage.name, **event)
        return prompt
