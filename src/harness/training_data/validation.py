"""Validate SFT rows BEFORE spending GPU hours on them.

Errors block training; warnings are worth reading. Checks cover row structure, the tool-call
format (the template double-encodes string `arguments`), structured-stage JSON against the
Pydantic schemas, hidden-test leakage into the implementer's context, duplicates, and (given a
tokenizer) chat-template rendering, token lengths and the share of tokens that carry loss.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pydantic import ValidationError

from harness.pipeline.artifacts import SCHEMA_BY_STAGE, validate_test_suite
from harness.training_data.chat_rendering import RenderError, render_row
from harness.training_data.sft_dataset import is_validation_task

VALID_ROLES = {"system", "user", "assistant", "tool"}
MIN_LEAK_LINE_CHARS = 10  # shorter assertion lines are too generic to count as a leak
MAX_EXAMPLES_PER_ISSUE = 3
LOW_LOSS_TOKEN_SHARE = 0.10
IMPLEMENT_DOMINANCE_SHARE = 0.85


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    count: int = 1
    examples: list = field(default_factory=list)


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = []
        for title, issues in (("ERRORS", self.errors), ("WARNINGS", self.warnings)):
            if issues:
                lines.append(f"{title}:")
                for issue in issues:
                    examples = f"  e.g. {issue.examples[:MAX_EXAMPLES_PER_ISSUE]}" if issue.examples else ""
                    lines.append(f"  [{issue.code}] x{issue.count}: {issue.message}{examples}")
        lines.append("STATS: " + json.dumps(self.stats, indent=2, default=str))
        verdict = "OK to train" if self.ok else f"{len(self.errors)} error type(s); fix before training"
        lines.append("RESULT: " + verdict)
        return "\n".join(lines)


class _IssueCollector:
    """Groups repeated problems under one Issue per code."""

    def __init__(self):
        self.issues_by_code: dict[str, Issue] = {}

    def add(self, severity: str, code: str, message: str, example=None) -> None:
        issue = self.issues_by_code.get(code)
        if issue is None:
            issue = self.issues_by_code[code] = Issue(severity, code, message, count=0)
        issue.count += 1
        if example is not None and len(issue.examples) < MAX_EXAMPLES_PER_ISSUE:
            issue.examples.append(example)

    def set_count(self, code: str, count: int) -> None:
        self.issues_by_code[code].count = count


def _check_row_structure(rows: list[dict], collector: _IssueCollector) -> None:
    for row_number, row in enumerate(rows):
        where = f"row {row_number} ({row.get('stage')}, task {row.get('task_id')})"
        if not isinstance(row.get("stage"), str) or "task_id" not in row:
            collector.add("error", "missing_keys", "row needs 'stage' and 'task_id'", where)
            continue
        prompt, completion = row.get("prompt"), row.get("completion")
        if not isinstance(prompt, list) or not prompt or not isinstance(completion, list):
            collector.add("error", "bad_shape", "'prompt' must be a non-empty message list and "
                          "'completion' a message list", where)
            continue
        tool_called_earlier = False
        for message in prompt:
            if (not isinstance(message, dict) or message.get("role") not in VALID_ROLES
                    or not isinstance(message.get("content"), str)):
                collector.add("error", "bad_message", "each message needs a valid role and string content", where)
                break
            if message["role"] == "assistant" and message.get("tool_calls"):
                tool_called_earlier = True
            if message["role"] == "tool" and not tool_called_earlier:
                collector.add("error", "orphan_tool_message",
                              "tool message with no earlier assistant tool_call", where)
        if prompt[0].get("role") != "system" and row["stage"] != "magicoder":
            collector.add("warning", "no_system_prompt", "first message is not a system prompt", where)
        if prompt[-1].get("role") not in ("user", "tool"):
            collector.add("error", "bad_last_prompt_role", "prompt must end with a user or tool message", where)
        if len(completion) != 1 or completion[0].get("role") != "assistant":
            collector.add("error", "bad_completion", "completion must be exactly one assistant message", where)
            continue
        assistant_message = completion[0]
        tool_calls = assistant_message.get("tool_calls") or []
        if not (assistant_message.get("content") or "").strip() and not tool_calls:
            collector.add("error", "empty_completion", "assistant message has neither text nor tool calls", where)
        available_tools = {schema["function"]["name"] for schema in (row.get("tools") or [])
                           if "function" in schema}
        for message in [*prompt, assistant_message]:
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                if not isinstance(function.get("arguments"), dict):
                    collector.add("error", "tool_args_not_dict",
                                  "tool-call arguments must be a dict; a JSON string is double-encoded "
                                  "by the chat template", where)
                if available_tools and function.get("name") not in available_tools:
                    collector.add("error", "unknown_tool",
                                  f"call to {function.get('name')!r} not in the row's tools", where)


def _check_structured_stages_and_leaks(rows: list[dict], collector: _IssueCollector) -> None:
    requirement_counts: dict = {}
    suites_by_task: dict = defaultdict(list)
    for row in rows:
        stage = row.get("stage")
        assistant_message = (row.get("completion") or [{}])[0]
        if stage not in SCHEMA_BY_STAGE or not isinstance(assistant_message.get("content"), str):
            continue
        schema = SCHEMA_BY_STAGE[stage]
        try:
            artifact = schema.model_validate_json(assistant_message["content"])
        except (ValidationError, ValueError) as error:
            collector.add("error", "bad_structured_completion",
                          f"{stage} completion is not valid {schema.__name__} JSON",
                          f"task {row['task_id']}: {str(error).splitlines()[0][:80]}")
            continue
        if stage == "requirements":
            requirement_counts[row["task_id"]] = len(artifact.requirements)
        if stage == "write_tests":
            suites_by_task[row["task_id"]].append(artifact)

    for task_id, suites in suites_by_task.items():  # a kept suite must still pass the gate
        for suite in suites:
            if task_id in requirement_counts and validate_test_suite(suite, requirement_counts[task_id], 1):
                collector.add("error", "suite_fails_gate", "a training test suite fails the quality gate",
                              f"task {task_id}")

    # Hidden-test leak: a task's assertion lines must never appear in its implement rows.
    for task_id, suites in suites_by_task.items():
        assertion_lines = {line.strip() for suite in suites for test in suite.tests
                           for line in test.code.splitlines()
                           if "assert" in line and len(line.strip()) >= MIN_LEAK_LINE_CHARS}
        for row in rows:
            if row.get("stage") != "implement" or row.get("task_id") != task_id:
                continue
            serialized_row = json.dumps([row.get("prompt"), row.get("completion")])
            # compare JSON-escaped forms, since the row is searched as JSON text
            leaked_line = next((line for line in assertion_lines if json.dumps(line)[1:-1] in serialized_row), None)
            if leaked_line:
                collector.add("error", "hidden_test_leak",
                              "an assertion from the hidden tests appears in an implement row",
                              f"task {task_id}: {leaked_line[:60]}")


def _check_token_lengths(rows: list[dict], tokenizer, max_length: int, collector: _IssueCollector) -> dict:
    lengths, completion_lengths = [], []
    overlong_by_stage: Counter = Counter()
    for row_number, row in enumerate(rows):
        try:
            rendered = render_row(tokenizer, row)
        except RenderError as error:
            collector.add("error", "render_failed", str(error), f"row {row_number} ({row.get('stage')})")
            continue
        lengths.append(rendered.token_count)
        completion_lengths.append(rendered.completion_token_count)
        if rendered.token_count > max_length:
            overlong_by_stage[row["stage"]] += 1
    if not lengths:
        return {}
    ordered = sorted(lengths)

    def percentile(quantile: float) -> int:
        return ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]

    total_tokens, loss_tokens = sum(lengths), sum(completion_lengths)
    if overlong_by_stage:
        overlong_count = sum(overlong_by_stage.values())
        collector.add("warning", "overlong_rows",
                      f"{overlong_count} rows exceed max_length={max_length} and will be dropped "
                      f"({dict(overlong_by_stage)}); raise max_length or shorten trajectories")
        collector.set_count("overlong_rows", overlong_count)
    loss_token_share = loss_tokens / total_tokens
    if loss_token_share < LOW_LOSS_TOKEN_SHARE:
        collector.add("warning", "low_loss_token_share",
                      f"only {loss_token_share:.1%} of tokens carry loss: long repeated prompts dominate "
                      "compute (tool schemas + history re-sent every turn)")
    return {"tokens": {"p50": percentile(0.5), "p90": percentile(0.9), "p99": percentile(0.99),
                       "max": ordered[-1], "total": total_tokens, "completion_total": loss_tokens,
                       "loss_token_share": round(loss_token_share, 4)},
            "rows_over_max_length": dict(overlong_by_stage)}


def validate_rows(rows: list[dict], tokenizer=None, max_length: int = 2048,
                  validation_fraction: float = 0.05, seed: int = 42, min_rows: int = 200) -> ValidationReport:
    collector = _IssueCollector()
    _check_row_structure(rows, collector)
    _check_structured_stages_and_leaks(rows, collector)

    row_fingerprints: Counter = Counter()
    for row in rows:
        serialized = json.dumps([row.get("prompt"), row.get("completion")], sort_keys=True, default=str)
        row_fingerprints[hashlib.sha1(serialized.encode()).hexdigest()] += 1
    duplicate_count = sum(count - 1 for count in row_fingerprints.values() if count > 1)
    if duplicate_count:
        collector.add("warning", "duplicate_rows", f"{duplicate_count} exact duplicate (prompt, completion) rows")
        collector.set_count("duplicate_rows", duplicate_count)

    rows_by_stage = Counter(row.get("stage") for row in rows)
    task_ids = {row.get("task_id") for row in rows}
    stages_by_task: dict = defaultdict(set)
    for row in rows:
        stages_by_task[row.get("task_id")].add(row.get("stage"))
    required_stages = set(SCHEMA_BY_STAGE) | {"implement"}
    incomplete_tasks = [task_id for task_id, stages in stages_by_task.items() if not required_stages <= stages]
    if incomplete_tasks:
        collector.add("warning", "incomplete_tasks", f"{len(incomplete_tasks)} tasks lack one or more stages",
                      incomplete_tasks[:MAX_EXAMPLES_PER_ISSUE])
        collector.set_count("incomplete_tasks", len(incomplete_tasks))
    if len(rows) < min_rows:
        collector.add("warning", "small_dataset", f"only {len(rows)} rows; expect little change from fine-tuning")
    if rows_by_stage.get("implement", 0) > IMPLEMENT_DOMINANCE_SHARE * max(len(rows), 1):
        collector.add("warning", "implement_dominates",
                      "implement turns are >85% of rows; the structured stages will be under-trained")
    validation_task_count = sum(is_validation_task(task_id, validation_fraction, seed) for task_id in task_ids)
    if validation_fraction > 0 and validation_task_count == 0:
        collector.add("warning", "no_validation_tasks",
                      "no task falls in the validation split; eval loss will be unavailable")
    stats = {"rows": len(rows), "tasks": len(task_ids), "validation_tasks": validation_task_count,
             "rows_by_stage": dict(rows_by_stage)}
    if tokenizer is not None:
        stats |= _check_token_lengths(rows, tokenizer, max_length, collector)
    return ValidationReport(list(collector.issues_by_code.values()), stats)
