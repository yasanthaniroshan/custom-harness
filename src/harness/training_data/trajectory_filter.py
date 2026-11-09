"""Decide which generated runs to trust, and turn the trusted ones into SFT rows.

The reward ("tests pass") comes from tests the same model wrote, so passing alone is not
trusted. A run is kept only if its tests look real:
  - enough test cases,
  - most of them FAIL against an empty implementation (rules out `assert True`),
  - the review found no [Blocker],
  - few failed tool calls,
and a task contributes at most `max_runs_per_task` distinct solutions.

Each model call of a kept run becomes one row; tool-loop turns carry their conversation so far.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.config import FilterConfig
from harness.sandbox import run_code_against_tests

BLOCKER_PATTERN = re.compile(r"\[\s*blocker\s*\]", re.IGNORECASE)
FILTER_WORKERS = 4  # sandbox subprocesses run in parallel


def rejection_reason(run: dict, config: FilterConfig, timeout_seconds: int = 10) -> str | None:
    """None if the run is trustworthy, otherwise a short reason code."""
    if run.get("failure_reason"):
        return "pipeline_failed"
    if run.get("tests_valid") is False:
        return "tests_gate_failed"
    if not run["passed"]:
        return "tests_failed"
    if run.get("tool_errors", 0) > config.max_tool_errors:
        return "tool_errors"
    if run["test_count"] < config.min_tests:
        return "too_few_tests"
    empty_code_report = run_code_against_tests("", run["artifacts"]["tests"], timeout_seconds)
    if empty_code_report.tests_unusable or not empty_code_report.cases:
        return "tests_unusable"
    if empty_code_report.passed_count / len(empty_code_report.cases) > config.max_stub_pass_fraction:
        return "tests_trivial"
    if config.reject_blockers and BLOCKER_PATTERN.search(run["artifacts"].get("review", "")):
        return "review_blocker"
    return None


def _code_fingerprint(run: dict) -> str:
    whitespace_free = "".join(run["artifacts"]["code"].split())
    return hashlib.sha1(whitespace_free.encode()).hexdigest()


def run_to_rows(run: dict) -> list[dict]:
    """One {stage, task_id, prompt, completion[, tools]} row per model call of a kept run."""
    rows = []
    for call in run["llm_calls"]:
        if call["stage"] == "implement" and not call.get("passed"):
            continue  # never train on a candidate whose code failed
        completion_message = call.get("completion_message") or {"role": "assistant",
                                                                "content": call["completion"]}
        row = {"stage": call["stage"], "task_id": run["task_id"], "prompt": call["messages"],
               "completion": [completion_message]}
        if call.get("tool_schemas"):
            row["tools"] = call["tool_schemas"]
        rows.append(row)
    return rows


def filter_runs(runs: Iterable[dict], config: FilterConfig, timeout_seconds: int = 10
                ) -> tuple[list[dict], Counter]:
    """(SFT rows, counts of every outcome) for a batch of run records."""
    runs = list(runs)
    with ThreadPoolExecutor(FILTER_WORKERS) as pool:
        reasons = list(pool.map(lambda run: rejection_reason(run, config, timeout_seconds), runs))
    stats: Counter = Counter()
    trusted_runs_by_task: dict = {}
    for run, reason in zip(runs, reasons, strict=True):
        stats[reason or "kept_candidate"] += 1
        if reason is None:
            trusted_runs_by_task.setdefault(run["task_id"], []).append(run)

    rows: list[dict] = []
    for trusted_runs in trusted_runs_by_task.values():
        seen_fingerprints: set[str] = set()
        kept_for_task = 0
        for run in sorted(trusted_runs, key=lambda run: run["steps"]):  # prefer shorter trajectories
            fingerprint = _code_fingerprint(run)
            if fingerprint in seen_fingerprints:
                stats["duplicate_code"] += 1
                continue
            if kept_for_task >= config.max_runs_per_task:
                stats["over_task_cap"] += 1
                continue
            seen_fingerprints.add(fingerprint)
            kept_for_task += 1
            rows.extend(run_to_rows(run))
            stats["kept"] += 1
    return rows, stats


def filter_run_files(runs_paths: list[Path], output_path: Path, config: FilterConfig,
                     timeout_seconds: int = 10) -> Counter:
    runs = [json.loads(line) for path in runs_paths for line in path.read_text().splitlines() if line.strip()]
    rows, stats = filter_runs(runs, config, timeout_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    stats["rows_written"] = len(rows)
    stats.update({f"rows_{stage}": count for stage, count in Counter(row["stage"] for row in rows).items()})
    return stats
