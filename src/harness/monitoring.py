"""Monitoring: structured events written as JSONL, and a summary report over them.

Each event is one JSON object per line: {"ts", "kind", "stage", ...}. Kinds:
  llm_call            one model request (tokens, truncation, latency)
  llm_error           a model request that raised
  tool_call           one workspace tool call (ok, output size)
  context_truncated   a prompt section or conversation turn was shortened to fit the budget
  context_overflow    the protected (never-truncated) part alone exceeds the budget
  run_end             one finished pipeline run

Truncated generations (finish_reason == "length") are the failure that otherwise goes
unnoticed: a cut-off implementation simply fails its tests.
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_current_stage: ContextVar[str | None] = ContextVar("harness_stage", default=None)


@contextmanager
def stage_scope(stage_name: str) -> Iterator[None]:
    """Tag every event emitted inside the block with this stage name."""
    reset_token = _current_stage.set(stage_name)
    try:
        yield
    finally:
        _current_stage.reset(reset_token)


class Tracer:
    """Thread-safe event sink. Writes JSONL to `path`; keeps events in memory if `keep_in_memory`."""

    def __init__(self, path: str | Path | None = None, keep_in_memory: bool | None = None):
        self.path = Path(path) if path else None
        self.keep_in_memory = (self.path is None) if keep_in_memory is None else keep_in_memory
        self.events: list[dict] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **fields) -> None:
        stage = fields.pop("stage_name", None) or _current_stage.get()
        event = {"ts": round(time.time(), 3), "kind": kind, "stage": stage, **fields}
        with self._lock:
            if self.keep_in_memory:
                self.events.append(event)
            if self.path:
                with self.path.open("a") as events_file:
                    events_file.write(json.dumps(event, default=str) + "\n")


NULL_TRACER = Tracer(None, keep_in_memory=False)


def tracer_for(enabled: bool, path: Path) -> Tracer:
    return Tracer(path) if enabled else NULL_TRACER


# ---- reporting --------------------------------------------------------------------------

def read_events(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def summarize_events(events: Iterable[dict]) -> dict:
    stages: dict = defaultdict(lambda: {"calls": 0, "prompt_tokens": [], "completion_tokens": [],
                                        "seconds": [], "truncated": 0})
    tools: dict = defaultdict(lambda: {"calls": 0, "errors": 0, "output_tokens": []})
    context_events: Counter = Counter()
    runs = {"count": 0, "passed": 0, "steps": [], "tool_errors": [], "stop_reasons": Counter()}
    for event in events:
        kind = event["kind"]
        if kind == "llm_call":
            stage = stages[event.get("stage") or "?"]
            stage["calls"] += 1
            stage["prompt_tokens"].append(event["prompt_tokens"])
            stage["completion_tokens"].append(event["completion_tokens"])
            if event.get("seconds") is not None:
                stage["seconds"].append(event["seconds"])
            stage["truncated"] += bool(event.get("truncated"))
        elif kind == "tool_call":
            tool = tools[event["tool"]]
            tool["calls"] += 1
            tool["errors"] += not event["ok"]
            tool["output_tokens"].append(event["output_tokens"])
        elif kind in ("context_truncated", "context_overflow"):
            context_events[f"{kind}:{event.get('section', '-')}"] += 1
        elif kind == "run_end":
            runs["count"] += 1
            runs["passed"] += bool(event["passed"])
            runs["steps"].append(event.get("steps", 0))
            runs["tool_errors"].append(event.get("tool_errors", 0))
            if event.get("stop_reason"):
                runs["stop_reasons"][event["stop_reason"]] += 1

    return {
        "stages": {
            name: {
                "calls": stage["calls"],
                "prompt_tokens_mean": _mean(stage["prompt_tokens"]),
                "prompt_tokens_p95": _percentile(stage["prompt_tokens"], 0.95),
                "completion_tokens_mean": _mean(stage["completion_tokens"]),
                "completion_tokens_max": max(stage["completion_tokens"], default=0),
                "seconds_mean": round(sum(stage["seconds"]) / len(stage["seconds"]), 2)
                if stage["seconds"] else None,
                "truncated": stage["truncated"],
                "truncation_rate": round(stage["truncated"] / stage["calls"], 4) if stage["calls"] else 0.0,
            }
            for name, stage in stages.items()
        },
        "tools": {
            name: {"calls": tool["calls"], "error_rate": round(tool["errors"] / tool["calls"], 3),
                   "output_tokens_mean": _mean(tool["output_tokens"])}
            for name, tool in tools.items()
        },
        "context": dict(context_events),
        "runs": {
            "count": runs["count"],
            "pass_rate": round(runs["passed"] / runs["count"], 3) if runs["count"] else None,
            "mean_steps": _mean(runs["steps"]),
            "mean_tool_errors": _mean(runs["tool_errors"]),
            "stop_reasons": dict(runs["stop_reasons"]),
        },
    }


def format_report(summary: dict) -> str:
    lines = ["STAGE          calls  prompt(mean/p95)  completion(mean/max)  seconds  truncated"]
    for name, stage in summary["stages"].items():
        seconds = "-" if stage["seconds_mean"] is None else stage["seconds_mean"]
        lines.append(
            f"{name:<14}{stage['calls']:>6}  {stage['prompt_tokens_mean']:>7}/{stage['prompt_tokens_p95']:<8}"
            f"  {stage['completion_tokens_mean']:>8}/{stage['completion_tokens_max']:<10}"
            f"  {seconds:>7}  {stage['truncated']} ({stage['truncation_rate']:.1%})"
        )
    runs = summary["runs"]
    if runs["count"]:
        lines.append(f"\nRuns: {runs['count']}, pass rate {runs['pass_rate']}, "
                     f"mean steps {runs['mean_steps']}, mean tool errors {runs['mean_tool_errors']}, "
                     f"stops {runs['stop_reasons']}")
    if summary["tools"]:
        lines.append("Tools: " + ", ".join(
            f"{name} x{tool['calls']} ({tool['error_rate']:.0%} err)"
            for name, tool in summary["tools"].items()))
    if summary["context"]:
        lines.append("Context events: " + ", ".join(
            f"{section} x{count}" for section, count in summary["context"].items()))
    return "\n".join(lines)
