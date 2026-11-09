"""Summary metrics over pipeline runs on the held-out tasks (use the same tasks every round).

Caveat: "pass" is judged by tests the model wrote itself, so it is a noisy proxy. The trusted
pass rate applies the training-data filters (non-trivial tests, no blockers) and is the more
honest number. HumanEval (humaneval.py) is the external yardstick.
"""
from __future__ import annotations

from collections import Counter

from harness.config import Config
from harness.evaluation.statistics import wilson_interval
from harness.training_data.trajectory_filter import rejection_reason


def summarize_runs(runs: list[dict], config: Config) -> dict:
    run_count = len(runs)
    denominator = run_count or 1
    outcomes = Counter(
        rejection_reason(run, config.filter, config.agent.test_timeout_seconds) or "trusted" for run in runs)
    passed = sum(run["passed"] for run in runs)
    trusted = outcomes["trusted"]
    tool_calls = sum(run.get("tool_calls", 0) for run in runs)
    tool_errors = sum(run.get("tool_errors", 0) for run in runs)
    return {
        "runs": run_count,
        "pass_rate": passed / denominator,
        "pass_rate_ci95": wilson_interval(passed, run_count),
        "trusted_pass_rate": trusted / denominator,
        "trusted_pass_rate_ci95": wilson_interval(trusted, run_count),
        "empty_or_broken_tests_rate": sum(run["test_count"] == 0 for run in runs) / denominator,
        "pipeline_failure_rate": sum(bool(run.get("failure_reason")) for run in runs) / denominator,
        "test_gate_failure_rate": sum(run.get("tests_valid") is False for run in runs) / denominator,
        "tool_error_rate": tool_errors / tool_calls if tool_calls else 0.0,
        "mean_agent_steps": sum(run.get("steps", 0) for run in runs) / denominator,
        "stalled_rate": sum(run.get("stop_reason") == "stalled" for run in runs) / denominator,
        "outcomes": dict(outcomes),
    }
