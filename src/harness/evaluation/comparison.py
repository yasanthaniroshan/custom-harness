"""Compare two evaluation outputs on the same tasks: intervals, a paired test and a verdict."""
from __future__ import annotations

import json
from pathlib import Path

from harness.config import Config
from harness.evaluation.statistics import (
    mcnemar_exact_p_value,
    paired_bootstrap_interval,
    wilson_interval,
)
from harness.monitoring import read_events, summarize_events
from harness.training_data.trajectory_filter import rejection_reason

MIN_PAIRED_TASKS = 30
SIGNIFICANCE_LEVEL = 0.05
OUTCOME_KINDS = ("passed", "trusted", "valid_tests")

Outcomes = dict[tuple, bool]  # (task_id, sample) -> success


def compare_outcomes(outcomes_a: Outcomes, outcomes_b: Outcomes) -> dict:
    """Pairs the (task, sample) keys present in both A and B."""
    shared_keys = sorted(set(outcomes_a) & set(outcomes_b), key=str)
    only_b_passed = sum(outcomes_b[key] and not outcomes_a[key] for key in shared_keys)
    only_a_passed = sum(outcomes_a[key] and not outcomes_b[key] for key in shared_keys)
    passed_a = sum(outcomes_a[key] for key in shared_keys)
    passed_b = sum(outcomes_b[key] for key in shared_keys)
    paired = len(shared_keys)
    return {
        "paired": paired,
        "a": {"rate": passed_a / paired if paired else 0.0, "ci95": wilson_interval(passed_a, paired)},
        "b": {"rate": passed_b / paired if paired else 0.0, "ci95": wilson_interval(passed_b, paired)},
        "delta": (passed_b - passed_a) / paired if paired else 0.0,
        "delta_ci95": paired_bootstrap_interval(
            [int(outcomes_b[key]) - int(outcomes_a[key]) for key in shared_keys]),
        "b_better": only_b_passed,
        "a_better": only_a_passed,
        "same": paired - only_b_passed - only_a_passed,
        "p_value": mcnemar_exact_p_value(only_a_passed, only_b_passed),
    }


def verdict(comparison: dict, significance_level: float = SIGNIFICANCE_LEVEL) -> str:
    delta_low, delta_high = comparison["delta_ci95"]
    if comparison["paired"] < MIN_PAIRED_TASKS:
        return f"only {comparison['paired']} paired tasks: too few to conclude anything"
    significant = comparison["p_value"] < significance_level
    if significant and delta_low > 0:
        return "B is significantly better"
    if significant and delta_high < 0:
        return "B is significantly WORSE (regression)"
    return "no significant difference"


def format_comparison(name: str, comparison: dict) -> str:
    a, b = comparison["a"], comparison["b"]
    delta_low, delta_high = comparison["delta_ci95"]
    return (f"{name:<14} A {a['rate']:6.1%} [{a['ci95'][0]:.0%}-{a['ci95'][1]:.0%}]   "
            f"B {b['rate']:6.1%} [{b['ci95'][0]:.0%}-{b['ci95'][1]:.0%}]   "
            f"delta {comparison['delta']:+.1%} [{delta_low:+.1%}, {delta_high:+.1%}]   "
            f"B better {comparison['b_better']} / A better {comparison['a_better']} / "
            f"same {comparison['same']}   p={comparison['p_value']:.3f}  -> {verdict(comparison)}")


def load_eval_runs(path: str | Path) -> list[dict]:
    """Runs from a runs JSONL file, or from a directory holding eval_runs.jsonl / runs.jsonl."""
    location = Path(path)
    if location.is_dir():
        location = next((location / name for name in ("eval_runs.jsonl", "runs.jsonl")
                         if (location / name).exists()), location / "eval_runs.jsonl")
    return [json.loads(line) for line in location.read_text().splitlines() if line.strip()]


def run_outcomes(runs: list[dict], config: Config, kind: str) -> Outcomes:
    """kind: passed | trusted (passes the training-data filters) | valid_tests."""
    outcomes = {}
    for run in runs:
        key = (run["task_id"], run["sample"])
        if kind == "passed":
            outcomes[key] = bool(run.get("passed"))
        elif kind == "trusted":
            outcomes[key] = rejection_reason(run, config.filter, config.agent.test_timeout_seconds) is None
        elif kind == "valid_tests":
            outcomes[key] = run.get("tests_valid") is not False and not run.get("failure_reason")
        else:
            raise ValueError(f"unknown outcome kind {kind!r}; use one of {OUTCOME_KINDS}")
    return outcomes


def load_humaneval_outcomes(path: str | Path) -> Outcomes:
    """{(task_id, 0): passed} from a humaneval_*.json file written by `harness bench`."""
    data = json.loads(Path(path).read_text())
    return {(result["task_id"], 0): bool(result["passed"]) for result in data["results"]}


def is_humaneval_file(path: str | Path) -> bool:
    return str(path).endswith(".json") and "humaneval" in Path(path).name


def compare_paths(path_a: str, path_b: str, config: Config) -> str:
    """Compare two evaluation outputs: round directories or humaneval_*.json files."""
    lines = [f"A = {path_a}", f"B = {path_b}", ""]
    if is_humaneval_file(path_a):
        comparison = compare_outcomes(load_humaneval_outcomes(path_a), load_humaneval_outcomes(path_b))
        lines.append(format_comparison("humaneval", comparison))
        return "\n".join(lines)
    runs_a, runs_b = load_eval_runs(path_a), load_eval_runs(path_b)
    for kind in OUTCOME_KINDS:
        comparison = compare_outcomes(run_outcomes(runs_a, config, kind), run_outcomes(runs_b, config, kind))
        lines.append(format_comparison(kind, comparison))
    for label, path in (("A", path_a), ("B", path_b)):
        events_path = Path(path) / "eval_events.jsonl" if Path(path).is_dir() else None
        if events_path and events_path.exists():
            summary = summarize_events(read_events(events_path))
            truncated = {stage: values["truncated"] for stage, values in summary["stages"].items()
                         if values["truncated"]}
            lines.append(f"{label} truncated generations: {truncated or 'none'}; "
                         f"mean agent steps {summary['runs']['mean_steps']}")
    return "\n".join(lines)
