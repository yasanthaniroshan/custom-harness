import json
import math

from harness.config import Config
from harness.evaluation.comparison import (
    compare_outcomes,
    compare_paths,
    format_comparison,
    verdict,
)
from harness.evaluation.harness_metrics import summarize_runs
from harness.evaluation.statistics import (
    mcnemar_exact_p_value,
    paired_bootstrap_interval,
    wilson_interval,
)
from harness.training.train_log import summarize_train_log


def test_wilson_interval_known_values():
    assert wilson_interval(0, 10)[0] == 0.0 and 0.2 < wilson_interval(0, 10)[1] < 0.3
    low, high = wilson_interval(50, 100)
    assert abs(low - 0.404) < 0.005 and abs(high - 0.596) < 0.005
    assert wilson_interval(10, 10)[1] == 1.0 and wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(5, 20)[1] - wilson_interval(5, 20)[0] > wilson_interval(50, 200)[1] - wilson_interval(50, 200)[0]  # more data, tighter


def test_mcnemar_exact_known_values():
    assert mcnemar_exact_p_value(0, 0) == 1.0
    assert math.isclose(mcnemar_exact_p_value(0, 5), 2 / 32)  # all 5 discordant pairs go one way
    assert mcnemar_exact_p_value(5, 5) == 1.0
    assert math.isclose(mcnemar_exact_p_value(2, 8), 2 * 56 / 1024)  # 0.109: not significant at n=10
    assert mcnemar_exact_p_value(1, 9) < 0.05 and mcnemar_exact_p_value(4, 6) > 0.5


def test_paired_bootstrap_interval_contains_the_mean_and_is_deterministic():
    differences = [1] * 30 + [0] * 60 + [-1] * 10
    low, high = paired_bootstrap_interval(differences)
    assert low < 0.20 < high and (low, high) == paired_bootstrap_interval(differences)
    assert paired_bootstrap_interval([]) == (0.0, 0.0)
    assert paired_bootstrap_interval([0] * 50) == (0.0, 0.0)


def outcomes(pattern):
    """'1010' -> {(0, 0): True, (1, 0): False, ...}"""
    return {(task, 0): character == "1" for task, character in enumerate(pattern)}


def test_compare_outcomes_counts_wins_losses_and_only_pairs_shared_tasks():
    baseline, candidate = outcomes("1100" * 25), outcomes("1110" * 25)
    comparison = compare_outcomes(baseline, candidate)
    assert comparison["paired"] == 100 and comparison["b_better"] == 25
    assert comparison["a_better"] == 0 and comparison["same"] == 75
    assert math.isclose(comparison["delta"], 0.25) and comparison["p_value"] < 0.001
    assert comparison["delta_ci95"][0] > 0
    assert verdict(comparison) == "B is significantly better"
    worse = compare_outcomes(candidate, baseline)
    assert verdict(worse) == "B is significantly WORSE (regression)"
    partial = compare_outcomes(outcomes("1010"), {(0, 0): True, (1, 0): True, (99, 0): True})
    assert partial["paired"] == 2  # task 99 is not in A: not paired


def test_small_samples_never_claim_a_result():
    comparison = compare_outcomes(outcomes("0" * 10), outcomes("1" * 10))
    assert comparison["p_value"] < 0.01 and "too few" in verdict(comparison)
    same = compare_outcomes(outcomes("10" * 30), outcomes("10" * 30))
    assert verdict(same) == "no significant difference" and same["p_value"] == 1.0
    assert "p=1.000" in format_comparison("x", same)


def run_record(task, sample, passed, **changes):
    return {"task_id": task, "sample": sample, "passed": passed, "test_count": 3, "tests_valid": True,
            "failure_reason": "", "tool_errors": 0, "tool_calls": 4, "steps": 5,
            "stop_reason": "finished",
            "artifacts": {"tests": "def test_a():\n    assert f(1) == 3\n",
                          "review": "ok\nNo findings.", "code": "def f(x):\n    return 3*x\n"}, **changes}


def test_summary_reports_intervals_and_pipeline_health():
    runs = [run_record(task, 0, task % 2 == 0) for task in range(10)]
    runs[1]["failure_reason"] = "design: boom"
    runs[3]["tests_valid"] = False
    runs[5]["tool_errors"] = 2
    summary = summarize_runs(runs, Config())
    assert summary["runs"] == 10 and summary["pass_rate"] == 0.5
    low, high = summary["pass_rate_ci95"]
    assert low < 0.5 < high
    assert summary["pipeline_failure_rate"] == 0.1 and summary["test_gate_failure_rate"] == 0.1
    assert math.isclose(summary["tool_error_rate"], 2 / 40) and summary["mean_agent_steps"] == 5
    assert summarize_runs([], Config())["runs"] == 0


def test_compare_paths_reads_eval_directories(tmp_path):
    for name, passing_tasks in (("a", 40), ("b", 70)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "eval_runs.jsonl").write_text(
            "".join(json.dumps(run_record(task, 0, task < passing_tasks)) + "\n" for task in range(100)))
    report = compare_paths(str(tmp_path / "a"), str(tmp_path / "b"), Config())
    assert "passed" in report and "delta +30.0%" in report and "B is significantly better" in report
    assert "valid_tests" in report and "trusted" in report


def test_compare_paths_reads_humaneval_files(tmp_path):
    def humaneval_file(name, passing_problems):
        path = tmp_path / f"humaneval_{name}.json"
        path.write_text(json.dumps({"results": [
            {"task_id": f"HumanEval/{number}", "passed": number < passing_problems, "reason": "x"}
            for number in range(100)]}))
        return str(path)

    report = compare_paths(humaneval_file("direct_a", 70), humaneval_file("direct_b", 50), Config())
    assert "humaneval" in report and "WORSE" in report  # the forgetting alarm


def test_train_log_summary_detects_overfitting(tmp_path):
    def write(entries):
        path = tmp_path / "log.jsonl"
        path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        return path

    healthy = [{"step": step, "loss": 2.0 - step * 0.01} for step in (10, 20, 30)] + [
        {"step": step, "eval_loss": 2.0 - step * 0.008} for step in (10, 20, 30)]
    assert summarize_train_log(write(healthy))["overfitting"] is False
    overfit = [{"step": 10, "loss": 2.0}, {"step": 60, "loss": 0.4},
               {"step": 20, "eval_loss": 1.5}, {"step": 40, "eval_loss": 1.2},
               {"step": 60, "eval_loss": 1.6}, {"step": 80, "eval_loss": 1.9}]
    summary = summarize_train_log(write(overfit))
    assert summary["overfitting"] is True and summary["best_eval_step"] == 40 and summary["best_eval_loss"] == 1.2
