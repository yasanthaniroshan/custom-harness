import copy

from harness.config import FilterConfig
from harness.pipeline import artifacts
from harness.training_data.seed_tasks import split_tasks
from harness.training_data.trajectory_filter import filter_runs, rejection_reason, run_to_rows
from tests.fakes import TRIVIAL_TESTS, Scenario, make_runs


def test_seed_task_split_is_deterministic_disjoint_and_eval_is_stable():
    rows = [{"index": number, "lang": "python" if number % 2 else "cpp", "problem": "p" * (number % 7 + 1)}
            for number in range(200)]
    train, evaluation = split_tasks(rows, 10, 20, seed=1, language="python", max_chars=5)
    assert {task["task_id"] for task in train}.isdisjoint(task["task_id"] for task in evaluation)
    assert all(task["task_id"] % 2 for task in train + evaluation)  # python only
    assert all(len(task["task"]) <= 5 for task in train + evaluation)
    larger_train, same_evaluation = split_tasks(rows, 10, 30, seed=1, language="python", max_chars=5)
    assert evaluation == same_evaluation and train == larger_train[:20]  # growing train never moves eval


def trusted_run() -> dict:
    return make_runs(task_count=1)[0]


def test_rejection_reasons():
    config = FilterConfig()
    good = trusted_run()
    assert rejection_reason(good, config) is None
    cases = {
        "pipeline_failed": {"failure_reason": "design: boom"},
        "tests_gate_failed": {"tests_valid": False},
        "tests_failed": {"passed": False},
        "tool_errors": {"tool_errors": 4},
        "too_few_tests": {"test_count": 2},
    }
    for expected, changes in cases.items():
        assert rejection_reason({**good, **changes}, config) == expected, expected

    trivial = copy.deepcopy(good)
    trivial["artifacts"]["tests"] = TRIVIAL_TESTS
    assert rejection_reason(trivial, config) == "tests_trivial"
    unusable = copy.deepcopy(good)
    unusable["artifacts"]["tests"] = "def test_x(:\n"
    assert rejection_reason(unusable, config) == "tests_unusable"
    blocker = copy.deepcopy(good)
    blocker["artifacts"]["review"] = "Looks off.\n[Blocker] f - wrong sign Fix: negate"
    assert rejection_reason(blocker, config) == "review_blocker"
    assert rejection_reason(blocker, FilterConfig(reject_blockers=False)) is None


def test_filter_deduplicates_code_and_caps_runs_per_task():
    runs = make_runs(task_count=3, samples_per_task=4)  # 4 identical solutions per task
    rows, stats = filter_runs(runs, FilterConfig(max_runs_per_task=2))
    assert stats["kept"] == 3 and stats["duplicate_code"] == 9  # one distinct solution per task
    assert len(rows) == 3 * (4 + 5)  # four prose/test rows + five implement turns per kept run
    assert all(row["prompt"][0]["role"] == "system" and row["completion"][0]["role"] == "assistant"
               for row in rows)


def test_implement_rows_come_only_from_the_passing_candidate():
    runs = make_runs(task_count=1, scenario=Scenario(implementation="fix"))
    rows, _ = filter_runs(runs, FilterConfig())
    implement_rows = [row for row in rows if row["stage"] == "implement"]
    assert len(implement_rows) == 5 and all("tools" in row for row in implement_rows)
    failed_candidate = copy.deepcopy(runs[0])
    for call in failed_candidate["llm_calls"]:
        call["passed"] = False if call["stage"] == "implement" else call.get("passed")
    assert not [row for row in run_to_rows(failed_candidate) if row["stage"] == "implement"]


def test_structured_stage_rows_hold_schema_json():
    rows, _ = filter_runs(make_runs(task_count=1), FilterConfig())
    design_row = next(row for row in rows if row["stage"] == "design")
    artifacts.Design.model_validate_json(design_row["completion"][0]["content"])  # does not raise
