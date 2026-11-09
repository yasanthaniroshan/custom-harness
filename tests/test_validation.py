import copy
import json

import pytest

from harness.training_data.sft_dataset import is_validation_task
from harness.training_data.validation import validate_rows
from tests.fakes import UNCOVERED_SUITE, cached_qwen_tokenizer, make_sft_rows


@pytest.fixture(scope="module")
def rows():
    return make_sft_rows(6)


def codes(report, severity=None):
    return {issue.code for issue in report.issues if severity in (None, issue.severity)}


def mutate(rows, stage, change, which=0):
    """A deep copy of `rows` with `change` applied to the `which`-th row of `stage`."""
    rows = copy.deepcopy(rows)
    target = [row for row in rows if row["stage"] == stage][which]
    change(target)
    return rows


def test_clean_rows_are_ok_and_stats_count_stages(rows):
    report = validate_rows(rows, min_rows=1)
    assert report.ok and not report.errors, report.format()
    assert report.stats["tasks"] == 6 and report.stats["rows_by_stage"]["requirements"] == 6
    assert report.stats["rows_by_stage"]["implement"] == 30  # 5 turns per task
    assert "RESULT: OK to train" in report.format()


def test_string_tool_arguments_are_caught_because_the_template_double_encodes_them(rows):
    def to_string(row):
        function = row["completion"][0]["tool_calls"][0]["function"]
        function["arguments"] = json.dumps(function["arguments"])

    bad = mutate(rows, "implement", to_string)
    assert "tool_args_not_dict" in codes(validate_rows(bad, min_rows=1), "error")


def test_structural_errors(rows):
    cases = {
        "orphan_tool_message": lambda row: row["prompt"].insert(1, {"role": "tool", "content": "x"}),
        "bad_last_prompt_role": lambda row: row["prompt"].append({"role": "assistant", "content": "x"}),
        "bad_completion": lambda row: row.__setitem__("completion", []),
        "empty_completion": lambda row: row.__setitem__("completion", [{"role": "assistant", "content": ""}]),
        "bad_message": lambda row: row["prompt"].__setitem__(0, {"role": "robot", "content": "x"}),
        "unknown_tool": lambda row: row["completion"][0]["tool_calls"][0]["function"].__setitem__("name", "rm_rf"),
        "bad_shape": lambda row: row.__setitem__("prompt", []),
    }
    for code, change in cases.items():
        assert code in codes(validate_rows(mutate(rows, "implement", change), min_rows=1), "error"), code


def test_typed_stage_completions_must_match_their_schema(rows):
    bad = mutate(rows, "design", lambda row: row["completion"][0].__setitem__("content", '{"functions": 5}'))
    report = validate_rows(bad, min_rows=1)
    assert "bad_structured_completion" in codes(report, "error")
    broken = mutate(rows, "requirements", lambda row: row["completion"][0].__setitem__("content", "not json"))
    assert "bad_structured_completion" in codes(validate_rows(broken, min_rows=1), "error")


def test_hidden_test_assertions_in_an_implement_row_are_a_leak(rows):
    def leak(row):
        row["prompt"][-1]["content"] += "\nhint: assert f(14) == 42"

    report = validate_rows(mutate(rows, "implement", leak), min_rows=1)
    assert "hidden_test_leak" in codes(report, "error")
    assert any("assert f(14) == 42" in str(example) for issue in report.issues for example in issue.examples)


def test_a_training_suite_that_fails_the_gate_is_an_error(rows):
    bad = mutate(rows, "write_tests",
                 lambda row: row["completion"][0].__setitem__("content", UNCOVERED_SUITE.model_dump_json()))
    assert "suite_fails_gate" in codes(validate_rows(bad, min_rows=1), "error")


def test_warnings_for_duplicates_incomplete_tasks_and_small_data(rows):
    dup = rows + copy.deepcopy(rows[:3])
    assert "duplicate_rows" in codes(validate_rows(dup, min_rows=1), "warning")
    no_review = [row for row in rows if row["stage"] != "review"]
    assert "incomplete_tasks" in codes(validate_rows(no_review, min_rows=1), "warning")
    assert "small_dataset" in codes(validate_rows(rows, min_rows=10_000), "warning")
    implement_only = [row for row in rows if row["stage"] == "implement"]
    assert "implement_dominates" in codes(validate_rows(implement_only, min_rows=1), "warning")
    no_system = mutate(rows, "requirements", lambda row: row["prompt"].pop(0))
    assert "no_system_prompt" in codes(validate_rows(no_system, min_rows=1), "warning")


def test_validation_split_is_reported_and_task_based(rows):
    report = validate_rows(rows, validation_fraction=0.5, seed=42, min_rows=1)
    assert report.stats["validation_tasks"] == sum(is_validation_task(task, 0.5, 42) for task in range(6))
    none = validate_rows(rows, validation_fraction=1e-9, min_rows=1)  # effectively no validation tasks
    assert none.stats["validation_tasks"] == 0 and "no_validation_tasks" in codes(none, "warning")


def test_real_chat_template_rendering_token_stats_and_loss_share(rows):
    tokenizer = cached_qwen_tokenizer()
    report = validate_rows(rows, tokenizer, max_length=8192, min_rows=1)
    assert report.ok, report.format()
    tokens = report.stats["tokens"]
    assert tokens["p50"] <= tokens["p90"] <= tokens["p99"] <= tokens["max"]
    assert tokens["total"] > tokens["completion_total"] > 0
    # long repeated tool schemas + history dominate: the validator must flag it
    assert tokens["loss_token_share"] < 0.10 and "low_loss_token_share" in codes(report, "warning")
    tight = validate_rows(rows, tokenizer, max_length=500, min_rows=1)
    assert "overlong_rows" in codes(tight, "warning") and tight.stats["rows_over_max_length"]
