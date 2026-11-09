from langchain_core.messages import AIMessage

from harness.config import Config
from harness.evaluation.humaneval import (
    extract_code_block,
    program_for,
    results_from_runs,
    run_direct,
    score,
    summarize_results,
)

# Same structure as real HumanEval problems: imports + docstring-only stub, a `check(candidate)`.
ADD_PROBLEM = {
    "task_id": "HumanEval/0", "entry_point": "add",
    "prompt": 'from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    """Sum the list."""\n',
    "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == 6\n    assert candidate([]) == 0\n",
}
HELPER_PROBLEM = {
    "task_id": "HumanEval/1", "entry_point": "decode",
    "prompt": ('def encode(s):\n    return s[::-1]\n\n\ndef decode(s):\n    """Inverse of encode."""\n'),
    "test": "def check(candidate):\n    assert candidate(encode('abc')) == 'abc'\n",
}


def test_full_function_body_only_and_import_free_answers_are_all_scored_fairly():
    full_function = "def add(xs):\n    return sum(xs)\n"  # no `from typing import List`: the prompt provides it
    assert score(ADD_PROBLEM, full_function) == (True, "ok")
    assert score(ADD_PROBLEM, "    return sum(xs)\n")[0]  # body-only completion
    assert score(ADD_PROBLEM, "def add(xs: List[int]) -> int:\n    return sum(xs)\n")[0]  # needs the prompt's import
    assert score(HELPER_PROBLEM, "def decode(s):\n    return s[::-1]\n")[0]  # helper defined only in the prompt


def test_wrong_empty_crashing_and_hanging_answers_fail_with_a_reason():
    passed, reason = score(ADD_PROBLEM, "def add(xs):\n    return 0\n")
    assert not passed and "AssertionError" in reason
    assert score(ADD_PROBLEM, "")[1] == "no_code" and score(ADD_PROBLEM, None) == (False, "no_code")
    assert "SyntaxError" in score(ADD_PROBLEM, "def add(:\n")[1] or "failed to load" in score(ADD_PROBLEM, "def add(:\n")[1]
    passed, reason = score(ADD_PROBLEM, "def add(xs):\n    while True: pass\n", timeout_seconds=1)
    assert not passed and "imed out" in reason


def test_program_prepends_the_prompt_and_keeps_the_model_definition_last():
    program = program_for(ADD_PROBLEM, "def add(xs):\n    return sum(xs)\n")
    assert program.startswith(ADD_PROBLEM["prompt"]) and program.rstrip().endswith("return sum(xs)")
    assert program_for(ADD_PROBLEM, "    return 1\n") == ADD_PROBLEM["prompt"] + "    return 1\n"


def test_cheating_by_redefining_check_is_not_possible():
    # the official `check` is appended AFTER the model's code, so redefining it has no effect
    redefines_check = "def check(candidate):\n    pass\n\ndef add(xs):\n    return 0\n"
    assert not score(ADD_PROBLEM, redefines_check)[0]


class FakeChatModel:
    def __init__(self, answers):
        self.answers = answers

    def invoke(self, messages):
        prompt = messages[-1].content
        problem = "decode" if "def decode" in prompt else "add"
        return AIMessage(content=self.answers[problem])


def test_run_direct_scores_each_problem_through_the_chat_api():
    chat_model = FakeChatModel({"add": "```python\ndef add(xs):\n    return sum(xs)\n```",
                   "decode": "```python\ndef decode(s):\n    return s\n```"})  # wrong
    results = run_direct(Config(), [ADD_PROBLEM, HELPER_PROBLEM], chat_model=chat_model)
    assert [(result["task_id"], result["passed"]) for result in results] == [("HumanEval/0", True), ("HumanEval/1", False)]
    summary = summarize_results(results)
    assert summary["problems"] == 2 and summary["passed"] == 1 and summary["pass_at_1"] == 0.5
    assert summary["ci95"][0] < 0.5 < summary["ci95"][1]


def test_harness_runs_are_scored_by_the_official_tests_and_missing_runs_count_as_failures():
    runs = [{"task_id": 0, "artifacts": {"code": "def add(xs):\n    return sum(xs)\n"}, "failure_reason": ""},
            {"task_id": 1, "artifacts": {}, "failure_reason": "design: unparseable output (x)"}]
    results = results_from_runs(Config(), [ADD_PROBLEM, HELPER_PROBLEM, {**ADD_PROBLEM, "task_id": "HumanEval/2"}], runs)
    assert [result["passed"] for result in results] == [True, False, False]
    assert results[1]["reason"].startswith("pipeline: design") and results[2]["reason"] == "no_run"


def test_extract_code_block_takes_the_first_fenced_block_or_the_raw_text():
    assert extract_code_block("hi\n```python\nx = 1\n```\nbye") == "x = 1\n"
    assert extract_code_block("no fence") == "no fence"
