"""HumanEval: an external yardstick that does not depend on tests the model wrote itself.

Two modes, both against the running vLLM server:
  direct   one plain chat completion per problem (no harness). Run it for the base model AND
           the fine-tuned one: a drop means fine-tuning damaged general coding ability.
  harness  the full SDLC pipeline, scored by the official HumanEval tests (hidden from it).
The dataset (openai/openai_humaneval, 164 problems) is downloaded at runtime.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.config import Config
from harness.evaluation.statistics import wilson_interval
from harness.sandbox import run_code_against_tests

DIRECT_PROMPT = ("Complete the following Python function. Return the complete function "
                 "(signature and body) in one ```python block.\n\n{prompt}")
HARNESS_TASK = ("Implement this Python function. It must be named `{entry_point}` and keep this "
                "signature.\n\n{prompt}")
MAX_REASON_CHARS = 40
TOP_FAILURE_REASONS = 8
_CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """The first fenced code block, or the raw text if there is none."""
    blocks = _CODE_FENCE.findall(text)
    return blocks[0] if blocks else text


def load_humaneval(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    problems = [dict(row) for row in load_dataset("openai/openai_humaneval", split="test")]
    return problems[:limit] if limit else problems


def program_for(problem: dict, code: str) -> str:
    """The program to test. The prompt (imports, helpers, a docstring-only stub of the entry
    function) comes first, so a model that returns only a body, or only the entry function, is
    still scored fairly; the model's own definition then overrides the stub."""
    entry_point = re.escape(problem["entry_point"])
    defines_entry_point = re.search(rf"^\s*(async\s+)?def\s+{entry_point}\s*\(", code, re.MULTILINE)
    if defines_entry_point:
        return problem["prompt"] + "\n\n" + code
    return problem["prompt"] + code  # a body-only completion


def score(problem: dict, code: str | None, timeout_seconds: int = 10) -> tuple[bool, str]:
    """(passed, reason). The official `check` runs after the model's code, so it cannot be faked."""
    if not code or not code.strip():
        return False, "no_code"
    tests = problem["test"] + f"\n\ndef test_humaneval():\n    check({problem['entry_point']})\n"
    report = run_code_against_tests(program_for(problem, code), tests, timeout_seconds)
    if report.passed:
        return True, "ok"
    failing_case = next((case for case in report.cases if not case.passed), None)
    if failing_case is not None:
        return False, failing_case.error or "failed"  # e.g. AssertionError, NameError: name 'x' ...
    return False, report.summary(200).splitlines()[0]  # load error, timeout or crash


def summarize_results(results: list[dict]) -> dict:
    problem_count = len(results)
    passed = sum(result["passed"] for result in results)
    failure_reasons: dict[str, int] = {}
    for result in results:
        if not result["passed"]:
            reason = result["reason"][:MAX_REASON_CHARS]
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    most_common = sorted(failure_reasons.items(), key=lambda item: -item[1])[:TOP_FAILURE_REASONS]
    return {"problems": problem_count, "passed": passed,
            "pass_at_1": passed / problem_count if problem_count else 0.0,
            "ci95": list(wilson_interval(passed, problem_count)),
            "failure_reasons": dict(most_common)}


def run_direct(config: Config, problems: list[dict], adapter_path: str | None = None,
               chat_model=None) -> list[dict]:
    """One greedy completion per problem through the plain chat API."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from harness.pipeline.chat_models import ChatModelFactory

    chat_model = chat_model or ChatModelFactory(config, adapter_path).get(0.0)

    def solve(problem: dict) -> dict:
        reply = chat_model.invoke([SystemMessage(config.agent.system_prompt),
                                   HumanMessage(DIRECT_PROMPT.format(prompt=problem["prompt"]))])
        passed, reason = score(problem, extract_code_block(str(reply.content)),
                               config.agent.test_timeout_seconds)
        return {"task_id": problem["task_id"], "passed": passed, "reason": reason}

    with ThreadPoolExecutor(config.serve.concurrency) as pool:
        return list(pool.map(solve, problems))


def results_from_runs(config: Config, problems: list[dict], runs: list[dict]) -> list[dict]:
    """Score pipeline runs with the official tests. Problems with no run count as failures."""
    runs_by_task = {run["task_id"]: run for run in runs}
    results = []
    for problem_index, problem in enumerate(problems):
        run = runs_by_task.get(problem_index)
        if run is None:
            results.append({"task_id": problem["task_id"], "passed": False, "reason": "no_run"})
            continue
        passed, reason = score(problem, run["artifacts"].get("code"), config.agent.test_timeout_seconds)
        failure_reason = run.get("failure_reason")
        if failure_reason and not passed:
            reason = f"pipeline: {failure_reason[:30]}"
        results.append({"task_id": problem["task_id"], "passed": passed, "reason": reason})
    return results


def run_harness(config: Config, problems: list[dict], output_dir: Path,
                adapter_path: str | None = None) -> list[dict]:
    from harness.pipeline.runner import read_jsonl
    from harness.pipeline.service import RunSettings, run_to_directory

    tasks = [{"task_id": index, "task": HARNESS_TASK.format(entry_point=problem["entry_point"],
                                                            prompt=problem["prompt"])}
             for index, problem in enumerate(problems)]
    runs_path = run_to_directory(config, tasks, output_dir, adapter_path, RunSettings(
        samples_per_task=1,
        prose_temperature=config.generate.eval_temperature,
        code_temperature=config.agent.code_temperature,
        review_only_passed=True,
        round_name=f"{output_dir.name}-humaneval",
        events_file_name="humaneval_events.jsonl",
        runs_file_name="humaneval_runs.jsonl",
    ))
    return results_from_runs(config, problems, read_jsonl(runs_path))


def save_results(output_dir: Path, mode: str, adapter_path: str | None, results: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"humaneval_{mode}.json"
    path.write_text(json.dumps({"mode": mode, "adapter": adapter_path, "summary": summarize_results(results),
                                "results": results}, indent=2))
    return path
