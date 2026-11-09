"""Shared test doubles: a fake LangChain chat model driven by a scripted Scenario, canned
artifacts for a tiny task ("triple a number"), and helpers for data/training tests."""
from __future__ import annotations

import json
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from harness.config import Config
from harness.monitoring import Tracer
from harness.pipeline import artifacts
from harness.pipeline.chat_models import ChatModelFactory
from harness.pipeline.dependencies import PipelineDependencies
from harness.prompt_context import PromptBuilder

GOOD_CODE = "def f(x):\n    return x * 3\n"
BAD_CODE = "def f(x):\n    return x * 2\n"
HIDDEN_TESTS = (
    "def test_triples():\n    assert f(14) == 42\n"
    "def test_zero():\n    assert f(0) == 0\n"
    "def test_negative():\n    assert f(-2) == -6\n"
)
TRIVIAL_TESTS = "def test_a():\n    assert True\ndef test_b():\n    assert 1 == 1\ndef test_c():\n    assert not None\n"
# Strings that would reveal the hidden tests if they appeared in the implementer's context.
TEST_LEAK_MARKERS = ("assert f(", "f(14)", "== 42", "test_triples():", "def test_")

REQUIREMENTS = artifacts.Requirements(
    requirements=["f(x) returns x tripled", "f(0) is 0", "negative inputs are tripled too"],
    edge_cases=["zero", "negative numbers"], assumptions=["inputs are ints"])
DESIGN = artifacts.Design(
    functions=[artifacts.FunctionSpec(name="f", signature="def f(x: int) -> int", behavior="returns 3 * x")],
    approach="multiply by three", decisions=["no input validation"])
GOOD_SUITE = artifacts.TestSuite(tests=[
    artifacts.TestCase(name="test_triples", covers=["R1"], code="def test_triples():\n    assert f(14) == 42\n"),
    artifacts.TestCase(name="test_zero", covers=["R2"], code="def test_zero():\n    assert f(0) == 0\n"),
    artifacts.TestCase(name="test_negative", covers=["R3"], code="def test_negative():\n    assert f(-2) == -6\n"),
])
UNCOVERED_SUITE = artifacts.TestSuite(tests=GOOD_SUITE.tests[:2] + [  # nothing covers R3
    artifacts.TestCase(name="test_extra", covers=["R1"], code="def test_extra():\n    assert f(1) == 3\n")])
TRIVIAL_SUITE = artifacts.TestSuite(tests=[
    artifacts.TestCase(name=f"test_t{number}", covers=[f"R{number}"], code=f"def test_t{number}():\n    assert True\n")
    for number in (1, 2, 3)])
REVIEW_OK = artifacts.Review(summary="Triples its input.", findings=[])


def tool_call(name: str, **arguments: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": uuid.uuid4().hex[:8]}])


class Scenario:
    """Responder for FakeChat. Structured stages return canned artifacts; the tool loop follows
    a script chosen by how many assistant turns the conversation already has."""

    def __init__(self, suites=(GOOD_SUITE,), implementation="fix", review=REVIEW_OK, fail_review_once=False):
        self.suites = list(suites)
        self.implementation = implementation
        self.review = review
        self.fail_review_once = fail_review_once
        self.lock = threading.Lock()
        self.calls: list[tuple[str, int]] = []  # (schema name or "tool_turn", message count)
        self.test_suite_requests = 0
        self.candidates_started = 0
        self.prompts: dict[str, list[str]] = {}

    def __call__(self, messages, tools, schema) -> AIMessage:
        with self.lock:
            kind = schema.__name__ if schema else "tool_turn"
            self.calls.append((kind, len(messages)))
            self.prompts.setdefault(kind, []).append(str(messages[-1].content))
        if schema is artifacts.Requirements:
            return AIMessage(content=REQUIREMENTS.model_dump_json())
        if schema is artifacts.Design:
            return AIMessage(content=DESIGN.model_dump_json())
        if schema is artifacts.TestSuite:
            with self.lock:
                suite = self.suites[min(self.test_suite_requests, len(self.suites) - 1)]
                self.test_suite_requests += 1
            return AIMessage(content=suite.model_dump_json())
        if schema is artifacts.Review:
            with self.lock:
                if self.fail_review_once:
                    self.fail_review_once = False
                    raise RuntimeError("simulated crash in review")
            return AIMessage(content=self.review.model_dump_json())
        return self._tool_turn(messages)

    def _tool_turn(self, messages) -> AIMessage:
        previous_replies = [message for message in messages if isinstance(message, AIMessage)]
        if callable(self.implementation):
            return self.implementation(messages, previous_replies)
        turn = len(previous_replies)
        if self.implementation == "fix":  # writes wrong code, sees FAIL, fixes it, re-checks
            script = [tool_call("write_file", path="solution.py", content=BAD_CODE), tool_call("run_tests"),
                      tool_call("str_replace", path="solution.py", old_string="x * 2", new_string="x * 3"),
                      tool_call("run_tests"), AIMessage(content="All tests pass.")]
        elif self.implementation == "good":
            script = [tool_call("write_file", path="solution.py", content=GOOD_CODE), tool_call("run_tests"),
                      AIMessage(content="Done.")]
        elif self.implementation == "best_of_n":  # candidates differ: only the second is right
            if turn == 0:
                with self.lock:
                    candidate_number = self.candidates_started
                    self.candidates_started += 1
                return tool_call("write_file", path="solution.py",
                                 content=GOOD_CODE if candidate_number == 1 else BAD_CODE)
            wrote_good_code = "x * 3" in str(previous_replies[0].tool_calls[0]["args"]["content"])
            script = [None, tool_call("run_tests"), AIMessage(content="Done." if wrote_good_code else "Gave up.")]
        elif self.implementation == "malformed":
            return AIMessage(content="<tool_call>{this is not json}</tool_call>")
        else:
            raise ValueError(self.implementation)
        return script[min(turn, len(script) - 1)]


class FakeChat(BaseChatModel):
    responder: Any = None
    tools: list = Field(default_factory=list)
    structured_schema: Any = None
    finish_reason: str = "stop"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self.responder(messages, self.tools, self.structured_schema)
        return ChatResult(generations=[ChatGeneration(
            message=reply, generation_info={"finish_reason": self.finish_reason})])

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools": list(tools)})

    def with_structured_output(self, schema, method=None, include_raw=False, **kwargs):
        bound = self.model_copy(update={"structured_schema": schema})

        def run(messages, config=None):
            raw = bound.invoke(messages, config)
            try:
                return {"raw": raw, "parsed": schema.model_validate_json(raw.content), "parsing_error": None}
            except ValueError as error:
                return {"raw": raw, "parsed": None, "parsing_error": error}

        return RunnableLambda(run)


def make_dependencies(scenario: Scenario, config: Config | None = None, tracer: Tracer | None = None,
                      review_only_passed: bool = True, finish_reason: str = "stop") -> PipelineDependencies:
    config = config or Config()
    return PipelineDependencies(
        config=config,
        chat_models=ChatModelFactory(
            config, build_model=lambda temperature: FakeChat(responder=scenario, finish_reason=finish_reason)),
        prompt_builder=PromptBuilder(config.context),
        tracer=tracer or Tracer(),
        prose_temperature=0.7,
        code_temperature=0.2,
        review_only_passed=review_only_passed,
    )


def make_runs(task_count: int = 6, samples_per_task: int = 1, scenario: Scenario | None = None) -> list[dict]:
    """Run records produced by the fake pipeline end to end."""
    from harness.pipeline.runner import open_pipeline_graph, read_jsonl, run_tasks

    dependencies = make_dependencies(scenario or Scenario())
    tasks = [{"task_id": number, "task": f"task {number}"} for number in range(task_count)]
    runs_path = Path(tempfile.mkdtemp()) / "runs.jsonl"
    with open_pipeline_graph(dependencies) as (graph, checkpointed):
        run_tasks(graph, dependencies.config, tasks, samples_per_task, runs_path, "test",
                  Tracer(), checkpointed=checkpointed, log=lambda *_: None)
    return read_jsonl(runs_path)


def make_sft_rows(task_count: int = 6) -> list[dict]:
    """SFT rows produced by the fake pipeline end to end (graph -> runs -> filter)."""
    from harness.training_data.trajectory_filter import filter_runs

    return filter_runs(make_runs(task_count), Config().filter)[0]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def cached_qwen_tokenizer():
    """The Qwen tokenizer from the LOCAL cache only (tests never download); skips if absent."""
    import pytest
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not in the local Hugging Face cache (tests never download)")
