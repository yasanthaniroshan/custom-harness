"""The SDLC pipeline as a LangGraph graph.

    requirements -> design -> write_tests -> validate_tests --valid--> implement x N -> select -> review
                                   ^                |
                                   +--invalid, retries left (validator feedback)

Prose stages produce typed artifacts through constrained decoding (structured outputs). Each
stage still sees only its declared input artifacts (stages.py); on a retry the test gate adds
its findings to the write_tests prompt. `implement` is the tool-using subgraph
(implementer.py), fanned out with Send for best-of-N.
"""
from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import LengthFinishReasonError
from pydantic import BaseModel

from harness.pipeline import artifacts
from harness.pipeline.dependencies import PipelineDependencies
from harness.pipeline.implementer import CLEAR_CANDIDATES, build_implementer_graph, merge_candidates
from harness.pipeline.messages import message_text, to_chat_dict
from harness.pipeline.stages import DESIGN, REQUIREMENTS, REVIEW, WRITE_TESTS, Stage
from harness.prompt_context import truncate_text
from harness.sandbox import run_code_against_tests

CUT_OFF_REASON = "output was cut off at max_new_tokens"
PREVIOUS_SUITE_MAX_TOKENS = 1200
LAST_ATTEMPT_MAX_UNCOVERED_FRACTION = 0.5  # share of requirements allowed untested on the last attempt


class PipelineState(TypedDict, total=False):
    task: str
    # artifacts as prompt text, plus their structured form
    requirements: str
    requirements_data: dict
    design: str
    design_data: dict
    tests: str
    tests_data: dict
    # the test-suite gate
    test_attempts: int
    test_issues: list[str]
    tests_valid: bool
    # implement candidates and the selected one
    candidates: Annotated[list[dict], merge_candidates]
    candidate_count: int
    candidates_passed: int
    code: str
    diff: str
    test_output: str
    passed: bool
    steps: int
    test_count: int
    tool_calls: int
    tool_errors: int
    stop_reason: str
    review: str
    review_data: dict
    # one record per model call, later turned into training rows
    llm_calls: Annotated[list[dict], operator.add]
    failure_reason: str


def call_structured(dependencies: PipelineDependencies, schema: type[BaseModel], messages: list,
                    temperature: float, runnable_config: RunnableConfig | None
                    ) -> tuple[BaseModel | None, AIMessage, str | None]:
    """One constrained-decoding call. Returns (parsed or None, raw reply, failure reason or None)."""
    model = dependencies.chat_models.get(temperature).with_structured_output(
        schema, method=dependencies.config.serve.structured_output_method, include_raw=True)
    try:
        output = model.invoke(messages, runnable_config)
    except LengthFinishReasonError as error:
        # The OpenAI client raises (instead of returning a parsing error) when the reply hit
        # max_tokens. Keep the partial text for the record and fail with a clear reason.
        partial_text = error.completion.choices[0].message.content or ""
        return None, AIMessage(content=partial_text), CUT_OFF_REASON
    raw_reply, parsed, parsing_error = output["raw"], output.get("parsed"), output.get("parsing_error")
    if parsed is None:
        finish_reason = (getattr(raw_reply, "response_metadata", {}) or {}).get("finish_reason")
        reason = CUT_OFF_REASON if finish_reason == "length" else f"unparseable output ({parsing_error})"
        return None, raw_reply, reason
    return parsed, raw_reply, None


def build_pipeline_graph(dependencies: PipelineDependencies, checkpointer=None):
    pipeline_config = dependencies.config
    graph_config = pipeline_config.graph
    system_prompt = pipeline_config.agent.system_prompt

    def structured_stage(stage: Stage, schema: type[BaseModel], render: Callable[[Any], str],
                         validate: Callable[[Any], list[str]] | None, temperature: float,
                         extra_prompt: Callable[[PipelineState], str] | None = None):
        """A node that asks for `schema`, validates it, and stores text + structured artifact."""
        text_key, data_key = stage.output, f"{stage.output}_data"
        is_test_stage = stage is WRITE_TESTS

        def node(state: PipelineState, config: RunnableConfig) -> dict:
            # LangGraph passes the run's RunnableConfig to a parameter named `config`
            prompt = dependencies.prompt_builder.build_prompt(stage, state)
            if extra_prompt:
                prompt += extra_prompt(state)
            messages = [SystemMessage(system_prompt), HumanMessage(prompt)]
            parsed, raw_reply, failure = call_structured(dependencies, schema, messages, temperature, config)
            attempt = state.get("test_attempts", 0) + 1 if is_test_stage else 1
            call_record = {"stage": stage.name, "attempt": attempt,
                           "messages": [to_chat_dict(message) for message in messages],
                           "completion": message_text(raw_reply.content)}
            update: dict = {"llm_calls": [call_record]}
            if is_test_stage:
                update["test_attempts"] = attempt
            issues = [failure] if parsed is None else (validate(parsed) if validate else [])
            if issues:
                return {**update, "failure_reason": f"{stage.name}: " + "; ".join(issues)}
            if stage is DESIGN:
                for warning in artifacts.design_warnings(parsed):
                    dependencies.tracer.emit("artifact_warning", stage_name=stage.name, warning=warning)
            return {**update, text_key: render(parsed), data_key: parsed.model_dump()}

        return node

    def previous_suite_feedback(state: PipelineState) -> str:
        issues = state.get("test_issues")
        if not issues:
            return ""
        previous_suite = truncate_text(state.get("tests", ""), PREVIOUS_SUITE_MAX_TOKENS,
                                       dependencies.prompt_builder.count_tokens, "middle")
        return ("\n\nYour previous test suite:\n```python\n" + previous_suite + "\n```\n"
                "It was rejected for these problems:\n- " + "\n- ".join(issues) +
                "\nWrite a corrected, complete test suite.")

    def run_against_empty_code(tests: str) -> dict[str, bool]:
        report = run_code_against_tests("", tests, pipeline_config.agent.test_timeout_seconds)
        return {case.name: case.passed for case in report.cases}

    def validate_tests(state: PipelineState) -> dict:
        requirement_count = len(artifacts.Requirements.model_validate(state["requirements_data"]).requirements)
        suite = artifacts.TestSuite.model_validate(state["tests_data"])
        attempt = state["test_attempts"]
        # Retries demand a test for every requirement; the last attempt accepts a suite that
        # still covers at least half of them (the gap is recorded) rather than discarding the run.
        is_last_attempt = attempt > graph_config.max_test_retries
        issues = artifacts.validate_test_suite(
            suite, requirement_count, graph_config.min_tests, run_against_empty_code,
            graph_config.max_stub_pass_fraction,
            max_uncovered_fraction=LAST_ATTEMPT_MAX_UNCOVERED_FRACTION if is_last_attempt else 0.0)
        uncovered_ids = artifacts.uncovered_requirements(suite, requirement_count)
        if not issues and uncovered_ids:
            dependencies.tracer.emit("artifact_warning", stage_name="write_tests",
                                     warning=f"accepted without a test for {', '.join(uncovered_ids)}")
        update: dict = {"test_issues": issues, "tests_valid": not issues,
                        "llm_calls": [{"stage": "write_tests_gate", "attempt": attempt, "valid": not issues}]}
        if issues and attempt > graph_config.max_test_retries and graph_config.abort_on_invalid_tests:
            update["failure_reason"] = (f"write_tests: invalid test suite after {attempt} attempts: "
                                        + "; ".join(issues))
        return update

    def route_after_test_gate(state: PipelineState):
        if state.get("failure_reason"):
            return END
        if state["test_issues"] and state["test_attempts"] <= graph_config.max_test_retries:
            return "write_tests"  # the gate said no: retry with the validator's feedback
        return [Send("implement", {"candidate_design": state["design"], "candidate_tests": state["tests"],
                                   "candidate_index": index})
                for index in range(graph_config.candidates)]

    def select_best_candidate(state: PipelineState) -> dict:
        candidates = state.get("candidates") or []
        if not candidates:
            return {"failure_reason": "implement: no candidate finished", "candidates": CLEAR_CANDIDATES}

        def ranking(candidate: dict) -> tuple:
            pass_fraction = candidate["tests_passed_count"] / max(candidate["test_count"], 1)
            return (candidate["passed"], pass_fraction, -candidate["tool_errors"],
                    -candidate["steps"], -len(candidate["diff"]))

        best = max(candidates, key=ranking)
        implement_calls = [
            {"stage": "implement", "attempt": turn_number, "messages": turn["messages"],
             "completion": turn["completion"]["content"], "completion_message": turn["completion"],
             "tool_schemas": best["tool_schemas"], "passed": best["passed"]}
            for turn_number, turn in enumerate(best["turns"], 1)
        ]
        return {
            "code": best["code"], "diff": best["diff"], "test_output": best["test_output"],
            "passed": best["passed"], "steps": best["steps"], "test_count": best["test_count"],
            "tool_calls": best["tool_calls"], "tool_errors": best["tool_errors"],
            "stop_reason": best["stop_reason"], "candidate_count": len(candidates),
            "candidates_passed": sum(candidate["passed"] for candidate in candidates),
            "llm_calls": implement_calls,
            "candidates": CLEAR_CANDIDATES,  # the losing trajectories are bulky; drop them
        }

    def route_after_select(state: PipelineState) -> str:
        if state.get("failure_reason"):
            return "end"
        return "review" if state["passed"] or not dependencies.review_only_passed else "end"

    def continue_unless_failed(state: PipelineState) -> str:
        return "end" if state.get("failure_reason") else "continue"

    graph = StateGraph(PipelineState)
    graph.add_node("requirements", structured_stage(
        REQUIREMENTS, artifacts.Requirements, artifacts.render_requirements,
        artifacts.validate_requirements, dependencies.prose_temperature))
    graph.add_node("design", structured_stage(
        DESIGN, artifacts.Design, artifacts.render_design, artifacts.validate_design,
        dependencies.prose_temperature))
    graph.add_node("write_tests", structured_stage(
        WRITE_TESTS, artifacts.TestSuite, artifacts.render_tests, None,
        dependencies.code_temperature, previous_suite_feedback))
    graph.add_node("validate_tests", validate_tests)
    graph.add_node("implement", build_implementer_graph(dependencies))
    graph.add_node("select", select_best_candidate)
    graph.add_node("review", structured_stage(
        REVIEW, artifacts.Review, artifacts.render_review, None, dependencies.prose_temperature))

    graph.add_edge(START, "requirements")
    for current_node, next_node in [("requirements", "design"), ("design", "write_tests"),
                                    ("write_tests", "validate_tests")]:
        graph.add_conditional_edges(current_node, continue_unless_failed,
                                    {"continue": next_node, "end": END})
    graph.add_conditional_edges("validate_tests", route_after_test_gate, ["write_tests", "implement", END])
    graph.add_edge("implement", "select")
    graph.add_conditional_edges("select", route_after_select, {"review": "review", "end": END})
    graph.add_edge("review", END)
    return graph.compile(checkpointer=checkpointer, name="sdlc")
