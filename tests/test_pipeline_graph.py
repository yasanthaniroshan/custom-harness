import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from harness.config import Config
from harness.monitoring import Tracer, summarize_events
from harness.pipeline import artifacts
from harness.pipeline.graph import build_pipeline_graph
from harness.pipeline.messages import ELIDED_TOOL_OUTPUT, compact_messages, to_chat_dict
from harness.pipeline.runner import state_to_run
from harness.pipeline.tracing import TracerCallback, run_config
from harness.training_data.trajectory_filter import rejection_reason
from tests.fakes import (
    GOOD_SUITE,
    TEST_LEAK_MARKERS,
    TRIVIAL_SUITE,
    UNCOVERED_SUITE,
    Scenario,
    make_dependencies,
    tool_call,
)


def invoke(scenario, review_only_passed=True, **graph_settings):
    config = Config()
    for name, value in graph_settings.items():
        setattr(config.graph, name, value)
    dependencies = make_dependencies(scenario, config, review_only_passed=review_only_passed)
    return build_pipeline_graph(dependencies).invoke({"task": "triple it"}, {"recursion_limit": 120})


def implement_calls(state):
    return [call for call in state["llm_calls"] if call["stage"] == "implement"]


# ---- the pipeline ------------------------------------------------------------------------

def test_full_pipeline_produces_typed_artifacts_end_to_end():
    state = invoke(Scenario())
    assert not state.get("failure_reason") and state["passed"] and state["stop_reason"] == "finished"
    assert state["requirements"].startswith("R1: ") and "def f(x: int) -> int" in state["design"]
    assert state["tests"].count("def test_") == 3 and "x * 3" in state["code"]
    assert "+    return x * 3" in state["diff"] and state["review"].endswith("No findings.")
    assert state["requirements_data"]["requirements"][0] == "f(x) returns x tripled"
    assert [call["stage"] for call in state["llm_calls"]][:4] == [
        "requirements", "design", "write_tests", "write_tests_gate"]


def test_implementer_never_sees_the_tests_or_their_source():
    state = invoke(Scenario())
    calls = implement_calls(state)
    assert len(calls) == 5
    serialized = json.dumps(calls)
    assert not any(marker in serialized for marker in TEST_LEAK_MARKERS)
    assert "test_triples: FAIL (AssertionError)" in serialized and "3/3 tests passed" in serialized
    assert "R1" not in json.dumps(calls[0]["messages"])  # not even the requirements


def test_each_stage_prompt_holds_only_its_declared_artifacts():
    scenario = Scenario()
    invoke(scenario)
    test_writer_prompt = scenario.prompts["TestSuite"][0]
    assert "R1: f(x) returns" in test_writer_prompt and "def f(x: int)" in test_writer_prompt
    assert "R1:" not in scenario.prompts["Design"][0].split("Requirements:")[0]
    review_prompt = scenario.prompts["Review"][0]
    assert "x * 3" in review_prompt and "3/3 tests passed" in review_prompt
    assert "R1:" not in review_prompt and "approach" not in review_prompt.lower()


def test_test_gate_retries_with_validator_feedback_then_accepts():
    scenario = Scenario(suites=[UNCOVERED_SUITE, GOOD_SUITE])
    state = invoke(scenario)
    assert state["tests_valid"] and state["test_attempts"] == 2 and state["passed"]
    retry_prompt = scenario.prompts["TestSuite"][1]
    assert "It was rejected for these problems" in retry_prompt
    assert "no test covers requirement(s) R3" in retry_prompt and "test_extra" in retry_prompt
    assert "rejected" not in scenario.prompts["TestSuite"][0]
    run = state_to_run(state, 1, 0, "system")
    test_writer_calls = [call for call in run["llm_calls"] if call["stage"] == "write_tests"]
    assert [call["attempt"] for call in test_writer_calls] == [2]  # only the accepted attempt
    assert not any(call["stage"] == "write_tests_gate" for call in run["llm_calls"])


def test_last_attempt_accepts_a_suite_missing_one_requirement_and_records_it():
    scenario = Scenario(suites=[UNCOVERED_SUITE])  # 2 of 3 requirements tested, every attempt
    state = invoke(scenario)
    assert state["test_attempts"] == 3 and state["tests_valid"] and not state.get("failure_reason")
    assert "code" in state


def test_test_gate_catches_trivial_tests_by_running_them_against_empty_code():
    scenario = Scenario(suites=[TRIVIAL_SUITE, GOOD_SUITE])
    state = invoke(scenario)
    assert "pass even with no implementation" in scenario.prompts["TestSuite"][1]
    assert state["tests_valid"] and state["passed"]


def test_test_gate_aborts_after_retries_so_no_implement_run_is_wasted():
    scenario = Scenario(suites=[TRIVIAL_SUITE])
    state = invoke(scenario)
    assert state["failure_reason"].startswith("write_tests: invalid test suite after 3 attempts")
    assert state["test_attempts"] == 3 and "code" not in state
    assert not any(kind == "tool_turn" for kind, _ in scenario.calls) and "Review" not in scenario.prompts
    assert rejection_reason(state_to_run(state, 1, 0, "system"), Config().filter) == "pipeline_failed"


def test_test_gate_can_proceed_with_invalid_tests_when_abort_is_off():
    state = invoke(Scenario(suites=[TRIVIAL_SUITE]), abort_on_invalid_tests=False)
    assert state["tests_valid"] is False and not state.get("failure_reason") and "code" in state
    run = state_to_run(state, 1, 0, "system")
    assert rejection_reason(run, Config().filter) == "tests_gate_failed"
    assert not [call for call in run["llm_calls"] if call["stage"] == "write_tests"]


def test_best_of_n_runs_candidates_in_parallel_and_selects_the_passing_one():
    state = invoke(Scenario(implementation="best_of_n"), candidates=3)
    assert state["candidate_count"] == 3 and state["candidates_passed"] == 1
    assert state["passed"] and "x * 3" in state["code"] and state["steps"] == 3
    calls = implement_calls(state)
    assert len(calls) == 3 and all(call["passed"] for call in calls)  # only the winner's trajectory
    assert state.get("candidates", []) == []  # the bulky candidate list was cleared


def test_an_unfinished_run_skips_review_unless_asked():
    scenario = Scenario(implementation="malformed")
    state = invoke(scenario)
    assert state["stop_reason"] == "stalled" and state["tool_errors"] == 3 and not state["passed"]
    assert "review" not in state and "Review" not in scenario.prompts
    assert "review" in invoke(Scenario(implementation="malformed"), review_only_passed=False)


def test_tool_errors_reach_the_model_verbatim_and_arguments_are_validated():
    def implementation(messages, previous_replies):
        script = [tool_call("read_file", path="nope.py"), tool_call("read_file"), tool_call("glob", pattern="*"),
                  AIMessage(content="done")]
        return script[min(len(previous_replies), 3)]

    state = invoke(Scenario(implementation=implementation))
    calls = implement_calls(state)
    tool_messages = [message["content"] for message in calls[-1]["messages"] if message["role"] == "tool"]
    assert tool_messages[0] == "No such file or directory: nope.py"  # no LangChain boilerplate
    assert tool_messages[1].startswith("Invalid arguments: path")
    assert state["tool_errors"] == 2 and state["tool_calls"] == 3
    assert calls[0]["completion_message"]["tool_calls"][0]["function"]["arguments"] == {"path": "nope.py"}
    tool_names = {schema["function"]["name"] for schema in calls[0]["tool_schemas"]}
    assert tool_names >= {"read_file", "str_replace", "run_tests"}


def test_hidden_tests_never_exist_in_the_workspace(tmp_path):
    seed = tmp_path / "seed"
    (seed / "tests").mkdir(parents=True)
    (seed / "tests" / "test_visible.py").write_text("def test_leak():\n    assert 1 == 1\n")

    def implementation(messages, previous_replies):
        script = [tool_call("read_file", path="tests/test_visible.py"), tool_call("grep", pattern="test_"),
                  AIMessage(content="done")]
        return script[min(len(previous_replies), 2)]

    config = Config()
    config.agent.workspace_seed = str(seed)
    config.tools.hidden_globs = ["tests/*"]
    state = build_pipeline_graph(make_dependencies(Scenario(implementation=implementation), config)).invoke(
        {"task": "t"}, {"recursion_limit": 120})
    tool_messages = [message["content"] for message in implement_calls(state)[-1]["messages"]
                     if message["role"] == "tool"]
    assert tool_messages[0] == "No such file or directory: tests/test_visible.py"
    assert tool_messages[1].startswith("No matches")


def test_a_structured_parse_failure_fails_the_run_cleanly():
    class GarbageDesign(Scenario):
        def __call__(self, messages, tools, schema):
            if schema is artifacts.Design:
                return AIMessage(content="not json")
            return super().__call__(messages, tools, schema)

    state = invoke(GarbageDesign())
    assert state["failure_reason"].startswith("design: unparseable output") and "tests" not in state


# ---- message utilities -------------------------------------------------------------------

def conversation(turn_count: int, output_chars: int = 400):
    messages = [SystemMessage("SYSTEM"), HumanMessage("TASK")]
    for turn in range(turn_count):
        messages.append(AIMessage(content="", tool_calls=[
            {"name": "grep", "args": {"pattern": str(turn)}, "id": f"call{turn}"}]))
        messages.append(ToolMessage(content="x" * output_chars, tool_call_id=f"call{turn}", name="grep"))
    return messages


def approximate_tokens(text):
    return (len(text) + 3) // 4


def test_compaction_never_orphans_tool_calls_and_stays_in_budget():
    messages = conversation(8)
    tracer = Tracer()
    compacted = compact_messages(messages, 300, approximate_tokens, tracer)
    assert sum(approximate_tokens(str(message.content)) for message in compacted) <= 330
    called_ids = [call["id"] for message in compacted if isinstance(message, AIMessage) for call in message.tool_calls]
    answered_ids = [message.tool_call_id for message in compacted if isinstance(message, ToolMessage)]
    assert called_ids == answered_ids  # every call has its result and vice versa
    assert compacted[-1].tool_call_id == "call7" and compacted[-1].content == "x" * 400  # newest intact
    assert any(message.content == ELIDED_TOOL_OUTPUT for message in compacted) or "omitted" in str(compacted[1].content)
    assert str(messages[3].content) == "x" * 400  # the originals are untouched
    assert {event["section"] for event in tracer.events} <= {
        "conversation:tool_output", "conversation:dropped_steps", "conversation:last_message"}
    assert compact_messages(conversation(1, 10), 10_000, approximate_tokens) == conversation(1, 10)


def test_chat_dicts_keep_tool_call_arguments_as_dicts():
    chat_dict = to_chat_dict(AIMessage(content="", tool_calls=[{"name": "g", "args": {"a": 1}, "id": "i"}]))
    assert chat_dict == {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "g", "arguments": {"a": 1}}}]}
    assert to_chat_dict(ToolMessage(content="r", tool_call_id="i", name="g"))["role"] == "tool"


# ---- tracing -----------------------------------------------------------------------------

def test_callback_tags_events_with_the_stage_and_detects_truncation():
    tracer = Tracer()
    dependencies = make_dependencies(Scenario(), tracer=tracer)
    build_pipeline_graph(dependencies).invoke(
        {"task": "t"}, run_config(dependencies.config, 1, 0, "round0", [TracerCallback(tracer)]))
    summary = summarize_events(tracer.events)
    assert set(summary["stages"]) == {"requirements", "design", "write_tests", "implement", "review"}
    assert summary["stages"]["implement"]["calls"] == 5 and summary["stages"]["requirements"]["calls"] == 1
    assert all(stage["truncated"] == 0 for stage in summary["stages"].values())

    truncating_tracer = Tracer()
    dependencies = make_dependencies(Scenario(), tracer=truncating_tracer, finish_reason="length")
    state = build_pipeline_graph(dependencies).invoke(
        {"task": "t"}, run_config(dependencies.config, 1, 0, "round0", [TracerCallback(truncating_tracer)]))
    assert not state.get("failure_reason")  # complete JSON still parses with finish_reason=length
    assert summarize_events(truncating_tracer.events)["stages"]["requirements"]["truncated"] == 1


def test_run_config_carries_langsmith_metadata_and_a_sufficient_recursion_limit():
    config = Config()
    config.langsmith.tags = ["pilot"]
    runnable_config = run_config(config, 42, 3, "round1", [])
    assert runnable_config["run_name"] == "sdlc:42:3"
    assert runnable_config["tags"] == ["harness", "round1", "pilot"]
    assert runnable_config["metadata"] == {"task_id": 42, "sample": 3, "round": "round1"}
    assert runnable_config["configurable"]["thread_id"] == "round1:42:3"
    assert runnable_config["recursion_limit"] >= 3 * config.agent.max_agent_steps + 30
