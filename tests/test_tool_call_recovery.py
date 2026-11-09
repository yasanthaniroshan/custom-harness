from langchain_core.messages import AIMessage

from harness.config import Config
from harness.pipeline.graph import build_pipeline_graph
from harness.pipeline.tool_call_recovery import recover_tool_calls, with_recovered_tool_calls
from tests.fakes import BAD_CODE, GOOD_CODE, Scenario, make_dependencies

TOOL_NAMES = {"write_file", "run_tests", "str_replace"}


def test_json_in_a_code_fence_is_recovered_as_a_tool_call():
    # what Qwen2.5-Coder actually wrote on Kaggle: the call inside a ```json fence
    text = '```json\n{\n  "name": "write_file",\n  "arguments": {"path": "solution.py", "content": "x = 1\\n"}\n}\n```'
    calls = recover_tool_calls(text, TOOL_NAMES)
    assert [(call["name"], call["args"]) for call in calls] == [
        ("write_file", {"path": "solution.py", "content": "x = 1\n"})]
    assert calls[0]["id"].startswith("recovered-")


def test_other_formats_bare_json_lists_tags_and_string_arguments():
    assert recover_tool_calls('{"name": "run_tests", "arguments": {}}', TOOL_NAMES)[0]["name"] == "run_tests"
    two = '[{"name": "run_tests", "arguments": {}}, {"name": "run_tests", "parameters": {}}]'
    assert len(recover_tool_calls(two, TOOL_NAMES)) == 2
    tagged = '<tool_call>{"name": "write_file", "arguments": "{\\"path\\": \\"a.py\\", \\"content\\": \\"\\"}"}'
    assert recover_tool_calls(tagged, TOOL_NAMES)[0]["args"] == {"path": "a.py", "content": ""}


def test_ordinary_text_and_unknown_tools_are_never_mistaken_for_calls():
    assert recover_tool_calls("```python\ndef f(x):\n    return x\n```", TOOL_NAMES) == []
    assert recover_tool_calls('{"name": "rm_rf", "arguments": {}}', TOOL_NAMES) == []
    assert recover_tool_calls('```json\n{"name": "Alice", "age": 3}\n```', TOOL_NAMES) == []
    assert recover_tool_calls("not json {", TOOL_NAMES) == []


def test_replies_with_real_tool_calls_are_left_alone():
    reply = AIMessage(content="", tool_calls=[{"name": "run_tests", "args": {}, "id": "real"}])
    assert with_recovered_tool_calls(reply, TOOL_NAMES) == (reply, 0)
    prose = AIMessage(content="All tests pass.")
    assert with_recovered_tool_calls(prose, TOOL_NAMES) == (prose, 0)


def fenced(name, **arguments):
    import json

    return AIMessage(content="```json\n" + json.dumps({"name": name, "arguments": arguments}) + "\n```")


def run_pipeline(implementation, tracer=None):
    dependencies = make_dependencies(Scenario(implementation=implementation), Config(), tracer=tracer)
    return build_pipeline_graph(dependencies).invoke({"task": "triple it"}, {"recursion_limit": 120})


def test_the_agent_completes_a_task_when_every_call_is_written_in_a_json_fence():
    from harness.monitoring import Tracer

    def implementation(messages, previous_replies):
        script = [fenced("write_file", path="solution.py", content=GOOD_CODE), fenced("run_tests"),
                  AIMessage(content="Done.")]
        return script[min(len(previous_replies), 2)]

    tracer = Tracer()
    state = run_pipeline(implementation, tracer)
    assert state["passed"] and state["tool_calls"] == 2 and state["stop_reason"] == "finished"
    first_turn = next(call for call in state["llm_calls"] if call["stage"] == "implement")
    # the training row records a proper tool call, not the fenced text
    assert first_turn["completion_message"]["tool_calls"][0]["function"]["name"] == "write_file"
    assert first_turn["completion_message"]["content"] == ""
    assert sum(event["count"] for event in tracer.events if event["kind"] == "tool_call_recovered") == 2


def test_code_pasted_as_text_gets_feedback_instead_of_ending_the_run():
    def implementation(messages, previous_replies):
        script = [AIMessage(content=f"```python\n{BAD_CODE}```"),
                  fenced("write_file", path="solution.py", content=GOOD_CODE),
                  AIMessage(content="Done.")]
        return script[min(len(previous_replies), 2)]

    state = run_pipeline(implementation)
    assert state["passed"] and state["steps"] == 3
    turns = [call for call in state["llm_calls"] if call["stage"] == "implement"]
    assert "nothing has been written to the workspace" in turns[1]["messages"][-1]["content"]


def test_a_model_that_never_writes_code_stalls_instead_of_looping_forever():
    def implementation(messages, previous_replies):
        return AIMessage(content="Here is the code:\n```python\ndef f(x): return 3 * x\n```")

    state = run_pipeline(implementation)
    assert state["stop_reason"] == "stalled" and not state["passed"] and state["steps"] == 3
