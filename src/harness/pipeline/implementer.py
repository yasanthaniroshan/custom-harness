r"""The tool-using implement stage, as a LangGraph subgraph run once per candidate.

    setup -> model --tool calls--> tools -> after_tools --continue--> model
               |--malformed call--> malformed_call ---/     \--stop--> finish
               \--no tool call--> finish

Each candidate gets its own temporary workspace. The hidden tests are only ever passed to the
`run_tests` tool and to the final verification, in memory; they are never in the workspace.
"""
from __future__ import annotations

import operator
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from harness.pipeline.dependencies import PipelineDependencies
from harness.pipeline.messages import compact_messages, message_text, to_chat_dict
from harness.pipeline.stages import IMPLEMENT
from harness.pipeline.tool_adapters import to_langchain_tools, tool_error_message
from harness.sandbox import run_workspace_against_tests
from harness.tools import Tool, Workspace, build_tool_registry

CLEAR_CANDIDATES = "__clear__"
MALFORMED_CALL_FEEDBACK = (
    "ERROR: your tool call could not be parsed. Reply with valid JSON: "
    '<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>')


def merge_candidates(existing: list | None, update: list | str) -> list:
    """State reducer: append candidate results; the CLEAR_CANDIDATES sentinel empties the list."""
    if update == CLEAR_CANDIDATES:
        return []
    return (existing or []) + list(update)


def make_run_tests_tool(workspace: Workspace, hidden_tests: str, timeout_seconds: int,
                        max_report_chars: int) -> Tool:
    def run_tests() -> str:
        return run_workspace_against_tests(workspace.root, hidden_tests, timeout_seconds).summary(max_report_chars)

    return Tool("run_tests", "Run the hidden tests against your workspace. Returns pass/fail per "
                "test (not the tests).", [], run_tests)


def collect_code(workspace: Workspace) -> str:
    """The implementation as text for later stages: the one changed .py file, or all of them
    with a header line each."""
    changed = [path for path in workspace.changed_files() if path.endswith(".py") and workspace.exists(path)]
    if len(changed) == 1:
        return workspace.read(changed[0])
    return "\n\n".join(f"# file: {path}\n{workspace.read(path)}" for path in changed)


@dataclass
class ImplementerSession:
    """Per-candidate objects that cannot live in graph state (it must stay serializable)."""

    workspace: Workspace
    temporary_directory: tempfile.TemporaryDirectory
    hidden_tests: str
    model_with_tools: Any
    tool_node: ToolNode
    tool_schemas: list[dict]


_SESSIONS: dict[str, ImplementerSession] = {}
_SESSIONS_LOCK = threading.Lock()


def open_session_count() -> int:
    with _SESSIONS_LOCK:
        return len(_SESSIONS)


def cleanup_sessions() -> None:
    """Delete the workspaces of candidates that crashed before reaching `finish`."""
    with _SESSIONS_LOCK:
        for session in _SESSIONS.values():
            session.temporary_directory.cleanup()
        _SESSIONS.clear()


class CandidateState(TypedDict, total=False):
    # Inputs. Every key here is prefixed so it never collides with a parent-graph key: parallel
    # candidates merging the same key in one step would be an InvalidUpdateError.
    candidate_design: str
    candidate_tests: str
    candidate_index: int
    # loop state
    messages: Annotated[list[AnyMessage], add_messages]
    candidate_turns: Annotated[list[dict], operator.add]
    candidate_session_id: str
    candidate_steps: int
    candidate_error_streak: int
    candidate_tool_calls: int
    candidate_tool_errors: int
    candidate_stop_reason: str
    # output, merged into the parent graph through its reducer
    candidates: Annotated[list[dict], merge_candidates]


def build_implementer_graph(dependencies: PipelineDependencies):
    agent_config, tools_config = dependencies.config.agent, dependencies.config.tools
    instruction = IMPLEMENT.instruction.format(solution_file=agent_config.solution_file)

    def setup(state: CandidateState) -> dict:
        temporary_directory = tempfile.TemporaryDirectory(prefix="harness_workspace_")
        root = Path(temporary_directory.name)
        if agent_config.workspace_seed:
            shutil.copytree(agent_config.workspace_seed, root, dirs_exist_ok=True)
        workspace = Workspace(root, tools_config.hidden_globs)
        run_tests_tool = make_run_tests_tool(workspace, state["candidate_tests"],
                                             agent_config.test_timeout_seconds,
                                             agent_config.max_test_report_chars)
        registry = build_tool_registry(workspace, tools_config, dependencies.tracer,
                                       dependencies.prompt_builder.count_tokens,
                                       extra_tools=[run_tests_tool])
        langchain_tools = to_langchain_tools(registry)
        session_id = uuid.uuid4().hex
        session = ImplementerSession(
            workspace=workspace,
            temporary_directory=temporary_directory,
            hidden_tests=state["candidate_tests"],
            model_with_tools=dependencies.chat_models.get(dependencies.code_temperature).bind_tools(langchain_tools),
            tool_node=ToolNode(langchain_tools, handle_tool_errors=tool_error_message),
            tool_schemas=[convert_to_openai_tool(tool) for tool in langchain_tools],
        )
        with _SESSIONS_LOCK:
            _SESSIONS[session_id] = session
        user_prompt = dependencies.prompt_builder.build_prompt(
            IMPLEMENT, {"design": state["candidate_design"]}, instruction)
        return {
            "candidate_session_id": session_id, "candidate_steps": 0, "candidate_error_streak": 0,
            "candidate_tool_calls": 0, "candidate_tool_errors": 0, "candidate_stop_reason": "",
            "messages": [SystemMessage(agent_config.system_prompt), HumanMessage(user_prompt)],
        }

    def call_model(state: CandidateState, config: RunnableConfig) -> dict:
        # LangGraph passes the run's RunnableConfig to a parameter named `config`
        session = _SESSIONS[state["candidate_session_id"]]
        visible_messages = compact_messages(
            state["messages"], dependencies.prompt_builder.config.max_prompt_tokens,
            dependencies.prompt_builder.count_tokens, dependencies.tracer)
        reply = session.model_with_tools.invoke(visible_messages, config)
        turn = {"messages": [to_chat_dict(message) for message in visible_messages],
                "completion": to_chat_dict(reply)}
        return {"messages": [reply], "candidate_turns": [turn],
                "candidate_steps": state["candidate_steps"] + 1}

    def route_after_model(state: CandidateState) -> str:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        if "<tool_call>" in message_text(last_message.content):  # the server could not parse it
            return "malformed_call"
        return "finish"

    def run_tools(state: CandidateState, config: RunnableConfig) -> dict:
        session = _SESSIONS[state["candidate_session_id"]]
        return session.tool_node.invoke({"messages": state["messages"]}, config)

    def stop_reason(state: CandidateState, error_streak: int) -> str:
        if error_streak >= agent_config.max_error_streak:
            return "stalled"
        if state["candidate_steps"] >= agent_config.max_agent_steps:
            return "max_steps"
        return ""

    def after_tools(state: CandidateState) -> dict:
        latest_results: list[ToolMessage] = []
        for message in reversed(state["messages"]):  # the tool results of the latest turn
            if not isinstance(message, ToolMessage):
                break
            latest_results.append(message)
        error_count = sum(message.status == "error" for message in latest_results)
        all_failed = bool(latest_results) and error_count == len(latest_results)
        error_streak = state["candidate_error_streak"] + 1 if all_failed else 0
        return {"candidate_error_streak": error_streak,
                "candidate_tool_calls": state["candidate_tool_calls"] + len(latest_results),
                "candidate_tool_errors": state["candidate_tool_errors"] + error_count,
                "candidate_stop_reason": stop_reason(state, error_streak)}

    def malformed_call(state: CandidateState) -> dict:
        error_streak = state["candidate_error_streak"] + 1
        return {"messages": [HumanMessage(MALFORMED_CALL_FEEDBACK)],
                "candidate_error_streak": error_streak,
                "candidate_tool_calls": state["candidate_tool_calls"] + 1,
                "candidate_tool_errors": state["candidate_tool_errors"] + 1,
                "candidate_stop_reason": stop_reason(state, error_streak)}

    def route_continue(state: CandidateState) -> str:
        return "finish" if state["candidate_stop_reason"] else "model"

    def finish(state: CandidateState) -> dict:
        with _SESSIONS_LOCK:
            session = _SESSIONS.pop(state["candidate_session_id"])
        try:  # authoritative verification, independent of whatever the model claimed
            report = run_workspace_against_tests(session.workspace.root, session.hidden_tests,
                                                 agent_config.test_timeout_seconds)
            code, diff = collect_code(session.workspace), session.workspace.diff()
        finally:
            session.temporary_directory.cleanup()
        return {"candidates": [{
            "index": state["candidate_index"],
            "passed": report.passed,
            "test_count": len(report.cases),
            "tests_passed_count": report.passed_count,
            "steps": state["candidate_steps"],
            "tool_calls": state["candidate_tool_calls"],
            "tool_errors": state["candidate_tool_errors"],
            "stop_reason": state["candidate_stop_reason"] or "finished",
            "code": code,
            "diff": diff,
            "test_output": report.summary(agent_config.max_test_report_chars),
            "turns": state["candidate_turns"],
            "tool_schemas": session.tool_schemas,
        }]}

    graph = StateGraph(CandidateState)
    graph.add_node("setup", setup)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("after_tools", after_tools)
    graph.add_node("malformed_call", malformed_call)
    graph.add_node("finish", finish)
    graph.add_edge(START, "setup")
    graph.add_edge("setup", "model")
    graph.add_conditional_edges("model", route_after_model,
                                {"tools": "tools", "malformed_call": "malformed_call", "finish": "finish"})
    graph.add_edge("tools", "after_tools")
    graph.add_conditional_edges("after_tools", route_continue, {"model": "model", "finish": "finish"})
    graph.add_conditional_edges("malformed_call", route_continue, {"model": "model", "finish": "finish"})
    graph.add_edge("finish", END)
    # No checkpoints inside a candidate: its temporary workspace cannot survive a process
    # restart, so the parent graph treats a whole candidate as one atomic step.
    return graph.compile(checkpointer=False)
