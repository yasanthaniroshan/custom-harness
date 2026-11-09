"""Observability for the graph: a LangChain callback that writes our events.jsonl, the
per-run RunnableConfig (names, tags, metadata for LangSmith), and LangSmith feedback scores.

LangSmith traces every LangChain/LangGraph call automatically once enabled (see
harness.environment.setup_langsmith); this module adds what LangSmith does not give us.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from harness.config import Config
from harness.monitoring import Tracer
from harness.pipeline.messages import message_text
from harness.prompt_context import TokenCounter, approximate_token_count

STAGE_BY_GRAPH_NODE = {"model": "implement"}  # the implementer's model node belongs to "implement"
SUPERSTEPS_PER_AGENT_STEP = 3  # model, tools, after_tools
RECURSION_LIMIT_HEADROOM = 40  # the prose stages, the test gate retries, select, review


class TracerCallback(BaseCallbackHandler):
    """Writes one llm_call event (tokens, truncation, latency) per model request."""

    raise_error = False  # monitoring must never break a run

    def __init__(self, tracer: Tracer, count_tokens: TokenCounter = approximate_token_count):
        self.tracer = tracer
        self.count_tokens = count_tokens
        self._started: dict[UUID, tuple[str | None, int, float]] = {}  # run id -> (stage, tokens, start)
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        graph_node = (metadata or {}).get("langgraph_node")
        estimated_prompt_tokens = sum(self.count_tokens(message_text(message.content))
                                      for batch in messages for message in batch)
        with self._lock:
            self._started[run_id] = (STAGE_BY_GRAPH_NODE.get(graph_node, graph_node),
                                     estimated_prompt_tokens, time.perf_counter())

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            stage, estimated_prompt_tokens, started = self._started.pop(
                run_id, (None, 0, time.perf_counter()))
        generation = response.generations[0][0]
        reply = getattr(generation, "message", None)
        usage = getattr(reply, "usage_metadata", None) or {}
        reply_text = message_text(getattr(reply, "content", generation.text))
        self.tracer.emit(
            "llm_call", stage_name=stage,
            prompt_tokens=usage.get("input_tokens") or estimated_prompt_tokens,
            completion_tokens=usage.get("output_tokens") or self.count_tokens(reply_text),
            truncated=(generation.generation_info or {}).get("finish_reason") == "length",
            seconds=round(time.perf_counter() - started, 3),
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            stage, _, _ = self._started.pop(run_id, (None, 0, 0.0))
        self.tracer.emit("llm_error", stage_name=stage, error=type(error).__name__)


def thread_id_for(checkpoint_scope: str, task_id: Any, sample: int) -> str:
    """Checkpoint key of one run: the same (scope, task, sample) resumes the same thread."""
    return f"{checkpoint_scope}:{task_id}:{sample}"


def run_config(config: Config, task_id: Any, sample: int, round_name: str, callbacks: list,
               run_id: UUID | None = None, checkpoint_scope: str | None = None) -> dict:
    """RunnableConfig for one graph run; name, tags and metadata show up in LangSmith.

    checkpoint_scope keys the run's checkpoint thread (default: round_name). The batch runner
    passes one tied to the output file, so checkpoints never leak into another output.
    """
    runnable_config: dict = {
        "run_name": f"sdlc:{task_id}:{sample}",
        "tags": ["harness", round_name, *config.langsmith.tags],
        "metadata": {"task_id": task_id, "sample": sample, "round": round_name},
        "callbacks": callbacks,
        "configurable": {"thread_id": thread_id_for(checkpoint_scope or round_name, task_id, sample)},
        "max_concurrency": config.serve.concurrency,
        # LangGraph's default limit of 25 supersteps is too low for the tool loop
        "recursion_limit": SUPERSTEPS_PER_AGENT_STEP * config.agent.max_agent_steps + RECURSION_LIMIT_HEADROOM,
    }
    if run_id is not None:
        runnable_config["run_id"] = run_id
    return runnable_config


def new_run_id() -> UUID:
    return uuid.uuid4()


def log_feedback(client: Any, run_id: UUID, run: dict, trusted: bool | None = None) -> None:
    """Attach scores to a traced run so LangSmith can filter and chart outcomes. Never raises."""
    scores = {
        "tests_passed": float(bool(run.get("passed"))),
        "tests_valid": float(run.get("tests_valid", True) is not False),
        "agent_steps": float(run.get("steps", 0)),
        "tool_errors": float(run.get("tool_errors", 0)),
    }
    if trusted is not None:
        scores["trusted"] = float(trusted)
    for key, score in scores.items():
        try:
            client.create_feedback(run_id=run_id, key=key, score=score)
        except Exception:  # noqa: BLE001, S112 - telemetry must never break generation
            continue
