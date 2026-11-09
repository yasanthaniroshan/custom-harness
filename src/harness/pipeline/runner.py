"""Concurrent, resumable batch runner for the SDLC graph.

Runs are independent graph invocations executed concurrently (`batch_as_completed`); vLLM's
continuous batching turns the concurrent requests into GPU batches. Each finished run is
appended to a JSONL file immediately, so a crash loses at most the runs in flight. With a
checkpointer, runs interrupted mid-graph resume from their last completed stage.

Checkpoints belong to one output file: their key includes a scope id stored next to it
(`runs.jsonl.checkpoint-scope`). Deleting the output directory therefore starts every run
fresh, and a different output directory (another round, a new adapter) never replays old runs.

A run record (one JSONL line) looks like:
    {"task_id", "sample", "system_prompt",
     "artifacts": {"task", "requirements", "design", "tests", "code", "diff", "test_output", "review"},
     "llm_calls": [{"stage", "attempt", "messages", "completion", ...}, ...],
     "passed", "test_count", "steps", "tool_calls", "tool_errors", "stop_reason",
     "tests_valid", "failure_reason", "candidate_count", "candidates_passed"}
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from harness.config import Config
from harness.monitoring import NULL_TRACER, Tracer
from harness.pipeline.dependencies import PipelineDependencies
from harness.pipeline.graph import build_pipeline_graph
from harness.pipeline.implementer import cleanup_sessions
from harness.pipeline.tracing import TracerCallback, log_feedback, new_run_id, run_config
from harness.prompt_context import TokenCounter, approximate_token_count
from harness.training_data.trajectory_filter import rejection_reason

ARTIFACT_KEYS = ("task", "requirements", "design", "tests", "code", "diff", "test_output", "review")
PROGRESS_EVERY_RUNS = 10
CHECKPOINT_SCOPE_SUFFIX = ".checkpoint-scope"


def checkpoint_scope_for(output_path: Path) -> str:
    """The checkpoint scope of `output_path`: created once, kept while its directory exists."""
    scope_file = output_path.with_name(output_path.name + CHECKPOINT_SCOPE_SUFFIX)
    if not scope_file.exists():
        scope_file.write_text(uuid.uuid4().hex + "\n")
    return scope_file.read_text().strip()


def state_to_run(state: dict, task_id: Any, sample: int, system_prompt: str) -> dict:
    """Final graph state -> the run record the data filters and evaluation read."""
    tests_valid = state.get("tests_valid")
    accepted_test_attempt = state.get("test_attempts")
    kept_calls = []
    for call in state.get("llm_calls", []):
        if call["stage"] == "write_tests_gate":
            continue
        if call["stage"] == "write_tests" and not (tests_valid and call["attempt"] == accepted_test_attempt):
            continue  # only the accepted suite is training data, never a rejected attempt
        kept_calls.append(call)
    return {
        "task_id": task_id,
        "sample": sample,
        "system_prompt": system_prompt,
        "artifacts": {key: state.get(key, "") for key in ARTIFACT_KEYS},
        "llm_calls": kept_calls,
        "passed": bool(state.get("passed")),
        "test_count": state.get("test_count", 0),
        "steps": state.get("steps", 0),
        "tool_calls": state.get("tool_calls", 0),
        "tool_errors": state.get("tool_errors", 0),
        "stop_reason": state.get("stop_reason", ""),
        "tests_valid": tests_valid,
        "failure_reason": state.get("failure_reason", ""),
        "candidate_count": state.get("candidate_count", 0),
        "candidates_passed": state.get("candidates_passed", 0),
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _completed_runs(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    return {(run["task_id"], run["sample"]) for run in read_jsonl(path)}


@contextmanager
def open_pipeline_graph(dependencies: PipelineDependencies) -> Iterator[tuple[Any, bool]]:
    """(compiled graph, whether it is checkpointed). Checkpoints need graph.checkpoint_db."""
    checkpoint_db = dependencies.config.graph.checkpoint_db
    if not checkpoint_db:
        yield build_pipeline_graph(dependencies), False
        return
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(checkpoint_db) as checkpointer:
        yield build_pipeline_graph(dependencies, checkpointer), True


def run_tasks(
    graph: Any,
    config: Config,
    tasks: list[dict],
    samples_per_task: int,
    output_path: Path,
    round_name: str,
    tracer: Tracer = NULL_TRACER,
    count_tokens: TokenCounter = approximate_token_count,
    checkpointed: bool = False,
    feedback_client: Any = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run every (task, sample) not yet in `output_path`, appending one run record per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = _completed_runs(output_path)
    pending = [(task, sample) for task in tasks for sample in range(samples_per_task)
               if (task["task_id"], sample) not in already_done]
    log(f"{len(tasks) * samples_per_task} runs, {len(already_done)} already done, {len(pending)} to run")
    callbacks = [TracerCallback(tracer, count_tokens)]
    run_ids = [new_run_id() for _ in pending]
    checkpoint_scope = checkpoint_scope_for(output_path)
    run_configs = [run_config(config, task["task_id"], sample, round_name, callbacks, run_id, checkpoint_scope)
                   for (task, sample), run_id in zip(pending, run_ids, strict=True)]
    # run_config's max_concurrency is also the size of batch_as_completed's worker pool
    graph_inputs: list[Any] = [{"task": task["task"]} for task, _ in pending]
    stats = {"completed": 0, "errors": 0, "resumed": 0, "from_checkpoint": 0}
    finished_in_checkpoint: list[tuple[int, dict]] = []
    to_run: list[int] = []
    for position, runnable_config in enumerate(run_configs):
        snapshot = graph.get_state(runnable_config) if checkpointed else None
        if snapshot is not None and snapshot.values and not snapshot.next:
            finished_in_checkpoint.append((position, snapshot.values))  # done but never written out
            stats["from_checkpoint"] += 1
            continue
        if snapshot is not None and snapshot.next:
            graph_inputs[position] = None  # None = resume from the last completed stage
            stats["resumed"] += 1
        to_run.append(position)

    def record(position: int, final_state: dict) -> None:
        task, sample = pending[position]
        run = state_to_run(final_state, task["task_id"], sample, config.agent.system_prompt)
        with output_path.open("a") as output_file:
            output_file.write(json.dumps(run) + "\n")
        tracer.emit("run_end", task_id=task["task_id"], sample=sample, passed=run["passed"],
                    test_count=run["test_count"], steps=run["steps"], tool_calls=run["tool_calls"],
                    tool_errors=run["tool_errors"],
                    stop_reason=run["stop_reason"] or ("failed" if run["failure_reason"] else ""))
        if feedback_client is not None:
            trusted = rejection_reason(run, config.filter, config.agent.test_timeout_seconds) is None
            log_feedback(feedback_client, run_ids[position], run, trusted)
        stats["completed"] += 1

    try:
        for position, final_state in finished_in_checkpoint:
            record(position, final_state)
        results = graph.batch_as_completed(
            [graph_inputs[position] for position in to_run],
            [run_configs[position] for position in to_run],
            return_exceptions=True)
        for index_in_batch, result in results:
            position = to_run[index_in_batch]
            if isinstance(result, Exception):
                stats["errors"] += 1
                task, sample = pending[position]
                log(f"run {task['task_id']}#{sample} failed: {type(result).__name__}: {result}")
                continue
            record(position, result)
            if stats["completed"] % PROGRESS_EVERY_RUNS == 0:
                log(f"  {stats['completed']}/{len(pending)} runs done")
    finally:
        cleanup_sessions()  # temporary workspaces of runs that crashed mid-implement
    log(f"done: {stats}")
    return stats
