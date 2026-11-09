import json
import os
import shutil

import pytest

from harness.config import Config
from harness.environment import setup_langsmith
from harness.monitoring import Tracer
from harness.pipeline import artifacts
from harness.pipeline.chat_models import ChatModelFactory, served_model_name
from harness.pipeline.implementer import open_session_count
from harness.pipeline.runner import open_pipeline_graph, read_jsonl, run_tasks, state_to_run
from harness.pipeline.tracing import log_feedback
from harness.serving import build_server_command
from harness.training_data.trajectory_filter import filter_runs
from tests.fakes import TEST_LEAK_MARKERS, Scenario, make_dependencies

TASKS = [{"task_id": number, "task": f"task {number}"} for number in range(3)]


def run(scenario, output_dir, samples_per_task=2, checkpoint_db=None, feedback_client=None,
        tracer=None, tasks=TASKS):
    config = Config()
    config.graph.checkpoint_db = checkpoint_db
    dependencies = make_dependencies(scenario, config, tracer=tracer)
    runs_path = output_dir / "runs.jsonl"
    with open_pipeline_graph(dependencies) as (graph, checkpointed):
        stats = run_tasks(graph, config, tasks, samples_per_task, runs_path, "round0", dependencies.tracer,
                          dependencies.prompt_builder.count_tokens, checkpointed, feedback_client,
                          log=lambda *_: None)
    return stats, read_jsonl(runs_path) if runs_path.exists() else []


class RecordingFeedbackClient:
    def __init__(self):
        self.sent = []

    def create_feedback(self, **feedback):
        self.sent.append(feedback)


def test_runs_execute_concurrently_write_jsonl_and_resume(tmp_path):
    tracer = Tracer()
    scenario = Scenario()
    stats, runs = run(scenario, tmp_path, tracer=tracer)
    assert stats["completed"] == 6 and stats["errors"] == 0 and len(runs) == 6
    assert {(run_record["task_id"], run_record["sample"]) for run_record in runs} == {
        (task, sample) for task in range(3) for sample in range(2)}
    assert all(run_record["passed"] and run_record["tests_valid"] and run_record["artifacts"]["code"]
               for run_record in runs)
    run_ends = [event for event in tracer.events if event["kind"] == "run_end"]
    assert len(run_ends) == 6 and all(event["steps"] == 5 for event in run_ends)
    call_count = len(scenario.calls)
    stats, runs = run(scenario, tmp_path)  # the same output file: nothing left to do
    assert stats["completed"] == 0 and len(runs) == 6 and len(scenario.calls) == call_count


def test_runs_become_sft_rows_through_the_trust_filter(tmp_path):
    _, runs = run(Scenario(), tmp_path, samples_per_task=2)
    rows, stats = filter_runs(runs, Config().filter)
    assert stats["kept"] == 3  # identical code per task is deduplicated
    rows_by_stage = {stage: sum(row["stage"] == stage for row in rows) for stage in {row["stage"] for row in rows}}
    assert rows_by_stage == {"requirements": 3, "design": 3, "write_tests": 3, "implement": 15, "review": 3}
    implement_rows = [row for row in rows if row["stage"] == "implement"]
    assert all(row["prompt"][-1]["role"] in ("user", "tool") and row["completion"][0]["role"] == "assistant"
               for row in implement_rows)
    assert any("tool_calls" in row["completion"][0] for row in implement_rows)
    assert all("tools" in row for row in implement_rows)
    assert not any(marker in json.dumps(implement_rows) for marker in TEST_LEAK_MARKERS)
    requirements_row = next(row for row in rows if row["stage"] == "requirements")
    assert json.loads(requirements_row["completion"][0]["content"])["requirements"][0] == "f(x) returns x tripled"


def test_failed_runs_are_counted_not_written_and_workspaces_are_cleaned(tmp_path):
    class ServerDownForTaskOne(Scenario):
        def __call__(self, messages, tools, schema):
            if schema is artifacts.Requirements and "task 1" in str(messages[-1].content):
                raise ConnectionError("server down")
            return super().__call__(messages, tools, schema)

    stats, runs = run(ServerDownForTaskOne(), tmp_path, samples_per_task=1)
    assert stats["errors"] == 1 and stats["completed"] == 2
    assert {run_record["task_id"] for run_record in runs} == {0, 2}
    assert open_session_count() == 0


def test_langsmith_feedback_is_logged_per_run_with_a_trusted_score(tmp_path):
    client = RecordingFeedbackClient()
    run(Scenario(), tmp_path, samples_per_task=1, feedback_client=client)
    assert len(client.sent) == 3 * 5  # five scores for each of the three runs
    assert len({feedback["run_id"] for feedback in client.sent}) == 3
    assert all(feedback["score"] == 1.0 for feedback in client.sent
               if feedback["key"] in ("tests_passed", "trusted", "tests_valid"))


def test_feedback_failures_never_propagate():
    class OfflineClient:
        def create_feedback(self, **feedback):
            raise ConnectionError("offline")

    log_feedback(OfflineClient(), "run-id", {"passed": True})  # must not raise
    client = RecordingFeedbackClient()
    log_feedback(client, "run-id", {"passed": True, "tests_valid": True, "steps": 5, "tool_errors": 1}, trusted=True)
    assert {feedback["key"]: feedback["score"] for feedback in client.sent} == {
        "tests_passed": 1.0, "tests_valid": 1.0, "agent_steps": 5.0, "tool_errors": 1.0, "trusted": 1.0}


def test_checkpoints_resume_only_the_unfinished_stage_after_a_crash(tmp_path):
    checkpoint_db = str(tmp_path / "checkpoints.db")
    scenario = Scenario(fail_review_once=True)
    stats, runs = run(scenario, tmp_path, samples_per_task=1, checkpoint_db=checkpoint_db, tasks=TASKS[:1])
    assert stats["errors"] == 1 and runs == []  # crashed in the final stage
    calls_before = list(scenario.calls)
    assert [kind for kind, _ in calls_before].count("Requirements") == 1
    stats, runs = run(scenario, tmp_path, samples_per_task=1, checkpoint_db=checkpoint_db, tasks=TASKS[:1])
    assert stats["resumed"] == 1 and stats["completed"] == 1 and runs[0]["passed"]
    assert [kind for kind, _ in scenario.calls[len(calls_before):]] == ["Review"]
    assert runs[0]["artifacts"]["review"].endswith("No findings.")


def test_completed_checkpoints_are_collected_without_running_again(tmp_path):
    checkpoint_db = str(tmp_path / "checkpoints.db")
    scenario = Scenario()
    run(scenario, tmp_path, samples_per_task=1, checkpoint_db=checkpoint_db, tasks=TASKS[:2])
    call_count = len(scenario.calls)
    (tmp_path / "runs.jsonl").unlink()  # finished in the checkpoint, but the records were lost
    stats, runs = run(scenario, tmp_path, samples_per_task=1, checkpoint_db=checkpoint_db, tasks=TASKS[:2])
    assert stats["from_checkpoint"] == 2 and stats["completed"] == 2 and len(scenario.calls) == call_count
    assert all(run_record["passed"] for run_record in runs)


def test_checkpoints_never_leak_into_another_or_a_cleared_output_directory(tmp_path):
    # the Kaggle bug: a shared checkpoint DB replayed an old round's results into a new directory
    checkpoint_db = str(tmp_path / "checkpoints.db")
    scenario = Scenario()
    first, second = tmp_path / "round0" / "baseline", tmp_path / "round1" / "baseline"
    run(scenario, first, samples_per_task=1, checkpoint_db=checkpoint_db, tasks=TASKS[:2])
    call_count = len(scenario.calls)
    stats, _ = run(scenario, second, samples_per_task=1, checkpoint_db=checkpoint_db, tasks=TASKS[:2])
    assert stats["from_checkpoint"] == 0 and stats["completed"] == 2 and len(scenario.calls) > call_count
    shutil.rmtree(first)  # clearing a directory also clears its checkpoints
    stats, _ = run(scenario, first, samples_per_task=1, checkpoint_db=checkpoint_db, tasks=TASKS[:2])
    assert stats["from_checkpoint"] == 0 and stats["completed"] == 2


def test_run_records_have_safe_defaults_for_aborted_runs():
    run_record = state_to_run({"task": "t", "failure_reason": "design: boom", "llm_calls": []}, 1, 0, "system")
    assert run_record["passed"] is False and run_record["test_count"] == 0
    assert run_record["artifacts"]["tests"] == ""
    with pytest.raises(KeyError):
        _ = run_record["artifacts"]["nonexistent"]


# ---- serving, models, environment --------------------------------------------------------

def test_server_command_enables_tool_calling_and_optionally_lora():
    config = Config()
    command = build_server_command(config)
    assert command[:3] == ["vllm", "serve", config.model.path]
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert command[command.index("--tensor-parallel-size") + 1] == "2" and "--enable-lora" not in command
    with_adapter = build_server_command(config, "/path/adapter")
    assert "adapter=/path/adapter" in with_adapter and "--enable-lora" in with_adapter
    assert served_model_name(config, None) == "harness" and served_model_name(config, "/x") == "adapter"
    assert config.serve.base_url == "http://127.0.0.1:8000/v1"


def test_chat_model_factory_caches_one_model_per_temperature():
    built = []
    factory = ChatModelFactory(Config(), build_model=lambda temperature: built.append(temperature) or object())
    first, second, third = factory.get(0.2), factory.get(0.2), factory.get(0.7)
    assert first is second and first is not third and built == [0.2, 0.7]


def test_langsmith_is_opt_in_and_never_prints_the_key(monkeypatch, capsys):
    for name in ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    config = Config()
    assert setup_langsmith(config) is False and os.environ["LANGSMITH_TRACING"] == "false"
    config.langsmith.enabled = True
    assert setup_langsmith(config) is False  # enabled but no key: stays off, and says so
    assert "tracing is OFF" in capsys.readouterr().out
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_SECRET")
    config.langsmith.project, config.langsmith.hide_inputs = "my-project", True
    assert setup_langsmith(config) is True
    assert os.environ["LANGSMITH_TRACING"] == "true" and os.environ["LANGSMITH_PROJECT"] == "my-project"
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true" and os.environ["LANGSMITH_HIDE_OUTPUTS"] == "false"
    assert "lsv2_SECRET" not in capsys.readouterr().out
