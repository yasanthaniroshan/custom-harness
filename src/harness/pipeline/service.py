"""High-level entry points the CLI uses: assemble the dependencies, then generate training runs,
evaluate on held-out tasks, or run one task interactively."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from harness.config import Config
from harness.environment import setup_huggingface_environment, setup_langsmith
from harness.monitoring import tracer_for
from harness.pipeline.chat_models import ChatModelFactory
from harness.pipeline.dependencies import PipelineDependencies
from harness.pipeline.runner import open_pipeline_graph, read_jsonl, run_tasks
from harness.prompt_context import (
    PromptBuilder,
    TokenCounter,
    approximate_token_count,
    tokenizer_token_counter,
)
from harness.serving import require_server


def make_token_counter(config: Config) -> TokenCounter:
    """Exact token counts when serve.tokenizer is set, otherwise ~4 characters per token."""
    if config.serve.tokenizer:
        from transformers import AutoTokenizer

        return tokenizer_token_counter(AutoTokenizer.from_pretrained(config.serve.tokenizer))
    return approximate_token_count


def make_feedback_client(config: Config, tracing_enabled: bool):
    if not (tracing_enabled and config.langsmith.log_feedback):
        return None
    from langsmith import Client

    return Client()


@dataclass
class RunSettings:
    """How one batch of pipeline runs is executed and where it is written."""

    samples_per_task: int
    prose_temperature: float
    code_temperature: float
    review_only_passed: bool
    round_name: str
    events_file_name: str
    runs_file_name: str


def run_to_directory(config: Config, tasks: list[dict], output_dir: Path, adapter_path: str | None,
                     settings: RunSettings) -> Path:
    """Run the pipeline over `tasks` and return the runs JSONL path (resumable)."""
    require_server(config)
    setup_huggingface_environment()
    tracing_enabled = setup_langsmith(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracer = tracer_for(config.monitor.enabled, output_dir / settings.events_file_name)
    count_tokens = make_token_counter(config)
    dependencies = PipelineDependencies(
        config=config,
        chat_models=ChatModelFactory(config, adapter_path),
        prompt_builder=PromptBuilder(config.context, count_tokens, tracer),
        tracer=tracer,
        prose_temperature=settings.prose_temperature,
        code_temperature=settings.code_temperature,
        review_only_passed=settings.review_only_passed,
    )
    runs_path = output_dir / settings.runs_file_name  # distinct per use: runs resume by (task, sample)
    with open_pipeline_graph(dependencies) as (graph, checkpointed):
        run_tasks(graph, config, tasks, settings.samples_per_task, runs_path, settings.round_name,
                  tracer, count_tokens, checkpointed, make_feedback_client(config, tracing_enabled))
    return runs_path


def generate_runs(config: Config, tasks: list[dict], output_dir: Path, adapter_path: str | None = None) -> Path:
    """Self-generation: several samples per task at the (higher) generation temperatures."""
    generate = config.generate
    return run_to_directory(config, tasks, output_dir, adapter_path, RunSettings(
        samples_per_task=generate.samples_per_task,
        prose_temperature=generate.prose_temperature,
        code_temperature=generate.code_temperature,
        review_only_passed=generate.review_only_passed,
        round_name=output_dir.name,
        events_file_name=config.monitor.events_file,
        runs_file_name="runs.jsonl",
    ))


def evaluate_on_tasks(config: Config, eval_tasks: list[dict], output_dir: Path,
                      adapter_path: str | None = None) -> dict:
    """One low-temperature run per held-out task; writes and returns eval.json."""
    from harness.evaluation.harness_metrics import summarize_runs

    runs_path = run_to_directory(config, eval_tasks, output_dir, adapter_path, RunSettings(
        samples_per_task=1,
        prose_temperature=config.generate.eval_temperature,
        code_temperature=config.agent.code_temperature,
        review_only_passed=True,
        round_name=f"{output_dir.name}-eval",
        events_file_name="eval_events.jsonl",
        runs_file_name="eval_runs.jsonl",
    ))
    summary = summarize_runs(read_jsonl(runs_path), config)
    (output_dir / "eval.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_single_task(config: Config, task: str, adapter_path: str | None = None) -> dict:
    """Interactive single task. Always reviews, even if the tests failed."""
    output_dir = Path(tempfile.mkdtemp(prefix="harness_run_"))
    runs_path = run_to_directory(config, [{"task_id": 0, "task": task}], output_dir, adapter_path, RunSettings(
        samples_per_task=1,
        prose_temperature=config.sampling.temperature,
        code_temperature=config.agent.code_temperature,
        review_only_passed=False,
        round_name="interactive",
        events_file_name=config.monitor.events_file,
        runs_file_name="runs.jsonl",
    ))
    return read_jsonl(runs_path)[0]
