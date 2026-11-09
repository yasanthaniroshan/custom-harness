"""Typed configuration loaded from YAML.

Every setting has a default here; `configs/default.yaml` mirrors these defaults (a test keeps
them in sync) and `configs/kaggle.yaml` overrides a few of them for Kaggle's 2x T4 machines.
Unknown keys in a YAML file are rejected, so a typo never silently falls back to a default.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """The base model and how it is loaded for training."""

    path: str = "/kaggle/input/qwen2.5-coder/transformers/7b-instruct/1"
    load_in_4bit: bool = True
    quantization_type: str = "nf4"
    double_quantization: bool = True
    compute_dtype: str = "float16"  # T4 GPUs have no bfloat16
    training_gpu_index: int = 0  # QLoRA training uses one GPU


@dataclass
class DataConfig:
    """Where seed tasks and training rows come from."""

    dataset: str = "ise-uiuc/Magicoder-OSS-Instruct-75K"  # seed tasks + optional mixed-in rows
    split: str = "train"
    seed: int = 42
    sft_paths: list[str] = field(default_factory=list)  # filtered self-generated JSONL files
    magicoder_mix: float = 0.25  # share of Magicoder rows in the final mix when sft_paths is set
    magicoder_samples: int = 10_000  # Magicoder rows used when there is no self-generated data
    magicoder_language: str | None = None  # restrict Magicoder rows to one language


@dataclass
class SamplingConfig:
    """Decoding settings sent with every chat request."""

    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class AgentConfig:
    """The implement stage's tool-using agent and the test sandbox."""

    system_prompt: str = "You are a careful senior Python engineer. Be concise."
    code_temperature: float = 0.2  # write_tests and implement at inference time
    test_timeout_seconds: int = 10
    max_test_report_chars: int = 1500  # cap on the pass/fail report shown to the model
    max_agent_steps: int = 12  # model turns in the implement tool loop
    max_error_streak: int = 3  # consecutive all-failed tool turns before the loop gives up
    solution_file: str = "solution.py"  # file the agent is told to write
    workspace_seed: str | None = None  # directory copied into each workspace (repo-level tasks)


@dataclass
class VLLMConfig:
    """How `harness serve` starts the vLLM server."""

    tensor_parallel_size: int = 2  # both T4s; an fp16 7B model needs ~7.5 GB per GPU
    max_model_len: int = 6144
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True
    dtype: str = "half"  # T4 GPUs have no bfloat16


@dataclass
class ServeConfig:
    """The OpenAI-compatible endpoint the pipeline talks to."""

    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: str = "harness"
    tool_call_parser: str = "hermes"  # Qwen2.5's tool-call format
    concurrency: int = 16  # pipeline runs in flight at once (vLLM batches them continuously)
    request_timeout_seconds: float = 300.0
    max_retries: int = 2
    structured_output_method: str = "json_schema"  # json_schema (guided decoding) | function_calling
    tokenizer: str | None = None  # tokenizer path for exact token counts (otherwise ~4 chars/token)
    vllm_executable: str = "vllm"  # e.g. a vllm binary inside a separate virtualenv

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass
class GenerateConfig:
    """Self-generation rounds: run the pipeline on seed tasks, keep verified runs."""

    seed_language: str = "python"
    num_tasks: int = 200  # pilot size; scale up once pass rate and speed are known
    eval_tasks: int = 100  # held out: never used for generation or training
    samples_per_task: int = 4
    prose_temperature: float = 0.8  # requirements, design, review
    code_temperature: float = 0.6  # higher than at inference, for sample diversity
    eval_temperature: float = 0.2
    max_task_chars: int = 1500
    review_only_passed: bool = True  # skip the review stage for runs that failed their tests
    output_dir: str = "outputs/round0"


@dataclass
class FilterConfig:
    """Which generated runs are trusted enough to become training data."""

    min_tests: int = 3
    max_stub_pass_fraction: float = 0.5  # tests that pass against empty code prove nothing
    max_runs_per_task: int = 2
    reject_blockers: bool = True  # drop runs whose review reports a [Blocker]
    max_tool_errors: int = 3


@dataclass
class ContextConfig:
    """Token budgets for prompts. Defaults leave room for generation: 6144 - 2048."""

    max_prompt_tokens: int = 4096
    artifact_max_tokens: int = 2500  # cap per input artifact (requirements, design, code, ...)
    test_output_max_tokens: int = 400
    conventions: dict[str, str] = field(default_factory=dict)  # stage name or "*" -> text
    few_shot_path: str | None = None  # filtered SFT JSONL to retrieve examples from
    few_shot_count: int = 0  # examples injected per stage (0 = off)
    few_shot_max_tokens: int = 600  # larger examples are skipped, never truncated


@dataclass
class ToolsConfig:
    """Limits for the workspace tools the implement agent uses."""

    max_result_tokens: int = 800
    max_read_lines: int = 200
    max_grep_matches: int = 50
    max_glob_results: int = 100
    max_repeat_calls: int = 2  # identical consecutive calls allowed before refusing
    hidden_globs: list[str] = field(default_factory=list)  # e.g. ["tests/*"]


@dataclass
class GraphConfig:
    """Quality gates and best-of-N in the SDLC graph."""

    candidates: int = 1  # parallel implement attempts per task (best-of-N)
    min_tests: int = 3
    max_stub_pass_fraction: float = 0.5
    max_test_retries: int = 2  # times an invalid test suite is regenerated with feedback
    abort_on_invalid_tests: bool = True  # stop instead of implementing against bad tests
    checkpoint_db: str | None = None  # SQLite file that lets interrupted runs resume


@dataclass
class MonitorConfig:
    enabled: bool = True
    events_file: str = "events.jsonl"  # written inside the round directory


@dataclass
class LangSmithConfig:
    """Opt-in tracing. When enabled, prompts, code and outputs are sent to LangSmith."""

    enabled: bool = False
    project: str = "custom-harness"
    tags: list[str] = field(default_factory=list)
    hide_inputs: bool = False
    hide_outputs: bool = False
    log_feedback: bool = True  # attach tests_passed / trusted scores to each traced run


@dataclass
class LoraAdapterConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    output_dir: str = "outputs/qwen-coder-sft"
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: float = 2.0
    max_steps: int = -1  # > 0 overrides epochs (smoke runs)
    max_length: int = 2048
    drop_overlong: bool = True  # never train on a row whose completion would be truncated
    warmup_ratio: float = 0.03
    learning_rate_scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch"  # paged_adamw_8bit saves memory on a T4 (needs bitsandbytes)
    fp16: bool = True  # T4 GPUs have no bfloat16
    gradient_checkpointing: bool = True
    seed: int = 42
    validation_fraction: float = 0.05  # held out BY TASK: a task is never in both splits
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50  # must be a multiple of eval_steps for best-model loading
    save_total_limit: int = 2
    early_stopping_patience: int = 3  # evaluations without eval_loss improvement; 0 = off


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    langsmith: LangSmithConfig = field(default_factory=LangSmithConfig)
    lora: LoraAdapterConfig = field(default_factory=LoraAdapterConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def apply_overrides(target: Any, overrides: dict[str, Any]) -> None:
    """Recursively copy `overrides` onto a config dataclass, rejecting unknown keys."""
    valid_keys = {config_field.name for config_field in fields(target)}
    for key, value in overrides.items():
        if key not in valid_keys:
            raise KeyError(f"Unknown config key '{key}' for {type(target).__name__}")
        current_value = getattr(target, key)
        if is_dataclass(current_value) and isinstance(value, dict):
            apply_overrides(current_value, value)
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None = None) -> Config:
    config = Config()
    if path is not None:
        overrides = yaml.safe_load(Path(path).read_text()) or {}
        apply_overrides(config, overrides)
    return config
