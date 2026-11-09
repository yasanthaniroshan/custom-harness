"""Command line interface: `harness <command> [options]`.

Every command is a small function `command_<name>(arguments, config)`. Heavy dependencies are
imported inside the command that needs them, so `harness --help` stays fast and the commands
that only read files work without the training stack.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from harness.config import Config, load_config

DEFAULT_SERVER_LOG = "outputs/vllm_server.log"


# ---- pipeline runs ----------------------------------------------------------------------

def command_run(arguments: argparse.Namespace, config: Config) -> None:
    from harness.pipeline.service import run_single_task

    run = run_single_task(config, arguments.task, arguments.adapter)
    if run["failure_reason"]:
        print("FAILED:", run["failure_reason"])
    print("TESTS PASSED:", run["passed"], "| agent steps:", run["steps"])
    print(run["artifacts"]["code"])
    print(run["artifacts"]["review"])


def command_generate(arguments: argparse.Namespace, config: Config) -> None:
    from harness.pipeline.service import generate_runs
    from harness.training_data.seed_tasks import load_seed_tasks

    output_dir = Path(arguments.out or config.generate.output_dir)
    train_tasks, _ = load_seed_tasks(config)
    print("Runs written to", generate_runs(config, train_tasks, output_dir, arguments.adapter))


def command_eval(arguments: argparse.Namespace, config: Config) -> None:
    from harness.pipeline.service import evaluate_on_tasks
    from harness.training_data.seed_tasks import load_seed_tasks

    _, eval_tasks = load_seed_tasks(config)
    output_dir = Path(arguments.out or config.generate.output_dir)
    print(json.dumps(evaluate_on_tasks(config, eval_tasks, output_dir, arguments.adapter), indent=2))


def command_bench(arguments: argparse.Namespace, config: Config) -> None:
    from harness.evaluation import humaneval
    from harness.serving import require_server

    require_server(config)
    problems = humaneval.load_humaneval(arguments.limit)
    output_dir = Path(arguments.out)
    if arguments.mode == "direct":
        results = humaneval.run_direct(config, problems, arguments.adapter)
    else:
        results = humaneval.run_harness(config, problems, output_dir, arguments.adapter)
    path = humaneval.save_results(output_dir, arguments.mode, arguments.adapter, results)
    print(json.dumps(humaneval.summarize_results(results), indent=2), "\nSaved", path)


# ---- training data ----------------------------------------------------------------------

def command_filter(arguments: argparse.Namespace, config: Config) -> None:
    from harness.training_data.trajectory_filter import filter_run_files

    round_dir = Path(config.generate.output_dir)
    runs_paths = [Path(path) for path in arguments.runs] if arguments.runs else [round_dir / "runs.jsonl"]
    output_path = Path(arguments.out) if arguments.out else round_dir / "sft.jsonl"
    stats = filter_run_files(runs_paths, output_path, config.filter, config.agent.test_timeout_seconds)
    print(json.dumps(dict(stats), indent=2), "\nSFT rows written to", output_path)


def command_validate_data(arguments: argparse.Namespace, config: Config) -> None:
    from harness.training_data.sft_dataset import load_rows
    from harness.training_data.validation import validate_rows

    tokenizer = None
    if not arguments.no_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer or config.serve.tokenizer or config.model.path)
    report = validate_rows(load_rows(arguments.data), tokenizer, config.train.max_length,
                           config.train.validation_fraction, config.train.seed)
    print(report.format())
    raise SystemExit(0 if report.ok else 1)


# ---- training -----------------------------------------------------------------------------

def command_train(arguments: argparse.Namespace, config: Config) -> None:
    from harness.training.trainer import run_training

    if arguments.sft_data:
        config.data.sft_paths = arguments.sft_data
    if arguments.output_dir:
        config.train.output_dir = arguments.output_dir
    if arguments.max_steps:
        config.train.max_steps = arguments.max_steps
    print("Adapter saved to", run_training(config, skip_validation=arguments.skip_validation))


def command_train_report(arguments: argparse.Namespace, config: Config) -> None:
    from harness.training.train_log import TRAIN_LOG_FILE, summarize_train_log

    path = Path(arguments.log)
    print(json.dumps(summarize_train_log(path / TRAIN_LOG_FILE if path.is_dir() else path), indent=2))


# ---- measuring --------------------------------------------------------------------------

def command_compare(arguments: argparse.Namespace, config: Config) -> None:
    from harness.evaluation.comparison import compare_paths

    print(compare_paths(arguments.a, arguments.b, config))


def command_report(arguments: argparse.Namespace, config: Config) -> None:
    from harness.monitoring import format_report, read_events, summarize_events

    path = Path(arguments.events)
    events_files = sorted(path.glob("*events.jsonl")) if path.is_dir() else [path]
    if not events_files:
        raise SystemExit(f"error: no *events.jsonl file in {path}")
    for events_file in events_files:  # e.g. events.jsonl, eval_events.jsonl, humaneval_events.jsonl
        if len(events_files) > 1:
            print(f"\n=== {events_file.name} ===")
        print(format_report(summarize_events(read_events(events_file))))


# ---- serving ----------------------------------------------------------------------------

def command_serve(arguments: argparse.Namespace, config: Config) -> None:
    from harness import serving

    log_path = Path(arguments.log)
    if arguments.stop:
        print("stopped" if serving.stop_server(log_path) else "no background server recorded")
    elif arguments.background:
        process = serving.start_in_background(config, arguments.adapter, log_path)
        serving.wait_until_ready(config.serve.base_url, process=process)
        print(f"vLLM server ready at {config.serve.base_url} (pid {process.pid}, log {log_path})")
    else:
        subprocess.run(serving.build_server_command(config, arguments.adapter), check=False)


COMMANDS: dict[str, Callable[[argparse.Namespace, Config], None]] = {
    "run": command_run,
    "generate": command_generate,
    "eval": command_eval,
    "bench": command_bench,
    "filter": command_filter,
    "validate-data": command_validate_data,
    "train": command_train,
    "train-report": command_train_report,
    "compare": command_compare,
    "report": command_report,
    "serve": command_serve,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness")
    parser.add_argument("--config", help="YAML file overriding the defaults (e.g. configs/kaggle.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_adapter_option(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--adapter", help="LoRA adapter directory (it must also be served)")

    run_parser = commands.add_parser("run", help="Run the SDLC pipeline on one task")
    run_parser.add_argument("task")
    add_adapter_option(run_parser)

    generate_parser = commands.add_parser("generate", help="Run the pipeline over seed tasks, record every run")
    generate_parser.add_argument("--out", help="Round directory (default: generate.output_dir)")
    add_adapter_option(generate_parser)

    eval_parser = commands.add_parser("eval", help="Evaluate the pipeline on the held-out tasks")
    eval_parser.add_argument("--out", help="Directory for eval.json (default: generate.output_dir)")
    add_adapter_option(eval_parser)

    bench_parser = commands.add_parser("bench", help="HumanEval pass@1 (direct model or full pipeline)")
    bench_parser.add_argument("--mode", choices=["direct", "harness"], required=True)
    bench_parser.add_argument("--limit", type=int, help="first N problems only")
    bench_parser.add_argument("--out", default="outputs/bench")
    add_adapter_option(bench_parser)

    filter_parser = commands.add_parser("filter", help="Keep trustworthy runs and write SFT rows")
    filter_parser.add_argument("--runs", nargs="+", help="runs.jsonl file(s) (default: <round dir>/runs.jsonl)")
    filter_parser.add_argument("--out", help="Output SFT JSONL (default: <round dir>/sft.jsonl)")

    validate_parser = commands.add_parser("validate-data", help="Check SFT rows before training")
    validate_parser.add_argument("--data", nargs="+", required=True, help="SFT JSONL file(s) from `harness filter`")
    validate_parser.add_argument("--tokenizer", help="Tokenizer path for exact token stats (default: model path)")
    validate_parser.add_argument("--no-tokenizer", action="store_true", help="structural checks only")

    train_parser = commands.add_parser("train", help="QLoRA fine-tune the model")
    train_parser.add_argument("--sft-data", nargs="+", help="Filtered self-generated JSONL file(s)")
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--skip-validation", action="store_true", help="train even if data validation fails")

    train_report_parser = commands.add_parser("train-report", help="Summarize a train_log.jsonl")
    train_report_parser.add_argument("log", help="train_log.jsonl or the training output directory")

    compare_parser = commands.add_parser("compare", help="Compare two eval outputs (intervals + paired test)")
    compare_parser.add_argument("a", help="baseline: round directory or humaneval_*.json")
    compare_parser.add_argument("b", help="candidate: same kind as A")

    report_parser = commands.add_parser("report", help="Summarize an events.jsonl monitoring file")
    report_parser.add_argument("events", help="events.jsonl or a round directory")

    serve_parser = commands.add_parser("serve", help="Start the vLLM server the pipeline uses")
    add_adapter_option(serve_parser)
    serve_parser.add_argument("--background", action="store_true", help="Detach, wait until ready, return")
    serve_parser.add_argument("--stop", action="store_true", help="Stop a background server (frees the GPUs)")
    serve_parser.add_argument("--log", default=DEFAULT_SERVER_LOG)
    return parser


def main(argv: list[str] | None = None) -> None:
    from harness.serving import ServerNotRunning

    arguments = build_parser().parse_args(argv)
    config = load_config(arguments.config)
    try:
        COMMANDS[arguments.command](arguments, config)
    except ServerNotRunning as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
