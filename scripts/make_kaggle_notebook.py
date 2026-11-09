"""Generate notebooks/kaggle_pipeline.ipynb. Cells are plain Python (no magics) so they can be
compiled in tests. Edit the cell text here and re-run: `python scripts/make_kaggle_notebook.py`."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle_pipeline.ipynb"

INTRO = """# Harness fine-tuning pipeline (Kaggle, 2x T4)

One round of the self-improvement loop, in stages you can run in separate sessions
(set `STAGE` below). **Settings -> Accelerator: GPU T4 x2, Internet: on.**

| STAGE | What it does | GPUs |
|---|---|---|
| `setup` | install code + deps, check GPUs | - |
| `baseline` | serve base model, evaluate on held-out tasks, HumanEval (direct + harness) | both (vLLM) |
| `generate` | run the harness over seed tasks, filter, **validate the training data** | both (vLLM) |
| `train` | stop the server, QLoRA fine-tune with validation + early stopping | GPU 0 |
| `evaluate` | serve the tuned adapter, re-evaluate, **compare with the baseline** | both (vLLM) |
| `save` | package adapter + metrics into /kaggle/working | - |

vLLM and training cannot share the GPUs, which is why they are separate stages.
Kaggle sessions last <= 12 h and GPU quota is limited, so run a **pilot first** (defaults are
small) and read the numbers before scaling `NUM_TASKS`.
"""

SETTINGS = '''# ---- settings (edit me) -------------------------------------------------------------------
ROUND = 0                  # round number; round 1+ trains from base on ALL accumulated data
STAGE = "setup"            # setup | baseline | generate | train | evaluate | save | all
NUM_TASKS = 50             # seed tasks to generate from (pilot size)
SAMPLES = 2                # runs per task
EVAL_TASKS = 40            # held-out tasks (fixed by seed; never trained on)
BENCH_LIMIT = 40           # HumanEval problems (164 = all)
# ---- where the code comes from ----
# Recommended: a GitHub repo pinned to a tag or commit, so every round is reproducible.
# Private repo: add a fine-grained, read-only token as the Kaggle secret GITHUB_TOKEN.
# Empty REPO_URL = use the attached Kaggle dataset built by `python scripts/bundle.py`.
REPO_URL = ""              # e.g. "https://github.com/<you>/custom-harness-development.git"
GIT_REF = "main"           # a tag (e.g. "v0.3.0") or commit SHA; a branch name is not reproducible
PREV_SFT = []              # filtered data from earlier rounds, e.g. ["/kaggle/input/harness-r0/round0/sft.jsonl"]
PREV_ADAPTER = None        # adapter that generates round>0 data, e.g. "/kaggle/input/harness-r0/train0/adapter"
VLLM_SPEC = "vllm"         # pin if the latest vLLM drops T4 support, e.g. "vllm==0.10.2"
'''

HELPERS = '''import os, subprocess, sys, json, time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "repo"
OUT = WORK / "outputs"
ROUND_DIR = OUT / f"round{ROUND}"
TRAIN_DIR = OUT / f"train{ROUND}"
CONFIG = WORK / "run.yaml"
ADAPTER = TRAIN_DIR / "adapter"


def sh(command, check=True):
    """Run a shell command and stream its output into the cell (plain subprocess output would
    only reach the kernel log)."""
    print("$", command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if check and process.returncode:
        raise RuntimeError(f"command failed with exit code {process.returncode}: {command}")
    return process.returncode


def stage(*names):
    """True when the selected STAGE is one of `names` (or "all")."""
    return STAGE == "all" or STAGE in names


def harness(arguments):
    sh(f"harness --config {CONFIG} {arguments}")


def write_config():
    """kaggle.yaml + this round's settings -> run.yaml (the one file every command reads)."""
    import yaml

    settings = yaml.safe_load((REPO / "configs" / "kaggle.yaml").read_text())
    settings["generate"].update(num_tasks=NUM_TASKS, eval_tasks=EVAL_TASKS, samples_per_task=SAMPLES,
                                output_dir=str(ROUND_DIR))
    settings["train"]["output_dir"] = str(TRAIN_DIR)
    settings.setdefault("data", {})["sft_paths"] = [*PREV_SFT, str(ROUND_DIR / "sft.jsonl")]
    CONFIG.write_text(yaml.safe_dump(settings))
    print("wrote", CONFIG)


def clone_repository(url, ref, destination):
    """Clone `url` at `ref`. A GITHUB_TOKEN secret (private repos) is used but never printed,
    and is removed from the clone's remote URL afterwards."""
    token = None
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        pass  # public repo, or the secret is not attached
    authenticated = url.replace("https://", f"https://x-access-token:{token}@") if token else url
    print(f"$ git clone {url} (ref {ref})")
    for git_command in (["git", "clone", "--quiet", authenticated, str(destination)],
                        ["git", "-C", str(destination), "checkout", "--quiet", ref],
                        ["git", "-C", str(destination), "remote", "set-url", "origin", url]):
        result = subprocess.run(git_command, capture_output=True, text=True)
        if result.returncode:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError("git failed: " + (message.replace(token, "***") if token else message))


def record_code_version():
    """Write which code produced this round's outputs (commit SHA or dataset), for reproducibility."""
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    version = head.stdout.strip() if head.returncode == 0 else "kaggle-dataset (no git metadata)"
    (ROUND_DIR / "code_version.txt").write_text(version + "\\n")
    print("code version:", version)
'''

SETUP = '''if stage("setup"):
    sh("nvidia-smi -L")
    # 1) the project code: a pinned git checkout, or the dataset built by scripts/bundle.py
    if not REPO.exists():
        if REPO_URL:
            clone_repository(REPO_URL, GIT_REF, REPO)
        else:
            bundled_source = next((manifest.parent for manifest in Path("/kaggle/input").glob("*/pyproject.toml")), None)
            assert bundled_source, "set REPO_URL, or attach the dataset built by `python scripts/bundle.py`"
            sh(f"cp -r {bundled_source} {REPO}")
    # 2) the project + training stack into the notebook environment (keeps Kaggle's torch)
    sh(f"pip install -q -e '{REPO}[train]'")
    # 3) vLLM in its OWN virtualenv: installing it here could replace torch and break training
    if not (WORK / "vllm-env").exists():
        sh(f"python -m venv {WORK / 'vllm-env'}")
        sh(f"{WORK / 'vllm-env' / 'bin' / 'pip'} install -q {VLLM_SPEC}")
    write_config()
    OUT.mkdir(parents=True, exist_ok=True)
    record_code_version()

# secrets: attach them under Add-ons -> Secrets. They are only put in the environment, never printed.
if stage("setup", "baseline", "generate", "train", "evaluate"):
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    for name in ("HF_TOKEN", "LANGSMITH_API_KEY"):
        try:
            os.environ[name] = secrets.get_secret(name)
        except Exception:
            print(f"secret {name} not set (fine unless you need it)")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if not CONFIG.exists():
        write_config()
'''

BASELINE = '''if stage("baseline"):
    harness("serve --background")                              # base model on both GPUs
    harness(f"eval --out {ROUND_DIR / 'baseline'}")            # held-out self-eval (noisy proxy)
    harness(f"bench --mode direct  --limit {BENCH_LIMIT} --out {ROUND_DIR / 'baseline'}")   # no harness
    harness(f"bench --mode harness --limit {BENCH_LIMIT} --out {ROUND_DIR / 'baseline'}")
    harness(f"report {ROUND_DIR / 'baseline'}")                 # CHECK: truncation + tool-error rates
    harness("serve --stop")
'''

GENERATE = '''if stage("generate"):
    # round > 0 generates with the previous adapter; round 0 uses the base model
    adapter = f"--adapter {PREV_ADAPTER}" if PREV_ADAPTER else ""
    harness(f"serve --background {adapter}")
    harness(f"generate --out {ROUND_DIR} {adapter}")
    harness("serve --stop")
    harness(f"report {ROUND_DIR}")                             # CHECK before trusting the data
    harness(f"filter --runs {ROUND_DIR / 'runs.jsonl'} --out {ROUND_DIR / 'sft.jsonl'}")
    # blocks training on structural errors, schema errors and hidden-test leaks
    harness("validate-data --data " + " ".join([*PREV_SFT, str(ROUND_DIR / "sft.jsonl")]))
'''

TRAIN = '''if stage("train"):
    harness("serve --stop")                                    # free both GPUs first
    harness("train")                                           # validates data, trains, writes metrics
    harness(f"train-report {TRAIN_DIR}")                       # CHECK: overfitting flag, best eval step
    print(json.dumps(json.loads((TRAIN_DIR / "train_summary.json").read_text()).get("per_stage_eval"), indent=1))
'''

EVALUATE = '''if stage("evaluate"):
    harness(f"serve --background --adapter {ADAPTER}")
    harness(f"eval --adapter {ADAPTER} --out {ROUND_DIR / 'tuned'}")
    harness(f"bench --mode direct  --adapter {ADAPTER} --limit {BENCH_LIMIT} --out {ROUND_DIR / 'tuned'}")
    harness(f"bench --mode harness --adapter {ADAPTER} --limit {BENCH_LIMIT} --out {ROUND_DIR / 'tuned'}")
    harness("serve --stop")
    base, tuned = ROUND_DIR / "baseline", ROUND_DIR / "tuned"
    print("=== self-eval (held-out tasks) ===")
    harness(f"compare {base} {tuned}")
    print("=== HumanEval, harness: does the pipeline improve? ===")
    harness(f"compare {base / 'humaneval_harness.json'} {tuned / 'humaneval_harness.json'}")
    print("=== HumanEval, direct model: did fine-tuning HURT general coding (forgetting)? ===")
    harness(f"compare {base / 'humaneval_direct.json'} {tuned / 'humaneval_direct.json'}")
'''

SAVE = '''if stage("save"):
    sh(f"cd {OUT} && zip -qr {WORK / f'round{ROUND}_results.zip'} round{ROUND} train{ROUND} "
       f"--exclude '*/checkpoint-*'")
    print("Saved", WORK / f"round{ROUND}_results.zip")
    print("Now: Save Version -> Save & Run All (Commit). Then add this notebook's output as a "
          "Dataset to use it as PREV_SFT / PREV_ADAPTER in the next round.")
'''


def cell(kind: str, source: str, number: int) -> dict:
    notebook_cell = {"id": f"cell-{number}", "cell_type": kind, "metadata": {},
                     "source": source.splitlines(keepends=True)}
    return notebook_cell | ({"outputs": [], "execution_count": None} if kind == "code" else {})


def build() -> dict:
    cells = [("markdown", INTRO), ("code", SETTINGS + "\n" + HELPERS), ("code", SETUP),
             ("code", BASELINE), ("code", GENERATE), ("code", TRAIN), ("code", EVALUATE), ("code", SAVE)]
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [cell(kind, source, number) for number, (kind, source) in enumerate(cells, 1)],
    }


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1))
    print("wrote", OUT)
