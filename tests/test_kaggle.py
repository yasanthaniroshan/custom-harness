import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from harness.config import Config, load_config
from harness.serving import pid_file_for, stop_server

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_committed_notebook_is_up_to_date_and_its_cells_compile():
    import make_kaggle_notebook

    committed = json.loads((PROJECT_ROOT / "notebooks" / "kaggle_pipeline.ipynb").read_text())
    expected = json.loads(json.dumps(make_kaggle_notebook.build()))
    assert committed == expected, "run: python scripts/make_kaggle_notebook.py"
    code_cells = [cell for cell in committed["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 7
    assert len({cell["id"] for cell in committed["cells"]}) == len(committed["cells"])
    for cell in code_cells:
        compile("".join(cell["source"]), cell["id"], "exec")  # plain Python: no magics


def test_notebook_drives_every_stage_through_the_cli_and_never_prints_secrets():
    notebook = json.loads((PROJECT_ROOT / "notebooks" / "kaggle_pipeline.ipynb").read_text())
    text = "".join("".join(cell["source"]) for cell in notebook["cells"])
    for command in ("serve --background", "serve --stop", "eval", "bench --mode direct",
                    "bench --mode harness", "generate", "filter", "validate-data", "train",
                    "train-report", "compare", "report"):
        assert command in text, command
    assert "print(os.environ" not in text and "print(secrets" not in text
    assert text.index("serve --stop") < text.index('harness("train")')  # GPUs freed before training
    assert text.index("validate-data") < text.index('harness("train")')
    # the code is pinned and recorded, and a GitHub token never reaches the output
    assert "GIT_REF" in text and "record_code_version()" in text
    assert 'print(f"$ git clone {url} (ref {ref})")' in text  # the clean URL, never the token
    assert 'replace(token, "***")' in text  # git errors are redacted
    # Kaggle's Python has no ensurepip: the vLLM env is built with uv, never `python -m venv`
    assert "-m venv" not in text and "uv venv" in text and "uv pip install" in text


def test_kaggle_config_loads_with_no_unknown_keys_and_sets_t4_friendly_values():
    config = load_config(PROJECT_ROOT / "configs" / "kaggle.yaml")  # raises KeyError on an unknown key
    assert config.model.path.startswith("/kaggle/input") and config.vllm.tensor_parallel_size == 2
    assert config.train.optimizer == "paged_adamw_8bit" and config.train.fp16 and config.train.max_length == 2048
    assert config.serve.vllm_executable == "/kaggle/working/vllm-env/bin/vllm"
    assert config.model.training_gpu_index == 0
    assert config.graph.checkpoint_db.startswith("/kaggle/working")


def test_default_yaml_mirrors_the_config_dataclasses():
    assert asdict(load_config(PROJECT_ROOT / "configs" / "default.yaml")) == asdict(Config())


def test_unknown_config_keys_are_rejected(tmp_path):
    typo = tmp_path / "typo.yaml"
    typo.write_text(yaml.safe_dump({"train": {"learning_rat": 0.1}}))
    with pytest.raises(KeyError, match="learning_rat"):
        load_config(typo)


def test_bundle_contains_source_and_excludes_environments_and_outputs(tmp_path):
    import bundle

    names = {path.relative_to(PROJECT_ROOT).as_posix() for path in bundle.files()}
    assert {"pyproject.toml", "src/harness/cli.py", "configs/kaggle.yaml"} <= names
    assert not any(".venv" in name or "__pycache__" in name or name.startswith("outputs") or name.endswith(".pyc")
                   for name in names)


def test_stop_server_kills_the_whole_process_group_and_cleans_up(tmp_path):
    log = tmp_path / "vllm.log"
    assert stop_server(log) is False  # nothing recorded
    server = subprocess.Popen(["sleep", "300"], start_new_session=True)  # stands in for vllm + workers
    pid_file_for(log).write_text(str(server.pid))
    assert stop_server(log, timeout_seconds=10) is True
    time.sleep(0.2)
    assert server.poll() is not None and not pid_file_for(log).exists()
    stale = pid_file_for(log)
    stale.write_text("999999")  # a pid that no longer exists
    assert stop_server(log) is True and not stale.exists()
    assert os.getpgid(os.getpid())  # we did not kill ourselves
    assert signal.SIGTERM
