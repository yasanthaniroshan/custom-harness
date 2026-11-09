"""Real TRL training on CPU with a tiny random Qwen2 model and the cached Qwen tokenizer.

Verifies the whole fine-tuning path (rendering -> pre-tokenized data -> SFTTrainer -> adapter ->
per-stage metrics) before a Kaggle session is spent on it. Learning is not checked: a random
model's loss only needs to be finite. Nothing is downloaded; skipped without the train extra
or the cached tokenizer.
"""
import copy
import json
import math

import pytest

pytest.importorskip("trl")
pytest.importorskip("peft")

from harness.config import Config
from harness.training.trainer import run_training
from tests.fakes import cached_qwen_tokenizer, make_sft_rows, write_jsonl


@pytest.fixture(scope="module")
def tokenizer():
    return cached_qwen_tokenizer()


def tiny_model(tokenizer):
    from transformers import AutoModelForCausalLM, Qwen2Config

    return AutoModelForCausalLM.from_config(Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192))


def config_for(tmp_path, rows):
    data_path = write_jsonl(tmp_path / "sft.jsonl", rows)
    config = Config()
    config.data.sft_paths, config.data.magicoder_mix = [str(data_path)], 0.0  # no Magicoder download
    train = config.train
    train.output_dir = str(tmp_path / "out")
    train.fp16, train.gradient_checkpointing, train.optimizer = False, False, "adamw_torch"  # CPU
    train.max_length, train.batch_size, train.gradient_accumulation_steps, train.max_steps = 8192, 1, 2, 4
    train.eval_steps = train.save_steps = 2
    train.logging_steps, train.validation_fraction, train.early_stopping_patience = 1, 0.4, 0
    return config


@pytest.fixture(scope="module")
def rows():
    return make_sft_rows(8)


def test_fine_tuning_produces_an_adapter_and_per_stage_metrics(tmp_path, rows, tokenizer):
    config = config_for(tmp_path, rows)
    adapter = run_training(config, tiny_model(tokenizer), tokenizer)
    assert (adapter / "adapter_config.json").exists()
    assert any(path.name.startswith("adapter_model") for path in adapter.iterdir())
    out = tmp_path / "out"
    summary = json.loads((out / "train_summary.json").read_text())
    assert summary["global_step"] == 4 and math.isfinite(summary["train_loss"])
    assert summary["best_checkpoint"] and math.isfinite(summary["best_eval_loss"])
    per_stage = summary["per_stage_eval"]
    assert set(per_stage) <= {"requirements", "design", "write_tests", "implement", "review"} and per_stage
    assert all(math.isfinite(stage["loss"]) and stage["rows"] > 0 for stage in per_stage.values())
    assert summary["data"]["train"]["kept"] > 0 and summary["data"]["validation"]["kept"] > 0
    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert any("loss" in entry for entry in log) and any("eval_loss" in entry for entry in log)
    assert summary["log"]["steps"] == 4 and summary["log"]["overfitting"] is False


def test_bad_training_data_aborts_before_any_training(tmp_path, rows, tokenizer):
    bad_rows = copy.deepcopy(rows)
    target = next(row for row in bad_rows if row["stage"] == "implement" and "tool_calls" in row["completion"][0])
    function = target["completion"][0]["tool_calls"][0]["function"]
    function["arguments"] = json.dumps(function["arguments"])  # the double-encoding mistake
    with pytest.raises(SystemExit, match="failed validation"):
        run_training(config_for(tmp_path, bad_rows), tiny_model(tokenizer), tokenizer)
    assert not (tmp_path / "out" / "train_summary.json").exists()
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    run_training(config_for(override_dir, bad_rows), tiny_model(tokenizer), tokenizer,
                 skip_validation=True)  # the explicit override still trains
