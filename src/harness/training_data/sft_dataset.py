"""Build the training and validation sets: self-generated rows (plus optional Magicoder rows),
split by task, rendered with the chat template, tokenized, overlong rows dropped."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.config import Config, DataConfig
from harness.training_data.chat_rendering import RenderError, render_row

HASH_BUCKETS = 10_000


def magicoder_to_row(example: dict) -> dict:
    """A Magicoder row in the conversational prompt/completion format."""
    return {
        "stage": "magicoder",
        "task_id": f"magicoder-{example['index']}",
        "prompt": [{"role": "user", "content": example["problem"]}],
        "completion": [{"role": "assistant", "content": example["solution"]}],
    }


def magicoder_count_for_mix(own_row_count: int, magicoder_share: float) -> int:
    """How many Magicoder rows make them `magicoder_share` of the final dataset."""
    if not 0 <= magicoder_share < 1:
        raise ValueError("magicoder_mix must be in [0, 1)")
    return round(own_row_count * magicoder_share / (1 - magicoder_share))


def load_rows(paths: list[str] | list[Path]) -> list[dict]:
    """SFT rows from JSONL files (the output of `harness filter`)."""
    rows: list[dict] = []
    for path in paths:
        rows += [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows


def is_validation_task(task_id: Any, validation_fraction: float, seed: int) -> bool:
    """Deterministic: a task is in validation if its hash falls in the lowest fraction."""
    digest = hashlib.sha1(f"{seed}:{task_id}".encode()).hexdigest()
    return int(digest, 16) % HASH_BUCKETS < validation_fraction * HASH_BUCKETS


def split_rows(rows: list[dict], validation_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split BY TASK: a task's requirements, design, tests and trajectory stay together, so the
    validation loss is never measured on a problem the model trained on in another stage."""
    if validation_fraction <= 0:
        return rows, []
    train_rows, validation_rows = [], []
    for row in rows:
        is_validation = is_validation_task(row["task_id"], validation_fraction, seed)
        (validation_rows if is_validation else train_rows).append(row)
    return train_rows, validation_rows


def load_magicoder_rows(config: DataConfig, count: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(config.dataset, split=config.split)
    if config.magicoder_language:
        dataset = dataset.filter(lambda example: example["lang"] == config.magicoder_language)
    dataset = dataset.shuffle(seed=config.seed).select(range(min(count, len(dataset))))
    return [magicoder_to_row(example) for example in dataset]


@dataclass
class TokenizedRows:
    input_ids: list[list[int]] = field(default_factory=list)
    completion_masks: list[list[int]] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)

    def to_dataset(self, stage: str | None = None):
        """A Hugging Face Dataset of all rows, or of one stage's rows."""
        from datasets import Dataset

        positions = [position for position, row_stage in enumerate(self.stages)
                     if stage is None or row_stage == stage]
        return Dataset.from_dict({
            "input_ids": [self.input_ids[position] for position in positions],
            "completion_mask": [self.completion_masks[position] for position in positions],
        })


def tokenize_rows(rows: list[dict], tokenizer, max_length: int, drop_overlong: bool = True
                  ) -> tuple[TokenizedRows, Counter]:
    tokenized, stats = TokenizedRows(), Counter()
    for row in rows:
        try:
            rendered = render_row(tokenizer, row)
        except RenderError:
            stats["dropped_unrenderable"] += 1
            continue
        if drop_overlong and rendered.token_count > max_length:
            stats["dropped_overlong"] += 1
            stats[f"dropped_overlong:{row['stage']}"] += 1
            continue
        tokenized.input_ids.append(rendered.input_ids)
        tokenized.completion_masks.append(rendered.completion_mask)
        tokenized.stages.append(row["stage"])
        stats["kept"] += 1
        stats[f"kept:{row['stage']}"] += 1
        stats["tokens"] += rendered.token_count
        stats["completion_tokens"] += rendered.completion_token_count
    return tokenized, stats


@dataclass
class PreparedData:
    train: Any  # datasets.Dataset
    validation: Any | None
    validation_by_stage: dict
    stats: dict


def prepare_training_data(config: Config, tokenizer) -> PreparedData:
    """Pre-tokenized train and validation sets, plus validation split per stage."""
    data_config, train_config = config.data, config.train
    own_rows = load_rows(data_config.sft_paths) if data_config.sft_paths else []
    if own_rows:
        magicoder_count = magicoder_count_for_mix(len(own_rows), data_config.magicoder_mix)
    else:
        magicoder_count = data_config.magicoder_samples
    rows = own_rows + (load_magicoder_rows(data_config, magicoder_count) if magicoder_count > 0 else [])
    train_rows, validation_rows = split_rows(rows, train_config.validation_fraction, train_config.seed)
    train_tokens, train_stats = tokenize_rows(train_rows, tokenizer, train_config.max_length,
                                              train_config.drop_overlong)
    validation_tokens, validation_stats = tokenize_rows(validation_rows, tokenizer, train_config.max_length,
                                                        train_config.drop_overlong)
    validation_by_stage = {stage: validation_tokens.to_dataset(stage)
                           for stage in sorted(set(validation_tokens.stages))}
    stats = {"train": dict(train_stats), "validation": dict(validation_stats),
             "own_rows": len(own_rows), "total_rows": len(rows)}
    validation = validation_tokens.to_dataset() if validation_tokens.stages else None
    return PreparedData(train_tokens.to_dataset(), validation, validation_by_stage, stats)
