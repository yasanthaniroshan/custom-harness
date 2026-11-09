"""Seed tasks for self-generation, with a fixed held-out evaluation split."""
from __future__ import annotations

import random
from collections.abc import Iterable

from harness.config import Config


def split_tasks(rows: Iterable[dict], eval_count: int, train_count: int, seed: int, language: str,
                max_chars: int) -> tuple[list[dict], list[dict]]:
    """Filter by language and length, shuffle deterministically, then take eval first, train next.

    The eval set depends only on (seed, eval_count), so growing `train_count` later never moves
    tasks between the two sets. Returns (train_tasks, eval_tasks).
    """
    candidates = [
        {"task_id": row["index"], "task": row["problem"]}
        for row in rows
        if row["lang"] == language and len(row["problem"]) <= max_chars
    ]
    candidates.sort(key=lambda task: task["task_id"])  # a stable order before shuffling
    random.Random(seed).shuffle(candidates)
    return candidates[eval_count:eval_count + train_count], candidates[:eval_count]


def load_seed_tasks(config: Config) -> tuple[list[dict], list[dict]]:
    """(train_tasks, eval_tasks) from the Magicoder `problem` column."""
    from datasets import load_dataset

    dataset = load_dataset(config.data.dataset, split=config.data.split)
    generate = config.generate
    return split_tasks(
        dataset.select_columns(["index", "lang", "problem"]),
        eval_count=generate.eval_tasks,
        train_count=generate.num_tasks,
        seed=config.data.seed,
        language=generate.seed_language,
        max_chars=generate.max_task_chars,
    )
