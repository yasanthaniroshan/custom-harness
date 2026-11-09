"""Read a train_log.jsonl (written during training) without needing the training stack."""
from __future__ import annotations

import json
from pathlib import Path

TRAIN_LOG_FILE = "train_log.jsonl"
OVERFITTING_TOLERANCE = 1.02  # eval loss this much above its best counts as getting worse
MIN_EVALS_FOR_OVERFITTING = 3


def summarize_train_log(path: str | Path) -> dict:
    """Condense a train_log.jsonl: the loss trajectory and an overfitting flag."""
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    train_entries = [entry for entry in entries if "loss" in entry]
    eval_entries = [entry for entry in entries if "eval_loss" in entry]
    summary: dict = {"log_entries": len(entries), "steps": max((entry["step"] for entry in entries), default=0)}
    if train_entries:
        summary |= {"first_loss": train_entries[0]["loss"], "last_loss": train_entries[-1]["loss"],
                    "min_loss": min(entry["loss"] for entry in train_entries)}
    if eval_entries:
        best = min(eval_entries, key=lambda entry: entry["eval_loss"])
        summary |= {"first_eval_loss": eval_entries[0]["eval_loss"],
                    "last_eval_loss": eval_entries[-1]["eval_loss"],
                    "best_eval_loss": best["eval_loss"], "best_eval_step": best["step"]}
        # overfitting: validation got worse after its best point while training kept improving
        summary["overfitting"] = bool(
            len(eval_entries) >= MIN_EVALS_FOR_OVERFITTING
            and eval_entries[-1]["eval_loss"] > best["eval_loss"] * OVERFITTING_TOLERANCE
            and train_entries and train_entries[-1]["loss"] < train_entries[0]["loss"])
    return summary
