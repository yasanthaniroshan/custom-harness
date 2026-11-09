"""QLoRA fine-tuning (needs the `train` extra: torch, transformers, trl, peft, ...).

    model_loading.py   4-bit base model + tokenizer
    trainer.py         SFTTrainer setup, training run, per-stage metrics, train-log summary
"""
