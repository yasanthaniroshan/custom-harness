"""From pipeline runs to training rows.

    seed_tasks.py         seed tasks and the fixed held-out evaluation split
    trajectory_filter.py  which runs to trust, and run -> SFT rows
    validation.py         checks SFT rows before any GPU time is spent
    chat_rendering.py     render a row with the real chat template -> input_ids + completion mask
    sft_dataset.py        split by task, tokenize, build the train/validation datasets
"""
