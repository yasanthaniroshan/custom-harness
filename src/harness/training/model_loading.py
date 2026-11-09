"""Load the base model in 4-bit (QLoRA) and its tokenizer."""
from __future__ import annotations

from harness.config import ModelConfig


def load_model_and_tokenizer(config: ModelConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = getattr(torch, config.compute_dtype)
    quantization = None
    if config.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization_type,
            bnb_4bit_use_double_quant=config.double_quantization,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    tokenizer = AutoTokenizer.from_pretrained(config.path)
    model = AutoModelForCausalLM.from_pretrained(
        config.path,
        quantization_config=quantization,
        device_map={"": config.training_gpu_index},  # the whole model on one GPU, not "auto"
        dtype=compute_dtype,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tokenizer
