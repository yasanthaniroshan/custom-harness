"""Process environment setup: Hugging Face auth, GPU pinning, LangSmith tracing.

Secrets come from environment variables first and Kaggle Secrets second. They are only placed
in the process environment and are never printed or logged.
"""
from __future__ import annotations

import os

from harness.config import Config


def read_kaggle_secret(name: str) -> str | None:
    """A Kaggle notebook secret, or None when not on Kaggle or the secret is not attached."""
    try:
        from kaggle_secrets import UserSecretsClient  # only importable on Kaggle
    except ImportError:
        return None
    try:
        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001 - Kaggle raises a generic error for a missing secret
        return None


def setup_huggingface_environment() -> None:
    """Set HF_TOKEN if available and enable the fast download backends."""
    if not os.environ.get("HF_TOKEN"):
        token = read_kaggle_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def pin_single_gpu(gpu_index: int = 0) -> None:
    """Expose only one GPU to this process. Must run before torch initializes CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def setup_langsmith(config: Config) -> bool:
    """Turn LangSmith tracing on if it is enabled in the config and a key is available.

    Returns whether tracing is on. While on, prompts, generated code and outputs are sent to
    LangSmith (hide them with langsmith.hide_inputs / hide_outputs).
    """
    settings = config.langsmith
    if not settings.enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    if not os.environ.get("LANGSMITH_API_KEY"):
        api_key = read_kaggle_secret("LANGSMITH_API_KEY")
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LangSmith is enabled in the config but LANGSMITH_API_KEY is not set; tracing is OFF.")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.project
    os.environ["LANGSMITH_HIDE_INPUTS"] = str(settings.hide_inputs).lower()
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = str(settings.hide_outputs).lower()
    return True
