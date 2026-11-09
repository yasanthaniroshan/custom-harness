"""Chat models for the pipeline: ChatOpenAI pointed at the vLLM OpenAI-compatible server."""
from __future__ import annotations

import threading
from collections.abc import Callable

from harness.config import Config

ADAPTER_MODEL_NAME = "adapter"  # the name vLLM serves a LoRA adapter under (see serving.py)


def served_model_name(config: Config, adapter_path: str | None) -> str:
    return ADAPTER_MODEL_NAME if adapter_path else config.serve.served_model_name


class ChatModelFactory:
    """One chat model per temperature.

    Temperature is a per-stage setting, and binding it after with_structured_output/bind_tools
    is unreliable, so each temperature gets its own model instance. `build_model` can be
    replaced (tests pass a fake).
    """

    def __init__(self, config: Config, adapter_path: str | None = None,
                 build_model: Callable[[float], object] | None = None):
        self.config = config
        self.adapter_path = adapter_path
        self._build_model = build_model or self._build_openai_model
        self._models_by_temperature: dict[float, object] = {}
        self._lock = threading.Lock()

    def _build_openai_model(self, temperature: float):
        from langchain_openai import ChatOpenAI

        serve, sampling = self.config.serve, self.config.sampling
        return ChatOpenAI(
            model=served_model_name(self.config, self.adapter_path),
            base_url=serve.base_url,
            api_key="EMPTY",  # vLLM does not check it
            temperature=temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_new_tokens,
            timeout=serve.request_timeout_seconds,
            max_retries=serve.max_retries,
        )

    def get(self, temperature: float):
        with self._lock:
            if temperature not in self._models_by_temperature:
                self._models_by_temperature[temperature] = self._build_model(temperature)
            return self._models_by_temperature[temperature]
