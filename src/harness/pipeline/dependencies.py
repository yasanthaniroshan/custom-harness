"""Everything a pipeline graph needs from the outside, bundled so nodes take one argument."""
from __future__ import annotations

from dataclasses import dataclass

from harness.config import Config
from harness.monitoring import NULL_TRACER, Tracer
from harness.pipeline.chat_models import ChatModelFactory
from harness.prompt_context import PromptBuilder


@dataclass
class PipelineDependencies:
    config: Config
    chat_models: ChatModelFactory
    prompt_builder: PromptBuilder
    tracer: Tracer = NULL_TRACER
    prose_temperature: float = 0.7  # requirements, design, review
    code_temperature: float = 0.2  # write_tests, implement
    review_only_passed: bool = True  # skip the review stage when the tests failed
