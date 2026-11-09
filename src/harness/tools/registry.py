"""Tool registry: typed parameters, argument validation, loop detection, result truncation
and tracing. The LangChain adapters (harness.pipeline.tool_adapters) expose these tools to the
model; every call still goes through ToolRegistry.call.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from harness.config import ToolsConfig
from harness.monitoring import NULL_TRACER, Tracer
from harness.prompt_context import TokenCounter, approximate_token_count, truncate_text
from harness.tools.workspace import ToolError

PARAMETER_TYPES: dict[str, type] = {"string": str, "integer": int, "boolean": bool, "array": list}


@dataclass
class ToolParameter:
    name: str
    type: str  # string | integer | boolean | array
    description: str
    required: bool = True


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter]
    function: Callable[..., str]


@dataclass
class ToolResult:
    ok: bool
    output: str


def _coerce_argument(value: Any, parameter_type: str) -> Any:
    """Forgive the usual small-model slips: "5" for 5, "true" for true."""
    if parameter_type == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    if parameter_type == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


class ToolRegistry:
    def __init__(self, tools: list[Tool], config: ToolsConfig | None = None,
                 tracer: Tracer = NULL_TRACER, count_tokens: TokenCounter = approximate_token_count,
                 allowed_tool_names: set[str] | None = None):
        self.config = config or ToolsConfig()
        self.tracer = tracer
        self.count_tokens = count_tokens
        self.tools = {tool.name: tool for tool in tools
                      if allowed_tool_names is None or tool.name in allowed_tool_names}
        self._previous_call: tuple[str, str] | None = None
        self._repeat_count = 0

    def _validate_arguments(self, tool: Tool, arguments: dict) -> dict:
        valid_names = {parameter.name for parameter in tool.parameters}
        unknown_names = set(arguments) - valid_names
        if unknown_names:
            raise ToolError(f"Unknown argument(s) {sorted(unknown_names)} for {tool.name}. "
                            f"Valid: {sorted(valid_names)}.")
        validated = {}
        for parameter in tool.parameters:
            if parameter.name not in arguments:
                if parameter.required:
                    raise ToolError(f"Missing required argument '{parameter.name}' for {tool.name}.")
                continue
            value = _coerce_argument(arguments[parameter.name], parameter.type)
            is_bool_posing_as_int = parameter.type == "integer" and isinstance(value, bool)
            if not isinstance(value, PARAMETER_TYPES[parameter.type]) or is_bool_posing_as_int:
                raise ToolError(f"Argument '{parameter.name}' must be {parameter.type}, "
                                f"got {type(value).__name__}.")
            validated[parameter.name] = value
        return validated

    def call(self, tool_name: str, arguments: dict) -> ToolResult:
        started = time.perf_counter()
        call_key = (tool_name, json.dumps(arguments, sort_keys=True, default=str))
        try:
            if tool_name not in self.tools:
                raise ToolError(f"Unknown tool '{tool_name}'. Available: {sorted(self.tools)}.")
            self._repeat_count = self._repeat_count + 1 if call_key == self._previous_call else 1
            self._previous_call = call_key
            if self._repeat_count > self.config.max_repeat_calls:
                raise ToolError("You already made this exact call and got the result. "
                                "Change the arguments or try a different tool.")
            tool = self.tools[tool_name]
            result = ToolResult(True, tool.function(**self._validate_arguments(tool, arguments)))
        except ToolError as error:
            result = ToolResult(False, str(error))
        except Exception as error:  # noqa: BLE001 - a tool bug must never crash the agent loop
            result = ToolResult(False, f"Internal tool error: {type(error).__name__}: {error}")
        result.output = truncate_text(result.output, self.config.max_result_tokens, self.count_tokens, "middle")
        self.tracer.emit(
            "tool_call", tool=tool_name, ok=result.ok,
            error=None if result.ok else result.output[:100],
            output_tokens=self.count_tokens(result.output),
            seconds=round(time.perf_counter() - started, 4),
        )
        return result
