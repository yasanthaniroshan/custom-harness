"""Expose the workspace tools as LangChain StructuredTools for bind_tools / ToolNode.

The adapters are thin: ToolRegistry still does path-jail checks, argument validation, loop
detection, result truncation and tracing. LangChain provides the JSON schema the model sees,
the tool-call protocol and ToolNode.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from langgraph.prebuilt.tool_node import ToolInvocationError
from pydantic import BaseModel, Field, ValidationError, create_model

from harness.monitoring import stage_scope
from harness.tools import Tool, ToolRegistry

PYTHON_TYPE_BY_PARAMETER_TYPE: dict[str, Any] = {
    "string": str, "integer": int, "boolean": bool, "array": list[dict],
}


def arguments_schema(tool: Tool) -> type[BaseModel]:
    """Pydantic schema built from the tool's parameters (this is what the model sees)."""
    schema_fields: dict[str, Any] = {}
    for parameter in tool.parameters:
        python_type = PYTHON_TYPE_BY_PARAMETER_TYPE[parameter.type]
        description = parameter.description or parameter.name
        if parameter.required:
            schema_fields[parameter.name] = (python_type, Field(description=description))
        else:
            schema_fields[parameter.name] = (python_type | None, Field(default=None, description=description))
    return create_model(f"{tool.name}_arguments", **schema_fields)


def to_langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    def make_runner(tool_name: str):
        def run(**arguments: Any) -> str:
            provided = {name: value for name, value in arguments.items() if value is not None}
            with stage_scope("implement"):
                result = registry.call(tool_name, provided)
            if not result.ok:
                raise ToolException(result.output)
            return result.output

        return run

    return [
        StructuredTool(name=tool.name, description=tool.description,
                       args_schema=arguments_schema(tool), func=make_runner(tool.name))
        for tool in registry.tools.values()
    ]


def tool_error_message(error: Exception) -> str:
    """The error text the model sees: our own message verbatim, a short one for bad arguments."""
    if isinstance(error, ToolInvocationError):  # ToolNode wraps bad arguments; unwrap them
        error = error.source
    if isinstance(error, ValidationError):
        first_problem = error.errors()[0]
        location = ".".join(str(part) for part in first_problem["loc"])
        return f"Invalid arguments: {location}: {first_problem['msg']}"
    return getattr(error, "message", None) or str(error)
