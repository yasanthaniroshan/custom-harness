"""Workspace tools for the implement agent: search, edit, diff.

    workspace = Workspace(root, hidden_globs=["tests/*"])
    registry = build_tool_registry(workspace, config.tools, tracer, count_tokens)
    result = registry.call("grep", {"pattern": "def main"})   # ToolResult(ok, output)
"""
from __future__ import annotations

from harness.config import ToolsConfig
from harness.monitoring import NULL_TRACER, Tracer
from harness.prompt_context import TokenCounter, approximate_token_count
from harness.tools.edit_tools import make_edit_tools
from harness.tools.registry import Tool, ToolParameter, ToolRegistry, ToolResult
from harness.tools.search_tools import make_search_tools
from harness.tools.workspace import ToolError, Workspace

READ_ONLY_TOOL_NAMES = {"read_file", "glob", "grep", "repo_map", "find_definition"}


def build_tool_registry(workspace: Workspace, config: ToolsConfig | None = None,
                        tracer: Tracer = NULL_TRACER,
                        count_tokens: TokenCounter = approximate_token_count,
                        allowed_tool_names: set[str] | None = None,
                        extra_tools: list[Tool] | None = None) -> ToolRegistry:
    """Every search and edit tool plus `extra_tools`, optionally limited to an allowlist."""
    config = config or ToolsConfig()
    tools = make_search_tools(workspace, config) + make_edit_tools(workspace, config) + list(extra_tools or [])
    return ToolRegistry(tools, config, tracer, count_tokens, allowed_tool_names)


__all__ = [
    "READ_ONLY_TOOL_NAMES", "Tool", "ToolError", "ToolParameter", "ToolRegistry", "ToolResult",
    "Workspace", "build_tool_registry",
]
