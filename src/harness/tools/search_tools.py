"""Read-only tools: read_file, glob, grep, repo_map, find_definition."""
from __future__ import annotations

import ast
import fnmatch
import re

from harness.config import ToolsConfig
from harness.tools.registry import Tool, ToolParameter
from harness.tools.workspace import MAX_FILE_BYTES, ToolError, Workspace

MAX_LINE_CHARS = 200
MAX_REPO_MAP_FILES = 200
MAX_GREP_CONTEXT_LINES = 5


def _shorten_line(line: str) -> str:
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + " …"


def _python_files(workspace: Workspace, subdirectory: str) -> list[str]:
    return [path for path in workspace.iter_files(subdirectory) if path.endswith(".py")]


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{keyword} {node.name}({ast.unparse(node.args)}){returns}"


def _docstring_hint(node: ast.AST) -> str:
    docstring = ast.get_docstring(node)
    return f"  # {docstring.splitlines()[0][:80]}" if docstring else ""


def make_search_tools(workspace: Workspace, config: ToolsConfig) -> list[Tool]:
    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        lines = workspace.read(path).splitlines()
        total_lines = len(lines)
        if start_line < 1:
            raise ToolError("start_line must be >= 1.")
        last_line = min(end_line or total_lines, total_lines, start_line + config.max_read_lines - 1)
        if start_line > max(total_lines, 1):
            raise ToolError(f"{path} has only {total_lines} lines.")
        body = "\n".join(f"{number:>5}\t{_shorten_line(lines[number - 1])}"
                         for number in range(start_line, last_line + 1))
        more = ""
        if last_line < total_lines:
            more = f"\n[{total_lines - last_line} more lines; call again with start_line={last_line + 1}]"
        return f"{path} (lines {start_line}-{last_line} of {total_lines})\n{body}{more}"

    def glob(pattern: str, max_results: int | None = None) -> str:
        limit = min(max_results or config.max_glob_results, config.max_glob_results)

        def matches(path: str) -> bool:
            file_name = path.rsplit("/", 1)[-1]
            return fnmatch.fnmatch(path, pattern) or ("/" not in pattern and fnmatch.fnmatch(file_name, pattern))

        hits = [path for path in workspace.iter_files(".") if matches(path)]
        if not hits:
            return f"No files match '{pattern}'."
        shown = "\n".join(hits[:limit])
        return shown + (f"\n[{len(hits) - limit} more not shown]" if len(hits) > limit else "")

    def grep(pattern: str, path: str = ".", file_glob: str | None = None,
             ignore_case: bool = False, context_lines: int = 0) -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as error:
            raise ToolError(f"Invalid regex: {error}") from error
        context_lines = max(0, min(context_lines, MAX_GREP_CONTEXT_LINES))
        output_lines: list[str] = []
        match_count = 0
        hit_limit = False
        for relative_path in workspace.iter_files(path):
            if file_glob and not fnmatch.fnmatch(relative_path.rsplit("/", 1)[-1], file_glob):
                continue
            absolute_path = workspace.root / relative_path
            if absolute_path.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                lines = absolute_path.read_text().splitlines()
            except UnicodeDecodeError:
                continue  # binary file
            for line_index, line in enumerate(lines):
                if not regex.search(line):
                    continue
                if match_count >= config.max_grep_matches:
                    hit_limit = True
                    break
                match_count += 1
                first = max(0, line_index - context_lines)
                last = min(len(lines), line_index + context_lines + 1)
                for shown_index in range(first, last):
                    separator = ":" if shown_index == line_index else "-"
                    output_lines.append(f"{relative_path}{separator}{shown_index + 1}{separator} "
                                        f"{_shorten_line(lines[shown_index])}")
                if context_lines:
                    output_lines.append("--")
            if hit_limit:
                break
        if not output_lines:
            return f"No matches for '{pattern}'."
        note = (f"\n[stopped at {config.max_grep_matches} matches; narrow the pattern or path]"
                if hit_limit else "")
        return "\n".join(output_lines) + note

    def repo_map(path: str = ".") -> str:
        """Python signatures only: far cheaper in context than reading whole files."""
        output_lines = []
        for relative_path in _python_files(workspace, path)[:MAX_REPO_MAP_FILES]:
            try:
                tree = ast.parse(workspace.read(relative_path))
            except SyntaxError:
                output_lines.append(f"{relative_path}\n  (syntax error)")
                continue
            output_lines.append(relative_path)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    bases = f"({', '.join(ast.unparse(base) for base in node.bases)})" if node.bases else ""
                    output_lines.append(f"  class {node.name}{bases}  [line {node.lineno}]{_docstring_hint(node)}")
                    output_lines += [
                        f"    {_function_signature(method)}  [line {method.lineno}]{_docstring_hint(method)}"
                        for method in node.body
                        if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef)]
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    output_lines.append(
                        f"  {_function_signature(node)}  [line {node.lineno}]{_docstring_hint(node)}")
        return "\n".join(output_lines) or "No Python files found."

    def find_definition(name: str) -> str:
        hits = []
        for relative_path in _python_files(workspace, "."):
            try:
                tree = ast.parse(workspace.read(relative_path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                is_definition = isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                if is_definition and node.name == name:
                    description = f"class {node.name}" if isinstance(node, ast.ClassDef) else _function_signature(node)
                    hits.append(f"{relative_path}:{node.lineno}: {description}")
        return "\n".join(hits) or f"No definition of '{name}' found."

    return [
        Tool("read_file", "Read a file with line numbers. Use start_line/end_line for large files.",
             [ToolParameter("path", "string", "file path"),
              ToolParameter("start_line", "integer", "first line (1-based)", required=False),
              ToolParameter("end_line", "integer", "last line", required=False)], read_file),
        Tool("glob", "List files matching a glob pattern, e.g. '*.py' or 'src/**/*.py'.",
             [ToolParameter("pattern", "string", "glob pattern"),
              ToolParameter("max_results", "integer", "cap", required=False)], glob),
        Tool("grep", "Regex search in files. Returns path:line: text.",
             [ToolParameter("pattern", "string", "regex"),
              ToolParameter("path", "string", "file or dir", required=False),
              ToolParameter("file_glob", "string", "e.g. '*.py'", required=False),
              ToolParameter("ignore_case", "boolean", "", required=False),
              ToolParameter("context_lines", "integer", "lines around each match (max 5)", required=False)],
             grep),
        Tool("repo_map", "Outline of Python classes and function signatures (no bodies).",
             [ToolParameter("path", "string", "dir or file", required=False)], repo_map),
        Tool("find_definition", "Find where a Python function or class is defined.",
             [ToolParameter("name", "string", "symbol name")], find_definition),
    ]
