"""Editing tools: str_replace, multi_edit, apply_patch, write_file, diff, undo.

Every edit is atomic and validated: Python files must still parse, otherwise the edit is
rejected and the file is untouched. Successful edits return a compact diff so the model
sees what changed without re-reading the file.
"""
from __future__ import annotations

import ast
import difflib

from harness.config import ToolsConfig
from harness.tools.patch import apply_file_patch, parse_patch
from harness.tools.registry import Tool, ToolParameter
from harness.tools.workspace import ToolError, Workspace

MAX_REPORTED_MATCH_LINES = 5


def check_python_syntax(path: str, content: str) -> None:
    if not path.endswith(".py"):
        return
    try:
        ast.parse(content)
    except SyntaxError as error:
        raise ToolError(f"Edit rejected: it would cause a SyntaxError in {path} at line "
                        f"{error.lineno}: {error.msg}. The file was not changed.") from error


def _line_number_at(content: str, character_index: int) -> int:
    return content.count("\n", 0, character_index) + 1


def _hint_for_missing_text(content: str, old_text: str) -> str:
    """Help the model recover from a failed exact match."""
    wanted_lines = [line.strip() for line in old_text.strip("\n").splitlines()]
    file_lines = content.splitlines()
    if wanted_lines:
        block_length = len(wanted_lines)
        for start in range(len(file_lines) - block_length + 1):
            if [line.strip() for line in file_lines[start:start + block_length]] == wanted_lines:
                return (f" A block matching ignoring indentation/whitespace exists at lines "
                        f"{start + 1}-{start + block_length}; use read_file and copy it exactly.")
        stripped_lines = [line.strip() for line in file_lines]
        close_matches = difflib.get_close_matches(wanted_lines[0], stripped_lines, n=1, cutoff=0.6)
        if close_matches:
            line_number = stripped_lines.index(close_matches[0]) + 1
            return f" Closest line: {line_number}: {file_lines[line_number - 1].strip()[:100]!r}."
    return " Use read_file or grep to find the exact text."


def _replace_text(path: str, content: str, old_text: str, new_text: str,
                  replace_all: bool) -> tuple[str, int]:
    """Returns (new content, number of replacements); raises ToolError on no/ambiguous match."""
    if old_text == "":
        raise ToolError("old_string must not be empty (use write_file to create files).")
    if old_text == new_text:
        raise ToolError("old_string and new_string are identical.")
    occurrences = content.count(old_text)
    if occurrences == 0:
        raise ToolError(f"old_string not found in {path}." + _hint_for_missing_text(content, old_text))
    if occurrences > 1 and not replace_all:
        line_numbers, search_from = [], 0
        for _ in range(min(occurrences, MAX_REPORTED_MATCH_LINES)):
            search_from = content.index(old_text, search_from)
            line_numbers.append(_line_number_at(content, search_from))
            search_from += 1
        raise ToolError(f"old_string matches {occurrences} places in {path} (lines {line_numbers}). "
                        "Add surrounding context to make it unique, or set replace_all=true.")
    return content.replace(old_text, new_text), occurrences


def make_edit_tools(workspace: Workspace, config: ToolsConfig) -> list[Tool]:
    def commit_changes(new_contents: dict[str, str]) -> str:
        for path, content in new_contents.items():  # validate everything before writing anything
            check_python_syntax(path, content)
        for path, content in new_contents.items():
            workspace.write(path, content)
        return "".join(workspace.diff(path) for path in new_contents)

    def str_replace(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        content = workspace.read(path)
        new_content, replacements = _replace_text(path, content, old_string, new_string, replace_all)
        diff = commit_changes({path: new_content})
        plural = "s" if replacements != 1 else ""
        return f"Edited {path} ({replacements} replacement{plural}).\n{diff}"

    def multi_edit(path: str, edits: list) -> str:
        content = workspace.read(path)
        if not edits:
            raise ToolError("edits must contain at least one {old_string, new_string} object.")
        for edit_number, edit in enumerate(edits, 1):
            if not isinstance(edit, dict) or "old_string" not in edit or "new_string" not in edit:
                raise ToolError(f"Edit {edit_number} must be an object with old_string and new_string.")
            try:
                content, _ = _replace_text(path, content, edit["old_string"], edit["new_string"],
                                           bool(edit.get("replace_all", False)))
            except ToolError as error:
                raise ToolError(f"Edit {edit_number} of {len(edits)} failed: {error} "
                                "No edits were applied.") from error
        return f"Applied {len(edits)} edits to {path}.\n{commit_changes({path: content})}"

    def apply_patch(patch: str) -> str:
        new_contents: dict[str, str] = {}
        for file_patch in parse_patch(patch):
            if file_patch.path in new_contents:  # several hunk groups for one file in one patch
                current_content = new_contents[file_patch.path]
            else:
                exists = workspace.exists(file_patch.path)
                if file_patch.is_new_file and exists:
                    raise ToolError(f"{file_patch.path} already exists; use hunks against its current content.")
                current_content = workspace.read(file_patch.path) if exists else None
            new_contents[file_patch.path] = apply_file_patch(current_content, file_patch)
        return f"Patched {', '.join(new_contents)}.\n{commit_changes(new_contents)}"

    def write_file(path: str, content: str, overwrite: bool = False) -> str:
        if workspace.exists(path) and not overwrite:
            raise ToolError(f"{path} already exists. Edit it with str_replace, or pass overwrite=true.")
        diff = commit_changes({path: content})
        return f"Wrote {path} ({len(content.splitlines())} lines).\n{diff}"

    def diff(path: str | None = None) -> str:
        return workspace.diff(path) or "No changes."

    def undo(path: str) -> str:
        workspace.undo(path)
        return f"Restored {path} to its original content."

    return [
        Tool("str_replace", "Replace exact text in a file. old_string must match exactly once "
             "(include surrounding lines to make it unique). Returns the diff.",
             [ToolParameter("path", "string", ""),
              ToolParameter("old_string", "string", "exact text to replace"),
              ToolParameter("new_string", "string", "replacement"),
              ToolParameter("replace_all", "boolean", "", required=False)], str_replace),
        Tool("multi_edit", "Apply several str_replace edits to one file atomically (all or none).",
             [ToolParameter("path", "string", ""),
              ToolParameter("edits", "array", "list of {old_string, new_string}")], multi_edit),
        Tool("apply_patch", "Apply a unified diff (---/+++/@@ hunks). Hunks are located by content, "
             "so line numbers may be approximate.",
             [ToolParameter("patch", "string", "unified diff text")], apply_patch),
        Tool("write_file", "Create a new file (refuses to overwrite unless overwrite=true).",
             [ToolParameter("path", "string", ""), ToolParameter("content", "string", ""),
              ToolParameter("overwrite", "boolean", "", required=False)], write_file),
        Tool("diff", "Show all changes made so far (or for one file) as a unified diff.",
             [ToolParameter("path", "string", "", required=False)], diff),
        Tool("undo", "Revert a file to its original content.",
             [ToolParameter("path", "string", "")], undo),
    ]
