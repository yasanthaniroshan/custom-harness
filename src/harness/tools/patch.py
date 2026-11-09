"""Tolerant unified-diff application.

Models get line numbers in diffs wrong, so hunks are located by their *content* (removed and
context lines), first exactly and then with relaxed whitespace; line numbers only break ties.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from harness.tools.workspace import ToolError

_HUNK_HEADER = re.compile(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


@dataclass
class Hunk:
    old_lines: list[str]  # context + removed lines, as they appear in the file now
    new_lines: list[str]  # context + added lines, what replaces them
    line_hint: int  # 1-based start line from the header (only a tie-breaker)


@dataclass
class FilePatch:
    path: str
    is_new_file: bool
    hunks: list[Hunk] = field(default_factory=list)


def _strip_path_prefix(name: str) -> str:
    name = name.split("\t")[0].strip()
    return name[2:] if name[:2] in ("a/", "b/") else name


def _is_file_header(lines: list[str], index: int) -> bool:
    return (lines[index].startswith("--- ") and index + 1 < len(lines)
            and lines[index + 1].startswith("+++ "))


def parse_patch(text: str) -> list[FilePatch]:
    lines = text.strip("\n").splitlines()
    patches: list[FilePatch] = []
    index = 0
    while index < len(lines):
        if _is_file_header(lines, index):
            old_name = _strip_path_prefix(lines[index][4:])
            new_name = _strip_path_prefix(lines[index + 1][4:])
            if new_name == "/dev/null":
                raise ToolError("Deleting files is not supported by apply_patch.")
            patches.append(FilePatch(new_name, is_new_file=old_name == "/dev/null"))
            index += 2
        elif lines[index].startswith("@@"):
            if not patches:
                raise ToolError("Patch has a hunk before any '--- a/file' / '+++ b/file' header.")
            header = _HUNK_HEADER.match(lines[index])
            hunk = Hunk([], [], int(header.group(1)) if header else 1)
            index += 1
            while index < len(lines) and not lines[index].startswith("@@") and not _is_file_header(lines, index):
                line = lines[index]
                if line.startswith("+"):
                    hunk.new_lines.append(line[1:])
                elif line.startswith("-"):
                    hunk.old_lines.append(line[1:])
                elif line.startswith("\\"):
                    pass  # "\ No newline at end of file"
                else:  # a context line; models often drop its leading space
                    context = line.removeprefix(" ")
                    hunk.old_lines.append(context)
                    hunk.new_lines.append(context)
                index += 1
            patches[-1].hunks.append(hunk)
        else:
            index += 1  # headers such as "diff --git" or "index ..."
    if not patches or not any(patch.hunks for patch in patches):
        raise ToolError("No hunks found. Provide a unified diff with '--- a/path', '+++ b/path' "
                        "and '@@ ... @@' hunks.")
    return patches


# Tried in order: exact, trailing whitespace ignored, surrounding whitespace ignored, all
# whitespace runs collapsed.
_LINE_NORMALIZERS = (
    lambda line: line,
    str.rstrip,
    str.strip,
    lambda line: " ".join(line.split()),
)


def _locate_hunk(file_lines: list[str], old_lines: list[str], line_hint: int, path: str) -> int:
    """Index in file_lines where `old_lines` starts; raises ToolError if absent or ambiguous."""
    if not old_lines:  # a pure addition goes at the hinted line
        return min(max(line_hint - 1, 0), len(file_lines))
    for normalize in _LINE_NORMALIZERS:
        target = [normalize(line) for line in old_lines]
        positions = [
            position for position in range(len(file_lines) - len(old_lines) + 1)
            if [normalize(line) for line in file_lines[position:position + len(old_lines)]] == target
        ]
        if not positions:
            continue
        if len(positions) == 1:
            return positions[0]

        def distance_from_hint(position: int) -> int:
            return abs(position + 1 - line_hint)

        closest = min(positions, key=distance_from_hint)
        if sum(distance_from_hint(position) == distance_from_hint(closest) for position in positions) > 1:
            raise ToolError(f"Hunk matches {len(positions)} places in {path}; add more context lines.")
        return closest
    first_line = old_lines[0].strip()
    found_at = next((number + 1 for number, line in enumerate(file_lines)
                     if first_line and line.strip() == first_line), None)
    hint = (f" The first line does appear at line {found_at}; re-read the file and rebuild the hunk."
            if found_at else "")
    raise ToolError(f"Hunk does not match {path}.{hint}")


def apply_file_patch(content: str | None, file_patch: FilePatch) -> str:
    """Apply all hunks of one file to `content` (None for a new file); returns the new content."""
    if file_patch.is_new_file:
        if content is not None:
            raise ToolError(f"{file_patch.path} already exists; use hunks against its current content.")
        return "\n".join(line for hunk in file_patch.hunks for line in hunk.new_lines) + "\n"
    if content is None:
        raise ToolError(f"No such file: {file_patch.path}")
    lines = content.splitlines()
    line_offset = 0  # earlier hunks shift the line numbers of later ones
    for hunk in file_patch.hunks:
        position = _locate_hunk(lines, hunk.old_lines, hunk.line_hint + line_offset, file_patch.path)
        lines[position:position + len(hunk.old_lines)] = hunk.new_lines
        line_offset += len(hunk.new_lines) - len(hunk.old_lines)
    trailing_newline = "\n" if content.endswith("\n") or not content else ""
    return "\n".join(lines) + trailing_newline
