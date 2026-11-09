"""A sandboxed file workspace: path jail, hidden paths, change tracking."""
from __future__ import annotations

import difflib
import fnmatch
import os
from collections.abc import Iterator
from pathlib import Path

MAX_FILE_BYTES = 1_000_000
SKIPPED_DIRECTORIES = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".ruff_cache"}


class ToolError(Exception):
    """An expected failure the model can fix. The message goes back to the model verbatim."""


class Workspace:
    """All tool file access goes through here.

    - Paths must be relative and resolve (after symlinks) inside `root`.
    - Paths matching `hidden_globs` do not exist as far as tools can tell, so hidden tests
      stay hidden from every search and edit tool.
    - The first write to a file records its original content, enabling diff() and undo().
    """

    def __init__(self, root: str | Path, hidden_globs: tuple[str, ...] | list[str] = ()):
        self.root = Path(root).resolve()
        self.hidden_globs = tuple(hidden_globs)
        self._original_contents: dict[str, str | None] = {}  # relative path -> text (None = new file)

    # ---- paths ----
    def is_hidden(self, relative_path: str) -> bool:
        return any(fnmatch.fnmatch(relative_path, pattern) for pattern in self.hidden_globs)

    def relative_path(self, absolute_path: Path) -> str:
        return absolute_path.relative_to(self.root).as_posix()

    def resolve(self, relative_path: str, must_exist: bool = False) -> Path:
        if relative_path is None or str(relative_path).strip() == "":
            raise ToolError("path is required")
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ToolError("Use a path relative to the workspace root, not an absolute path.")
        absolute_path = (self.root / requested).resolve()
        if absolute_path != self.root and not absolute_path.is_relative_to(self.root):
            raise ToolError(f"Path escapes the workspace: {relative_path}")
        if absolute_path != self.root and self.is_hidden(self.relative_path(absolute_path)):
            raise ToolError(f"No such file or directory: {relative_path}")
        if must_exist and not absolute_path.exists():
            raise ToolError(f"No such file or directory: {relative_path}")
        return absolute_path

    def iter_files(self, subdirectory: str = ".") -> Iterator[str]:
        """Relative posix paths of visible files under `subdirectory`, sorted; symlinks skipped."""
        start = self.resolve(subdirectory, must_exist=True)
        if start.is_file():
            yield self.relative_path(start)
            return
        for directory, child_directories, file_names in os.walk(start):
            child_directories[:] = sorted(name for name in child_directories
                                          if name not in SKIPPED_DIRECTORIES)
            for file_name in sorted(file_names):
                absolute_path = Path(directory) / file_name
                relative_path = self.relative_path(absolute_path)
                if not self.is_hidden(relative_path) and not absolute_path.is_symlink():
                    yield relative_path

    # ---- content ----
    def read(self, relative_path: str) -> str:
        absolute_path = self.resolve(relative_path, must_exist=True)
        if absolute_path.is_dir():
            raise ToolError(f"{relative_path} is a directory; use glob or repo_map to list files.")
        size = absolute_path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ToolError(f"{relative_path} is too large to read ({size} bytes).")
        try:
            return absolute_path.read_text()
        except UnicodeDecodeError as error:
            raise ToolError(f"{relative_path} is not a UTF-8 text file.") from error

    def exists(self, relative_path: str) -> bool:
        return self.resolve(relative_path).exists()

    def write(self, relative_path: str, content: str) -> None:
        absolute_path = self.resolve(relative_path)
        if relative_path not in self._original_contents:
            self._original_contents[relative_path] = (
                absolute_path.read_text() if absolute_path.exists() else None)
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text(content)

    # ---- change tracking ----
    def changed_files(self) -> list[str]:
        return sorted(path for path, original in self._original_contents.items()
                      if self._current_content(path) != original)

    def _current_content(self, relative_path: str) -> str | None:
        absolute_path = self.root / relative_path
        return absolute_path.read_text() if absolute_path.exists() else None

    def diff(self, relative_path: str | None = None, context_lines: int = 3) -> str:
        """Unified diff of the changes since each file was first written ('' if none)."""
        paths = [relative_path] if relative_path else self.changed_files()
        diff_lines = []
        for path in paths:
            if path not in self._original_contents:
                continue
            original, current = self._original_contents[path], self._current_content(path)
            if original == current:
                continue
            diff_lines.extend(difflib.unified_diff(
                (original or "").splitlines(keepends=True), (current or "").splitlines(keepends=True),
                fromfile=f"a/{path}" if original is not None else "/dev/null",
                tofile=f"b/{path}", n=context_lines))
        text = "".join(diff_lines)
        return text if text.endswith("\n") or not text else text + "\n"

    def undo(self, relative_path: str) -> None:
        if relative_path not in self._original_contents:
            raise ToolError(f"{relative_path} has not been modified.")
        original = self._original_contents.pop(relative_path)
        absolute_path = self.resolve(relative_path)
        if original is None:
            absolute_path.unlink(missing_ok=True)
        else:
            absolute_path.write_text(original)
