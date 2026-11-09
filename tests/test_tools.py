import os

import pytest

from harness.config import ToolsConfig
from harness.monitoring import Tracer
from harness.tools import Workspace, build_tool_registry

CALCULATOR_SOURCE = '''"""Calculator."""


class Calc:
    """Does math."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def sub(self, a: int, b: int) -> int:
        return a - b


def helper(x):
    return x * 2
'''


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(CALCULATOR_SOURCE)
    (tmp_path / "src" / "util.py").write_text("def helper(x):\n    return x\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text("def test_secret():\n    assert 42 == 42\n")
    (tmp_path / "README.md").write_text("# Project\nsee calc\n")
    return Workspace(tmp_path, hidden_globs=["tests/*"])


@pytest.fixture
def registry(workspace):
    return build_tool_registry(workspace, ToolsConfig(max_repeat_calls=2), Tracer())


def call(registry, tool_name, **arguments):
    return registry.call(tool_name, arguments)


# ---- security ----------------------------------------------------------------------------

def test_path_jail_blocks_traversal_absolute_paths_and_symlink_escape(registry, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    (tmp_path / "link.txt").symlink_to(outside)
    for escaping_path in ("../outside.txt", "/etc/passwd", "link.txt", "src/../../outside.txt"):
        result = call(registry, "read_file", path=escaping_path)
        assert not result.ok and "secret" not in result.output, escaping_path
    assert not call(registry, "write_file", path="../evil.py", content="x = 1").ok
    assert not (tmp_path.parent / "evil.py").exists()


def test_hidden_paths_are_invisible_to_every_tool(registry):
    assert not call(registry, "read_file", path="tests/test_calc.py").ok
    assert "test_calc" not in call(registry, "glob", pattern="*.py").output
    assert call(registry, "grep", pattern="test_secret").output.startswith("No matches")
    assert "test_secret" not in call(registry, "repo_map").output
    assert not call(registry, "str_replace", path="tests/test_calc.py", old_string="42", new_string="43").ok
    assert call(registry, "find_definition", name="test_secret").output.startswith("No definition")


# ---- search ------------------------------------------------------------------------------

def test_read_file_ranges_and_pagination(workspace, registry):
    output = call(registry, "read_file", path="src/calc.py", start_line=4, end_line=6).output
    assert "(lines 4-6 of" in output and "class Calc:" in output and "def add" not in output
    small_pages = build_tool_registry(workspace, ToolsConfig(max_read_lines=3))
    output = small_pages.call("read_file", {"path": "src/calc.py"}).output
    assert "more lines; call again with start_line=4" in output


def test_glob_and_grep(registry):
    assert call(registry, "glob", pattern="*.py").output.splitlines() == ["src/calc.py", "src/util.py"]
    assert "No files match" in call(registry, "glob", pattern="*.rs").output
    grep_output = call(registry, "grep", pattern=r"def (add|sub)", path="src", context_lines=1).output
    assert "src/calc.py:7: " in grep_output and "src/calc.py-6- " in grep_output
    assert "src/calc.py:10: " in grep_output
    assert "No matches" in call(registry, "grep", pattern="nonexistent_zzz").output
    invalid = call(registry, "grep", pattern="(")
    assert not invalid.ok and "Invalid regex" in invalid.output
    single_file = call(registry, "grep", pattern="helper", ignore_case=True, file_glob="util.py").output
    assert single_file.count("\n") == 0


def test_grep_stops_at_the_match_cap(workspace):
    capped = build_tool_registry(workspace, ToolsConfig(max_grep_matches=2))
    assert "stopped at 2 matches" in capped.call("grep", {"pattern": "def "}).output


def test_repo_map_is_signatures_only_and_find_definition(registry):
    outline = call(registry, "repo_map").output
    assert "class Calc  [line 4]  # Does math." in outline
    assert "def add(self, a: int, b: int) -> int  [line 7]" in outline
    assert "return a + b" not in outline  # no bodies: that is the point
    definitions = call(registry, "find_definition", name="helper").output.splitlines()
    assert definitions == ["src/calc.py:14: def helper(x)", "src/util.py:1: def helper(x)"]


# ---- editing -----------------------------------------------------------------------------

def test_str_replace_success_returns_a_diff(workspace, registry):
    result = call(registry, "str_replace", path="src/calc.py", old_string="return a + b", new_string="return b + a")
    assert result.ok and "-        return a + b" in result.output and "+        return b + a" in result.output
    assert "return b + a" in workspace.read("src/calc.py")


def test_str_replace_requires_a_unique_match_unless_replace_all(workspace, registry):
    result = call(registry, "str_replace", path="src/calc.py", old_string="int", new_string="float")
    assert not result.ok and "matches" in result.output and "replace_all" in result.output
    assert workspace.read("src/calc.py") == CALCULATOR_SOURCE  # untouched
    result = call(registry, "str_replace", path="src/calc.py", old_string="int", new_string="float",
                  replace_all=True)
    assert result.ok and "float" in workspace.read("src/calc.py")


def test_str_replace_failure_hints(registry):
    result = call(registry, "str_replace", path="src/calc.py", old_string="def add(self,a:int,b:int)", new_string="x")
    assert not result.ok and "Closest line: 7" in result.output
    assert call(registry, "str_replace", path="src/calc.py", old_string="return a + b\n", new_string="y").ok
    assert not call(registry, "str_replace", path="src/calc.py", old_string="", new_string="x").ok
    identical = call(registry, "str_replace", path="src/calc.py", old_string="return a - b", new_string="return a - b")
    assert "identical" in identical.output


def test_whitespace_mismatch_hint_points_at_the_block(registry):
    wrongly_indented = "def add(self, a: int, b: int) -> int:\n  return a + b"  # body indented by 2, not 8
    result = call(registry, "str_replace", path="src/calc.py", old_string=wrongly_indented, new_string="x")
    assert not result.ok and "ignoring indentation" in result.output and "lines 7-8" in result.output


def test_an_edit_that_breaks_syntax_is_rejected_and_the_file_is_untouched(workspace, registry):
    result = call(registry, "str_replace", path="src/calc.py",
                  old_string="def add(self, a: int, b: int) -> int:",
                  new_string="def add(self, a: int, b: int) -> int")
    assert not result.ok and "SyntaxError" in result.output and "not changed" in result.output
    assert workspace.read("src/calc.py") == CALCULATOR_SOURCE and workspace.diff() == ""


def test_multi_edit_is_atomic(workspace, registry):
    both_valid = [{"old_string": "return a + b", "new_string": "return a + b + 0"},
                  {"old_string": "return a - b", "new_string": "return a - b - 0"}]
    assert call(registry, "multi_edit", path="src/calc.py", edits=both_valid).ok
    content = workspace.read("src/calc.py")
    assert "a + b + 0" in content and "a - b - 0" in content
    second_fails = [{"old_string": "a + b + 0", "new_string": "a + b"}, {"old_string": "NOPE", "new_string": "x"}]
    result = call(registry, "multi_edit", path="src/calc.py", edits=second_fails)
    assert not result.ok and "Edit 2 of 2 failed" in result.output and "No edits were applied" in result.output
    assert workspace.read("src/calc.py") == content  # the first edit was not applied either
    assert not call(registry, "multi_edit", path="src/calc.py", edits=["x"]).ok


def test_apply_patch_tolerates_wrong_line_numbers_and_whitespace_drift(workspace, registry):
    wrong_line_numbers = """--- a/src/calc.py
+++ b/src/calc.py
@@ -99,3 +99,3 @@
     def sub(self, a: int, b: int) -> int:
-        return a - b
+        return b - a
"""
    assert call(registry, "apply_patch", patch=wrong_line_numbers).ok
    assert "return b - a" in workspace.read("src/calc.py")  # line 99 was wrong; the content matched
    extra_space = "--- a/src/util.py\n+++ b/src/util.py\n@@ -1,2 +1,2 @@\n def helper(x):\n-    return  x\n+    return x + 1\n"
    assert call(registry, "apply_patch", patch=extra_space).ok  # whitespace-relaxed match
    assert "x + 1" in workspace.read("src/util.py")


def test_apply_patch_new_file_multiple_files_and_errors(workspace, registry):
    two_files = """--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,2 @@
+def new():
+    return 1
--- a/src/util.py
+++ b/src/util.py
@@ -1,2 +1,2 @@
 def helper(x):
-    return x
+    return -x
"""
    assert call(registry, "apply_patch", patch=two_files).ok
    assert workspace.read("src/new.py") == "def new():\n    return 1\n" and "return -x" in workspace.read("src/util.py")
    not_matching = "--- a/src/util.py\n+++ b/src/util.py\n@@ -1,1 +1,1 @@\n-def totally_different():\n+x\n"
    assert not call(registry, "apply_patch", patch=not_matching).ok
    deletion = "--- a/x.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    assert "Deleting files" in call(registry, "apply_patch", patch=deletion).output
    assert not call(registry, "apply_patch", patch="just some text").ok


def test_apply_patch_ambiguous_hunk_and_atomicity(workspace, registry):
    (workspace.root / "dup.py").write_text("x = 1\ny = 2\nx = 1\ny = 3\n")
    ambiguous = "--- a/dup.py\n+++ b/dup.py\n@@ -2,1 +2,1 @@\n-x = 1\n+x = 9\n"
    result = call(registry, "apply_patch", patch=ambiguous)  # the hint (line 2) is equidistant from both
    assert not result.ok and "matches 2 places" in result.output
    second_file_fails = (
        "--- a/src/util.py\n+++ b/src/util.py\n@@ -1,2 +1,2 @@\n def helper(x):\n-    return x\n+    return 0\n"
        "--- a/src/calc.py\n+++ b/src/calc.py\n@@ -1,1 +1,1 @@\n-NOT PRESENT\n+x\n")
    assert not call(registry, "apply_patch", patch=second_file_fails).ok
    assert "return x" in workspace.read("src/util.py")  # nothing written because the second file failed


def test_write_file_diff_and_undo(workspace, registry):
    assert call(registry, "write_file", path="pkg/mod.py", content="a = 1\n").ok
    assert (workspace.root / "pkg" / "mod.py").exists()
    assert not call(registry, "write_file", path="pkg/mod.py", content="a = 2\n").ok  # no silent overwrite
    assert call(registry, "write_file", path="pkg/mod.py", content="a = 2\n", overwrite=True).ok
    call(registry, "str_replace", path="src/util.py", old_string="return x", new_string="return x + 1")
    diff_output = call(registry, "diff").output
    assert "+++ b/pkg/mod.py" in diff_output and "+++ b/src/util.py" in diff_output
    assert "--- /dev/null" in diff_output
    assert call(registry, "undo", path="src/util.py").ok and workspace.read("src/util.py").endswith("return x\n")
    assert call(registry, "undo", path="pkg/mod.py").ok and not (workspace.root / "pkg" / "mod.py").exists()
    assert call(registry, "diff").output == "No changes."
    assert not call(registry, "undo", path="README.md").ok


# ---- registry ----------------------------------------------------------------------------

def test_argument_validation_coercion_unknown_tool_and_allowlist(workspace):
    restricted = build_tool_registry(workspace, ToolsConfig(), allowed_tool_names={"read_file", "grep"})
    assert set(restricted.tools) == {"read_file", "grep"}
    assert "Unknown tool 'str_replace'" in restricted.call("str_replace", {}).output
    assert "Missing required argument 'path'" in restricted.call("read_file", {}).output
    assert "Unknown argument(s) ['file']" in restricted.call("read_file", {"file": "x"}).output
    assert "must be integer" in restricted.call("read_file", {"path": "README.md", "start_line": "abc"}).output
    assert restricted.call("read_file", {"path": "README.md", "start_line": "1"}).ok  # "1" coerced to 1
    assert not restricted.call("read_file", {"path": "README.md", "start_line": True}).ok


def test_repeated_identical_calls_are_refused_and_results_truncated(workspace):
    registry = build_tool_registry(workspace, ToolsConfig(max_repeat_calls=2, max_result_tokens=30))
    results = [registry.call("glob", {"pattern": "*.py"}) for _ in range(3)]
    assert [result.ok for result in results] == [True, True, False] and "exact call" in results[2].output
    assert registry.call("glob", {"pattern": "*.md"}).ok  # a different call resets the counter
    assert "tokens truncated" in registry.call("read_file", {"path": "src/calc.py"}).output


def test_internal_tool_errors_do_not_crash(workspace):
    registry = build_tool_registry(workspace)
    registry.tools["glob"].function = lambda **arguments: 1 / 0
    result = registry.call("glob", {"pattern": "*"})
    assert not result.ok and "Internal tool error: ZeroDivisionError" in result.output


def test_tool_calls_are_traced(workspace):
    tracer = Tracer()
    registry = build_tool_registry(workspace, tracer=tracer)
    registry.call("grep", {"pattern": "def "})
    registry.call("read_file", {"path": "nope.py"})
    events = [event for event in tracer.events if event["kind"] == "tool_call"]
    assert [(event["tool"], event["ok"]) for event in events] == [("grep", True), ("read_file", False)]
    assert "No such file" in events[1]["error"]


def test_workspace_skips_version_control_directories_and_symlinks(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x")
    (tmp_path / "a.py").write_text("x = 1\n")
    os.symlink(tmp_path / "a.py", tmp_path / "alias.py")
    assert list(Workspace(tmp_path).iter_files()) == ["a.py"]
