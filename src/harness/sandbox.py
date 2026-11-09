"""Run code against hidden tests in a subprocess and report sanitized results.

The report deliberately exposes only test names, pass/fail, and error types. It never includes
assertion source lines, expected values, or AssertionError messages, so the implementer can
see *whether* tests pass without being handed the answers.

NOTE: this is process isolation with a timeout, not a security sandbox.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

RESULT_MARKER = "@@HARNESS_RESULTS@@"

# Executed in the subprocess. Reads {"code" | "workspace_root", "tests"} as JSON on stdin and
# prints RESULT_MARKER followed by the JSON report.
_TEST_RUNNER_SOURCE = r'''
import importlib, json, os, sys

# Never cache bytecode: .pyc invalidation uses the modification time (to the second) and the
# file size, so a quick same-length edit (x * 2 -> x * 3) would otherwise run the OLD code.
sys.dont_write_bytecode = True

payload = json.loads(sys.stdin.read())
namespace = {"__name__": "generated"}
report = {"load_error": None, "tests_error": None, "cases": []}

def describe(error):
    # AssertionError text can reveal expected values; other errors come from the code itself.
    if isinstance(error, AssertionError):
        return "AssertionError"
    return f"{type(error).__name__}: {error}"

if "workspace_root" in payload:  # import every top-level module into the test namespace
    workspace_root = payload["workspace_root"]
    sys.path.insert(0, workspace_root)
    os.chdir(workspace_root)
    try:
        for file_name in sorted(os.listdir(workspace_root)):
            if file_name.endswith(".py"):
                module = importlib.import_module(file_name[:-3])
                namespace.update({name: value for name, value in vars(module).items()
                                  if not name.startswith("_")})
    except BaseException as error:
        report["load_error"] = describe(error)
else:
    try:
        exec(compile(payload["code"], "<code>", "exec"), namespace)
    except BaseException as error:
        report["load_error"] = describe(error)

if report["load_error"] is None:
    try:
        exec(compile(payload["tests"], "<tests>", "exec"), namespace)
    except BaseException as error:
        report["tests_error"] = describe(error)

if report["load_error"] is None and report["tests_error"] is None:
    test_names = [name for name, value in namespace.items()
                  if name.startswith("test_") and callable(value)]
    for test_name in test_names:
        try:
            namespace[test_name]()
            report["cases"].append({"name": test_name, "passed": True, "error": None})
        except BaseException as error:
            report["cases"].append({"name": test_name, "passed": False, "error": describe(error)})

sys.stdout.flush()
print("RESULT_MARKER" + json.dumps(report))
'''.replace("RESULT_MARKER", RESULT_MARKER)


@dataclass
class TestCaseResult:
    __test__ = False  # not a pytest class

    name: str
    passed: bool
    error: str | None = None


@dataclass
class TestReport:
    __test__ = False

    cases: list[TestCaseResult] = field(default_factory=list)
    load_error: str | None = None  # the code failed to import or compile
    tests_error: str | None = None  # the tests themselves are broken (not the code's fault)
    timed_out: bool = False
    crashed: str | None = None  # the runner died without reporting

    @property
    def passed(self) -> bool:
        no_errors = not (self.load_error or self.tests_error or self.timed_out or self.crashed)
        return bool(self.cases) and all(case.passed for case in self.cases) and no_errors

    @property
    def passed_count(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def tests_unusable(self) -> bool:
        """The test suite is broken or empty, so no implementation can make it pass."""
        tests_ran = not (self.load_error or self.timed_out or self.crashed)
        return bool(self.tests_error) or (tests_ran and not self.cases)

    def summary(self, max_chars: int = 1500) -> str:
        """The sanitized report shown to the model: names, PASS/FAIL and error types only."""
        if self.timed_out:
            return "Timed out while running the tests (possible infinite loop)."
        if self.crashed:
            return f"Test runner crashed: {self.crashed}"
        if self.load_error:
            return f"Code failed to load: {self.load_error}"
        if self.tests_error:
            return f"The test suite itself is broken: {self.tests_error}"
        if not self.cases:
            return "No test_* functions were found in the test suite."
        lines = [f"{self.passed_count}/{len(self.cases)} tests passed"]
        for case in self.cases:
            lines.append(f"- {case.name}: PASS" if case.passed else f"- {case.name}: FAIL ({case.error})")
        return "\n".join(lines)[:max_chars]


def _run_in_subprocess(payload: dict, timeout_seconds: int) -> TestReport:
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _TEST_RUNNER_SOURCE],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TestReport(timed_out=True)

    # The LAST marker line wins, so code that prints a forged marker earlier cannot fake a pass.
    result_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(RESULT_MARKER)),
        None,
    )
    if result_line is None:  # e.g. the code called sys.exit / os._exit or crashed the interpreter
        return TestReport(crashed=f"exit code {completed.returncode} with no result report")
    data = json.loads(result_line[len(RESULT_MARKER):])
    return TestReport(
        cases=[TestCaseResult(**case) for case in data["cases"]],
        load_error=data["load_error"],
        tests_error=data["tests_error"],
    )


def run_code_against_tests(code: str, tests: str, timeout_seconds: int = 10) -> TestReport:
    """Execute `code`, then the `test_*` functions in `tests`, in one fresh interpreter."""
    return _run_in_subprocess({"code": code, "tests": tests}, timeout_seconds)


def run_workspace_against_tests(workspace_root: str | Path, tests: str,
                                timeout_seconds: int = 10) -> TestReport:
    """Run hidden `tests` against the files in a workspace directory.

    Every top-level module is imported and its public names put in the tests' namespace, so
    tests written as "assume the implementation is defined above" work, and so do tests that
    import normally. The tests are passed in memory and never written to the workspace.
    """
    return _run_in_subprocess({"workspace_root": str(workspace_root), "tests": tests}, timeout_seconds)
