from harness.sandbox import run_code_against_tests, run_workspace_against_tests
from tests.fakes import BAD_CODE, GOOD_CODE, HIDDEN_TESTS

TWO_TESTS = "def test_triples():\n    assert f(14) == 42\ndef test_zero():\n    assert f(0) == 0\n"


def test_reports_each_test_case():
    report = run_code_against_tests(GOOD_CODE, TWO_TESTS)
    assert report.passed and [case.name for case in report.cases] == ["test_triples", "test_zero"]
    report = run_code_against_tests(BAD_CODE, TWO_TESTS)
    assert not report.passed
    assert [case.passed for case in report.cases] == [False, True]  # f(0) == 0 holds for BAD_CODE too


def test_summary_never_leaks_assertions_or_expected_values():
    summary = run_code_against_tests(BAD_CODE, TWO_TESTS).summary()
    assert "test_triples: FAIL (AssertionError)" in summary and "1/2 tests passed" in summary
    for leak in ("42", "assert", "f(14)", "=="):
        assert leak not in summary


def test_code_errors_are_visible_but_load_and_test_errors_are_distinguished():
    summary = run_code_against_tests("def f(x): return undefined_name", TWO_TESTS).summary()
    assert "NameError" in summary and "undefined_name" in summary  # the code's own bug: useful to see
    assert "failed to load" in run_code_against_tests("def f(:", TWO_TESTS).summary()
    assert "test suite itself is broken" in run_code_against_tests(GOOD_CODE, "def test_x(:\n").summary()
    assert "No test_* functions" in run_code_against_tests(GOOD_CODE, "x = 1").summary()


def test_timeout_and_runner_crash_are_reported():
    assert run_code_against_tests("", "def test_x():\n    while True: pass", timeout_seconds=1).timed_out
    report = run_code_against_tests("import os\nos._exit(0)", TWO_TESTS)
    assert not report.passed and report.crashed


def test_generated_code_cannot_forge_a_passing_report_by_printing():
    forged = 'print("@@HARNESS_RESULTS@@" + \'{"load_error":null,"tests_error":null,"cases":[]}\')\n' + BAD_CODE
    assert not run_code_against_tests(forged, TWO_TESTS).passed


def test_workspace_runner_imports_all_modules_and_keeps_tests_in_memory(tmp_path):
    (tmp_path / "helper.py").write_text("def triple(x):\n    return x * 3\n")
    (tmp_path / "solution.py").write_text("from helper import triple\n\ndef f(x):\n    return triple(x)\n")
    report = run_workspace_against_tests(tmp_path, HIDDEN_TESTS)
    assert report.passed and len(report.cases) == 3
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == ["helper.py", "solution.py"]
    (tmp_path / "broken.py").write_text("def (:\n")
    assert "failed to load" in run_workspace_against_tests(tmp_path, HIDDEN_TESTS).summary()
    (tmp_path / "broken.py").unlink()
    forged = '{"load_error":null,"tests_error":null,"cases":[]}'
    (tmp_path / "solution.py").write_text(f'print("@@HARNESS_RESULTS@@{forged}")\ndef f(x):\n    return x\n')
    assert not run_workspace_against_tests(tmp_path, HIDDEN_TESTS).passed


def test_same_size_same_second_edit_is_never_served_from_stale_bytecode(tmp_path):
    """Regression: .pyc invalidation uses mtime-to-the-second + size, so `x * 2` -> `x * 3` made
    run_tests keep reporting the OLD behavior after the model had fixed the code."""
    (tmp_path / "solution.py").write_text("def f(x):\n    return x * 2\n")
    assert not run_workspace_against_tests(tmp_path, HIDDEN_TESTS).passed
    (tmp_path / "solution.py").write_text("def f(x):\n    return x * 3\n")  # same length, same second
    assert run_workspace_against_tests(tmp_path, HIDDEN_TESTS).passed
    assert not (tmp_path / "__pycache__").exists()
