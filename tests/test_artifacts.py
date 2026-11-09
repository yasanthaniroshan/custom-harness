from harness.pipeline import artifacts
from tests.fakes import DESIGN, GOOD_SUITE, REQUIREMENTS, REVIEW_OK, TRIVIAL_SUITE, UNCOVERED_SUITE

GOOD_SUITE_NAMES = ("test_triples", "test_zero", "test_negative")


def empty_code_results(passing_names: set[str]):
    """A stand-in for running the suite against empty code: only `passing_names` pass."""
    return lambda tests: {name: name in passing_names for name in GOOD_SUITE_NAMES}


def test_rendering_assigns_requirement_ids_and_matches_downstream_formats():
    text = artifacts.render_requirements(REQUIREMENTS)
    assert text.startswith("R1: f(x) returns x tripled\nR2: f(0) is 0\nR3:")
    assert "Edge cases:\n- zero" in text and "Assumptions:\n- inputs are ints" in text
    assert "def f(x: int) -> int\n  returns 3 * x" in artifacts.render_design(DESIGN)
    assert artifacts.render_tests(GOOD_SUITE).count("def test_") == 3
    review = artifacts.Review(summary="S.", findings=[artifacts.Finding(
        severity="Blocker", location="f", problem="off by one", fix="use <=")])
    assert "[Blocker] f - off by one Fix: use <=" in artifacts.render_review(review)  # the filter's format
    assert artifacts.render_review(REVIEW_OK).endswith("No findings.")


def test_a_valid_suite_passes_the_gate():
    assert artifacts.validate_test_suite(GOOD_SUITE, 3, 3, empty_code_results(set())) == []


def test_requirement_coverage_is_checked_by_code_not_trusted():
    assert artifacts.validate_test_suite(UNCOVERED_SUITE, 3, 3) == ["no test covers requirement(s) R3"]


def test_structural_problems_are_all_reported():
    broken = artifacts.TestSuite(tests=[
        artifacts.TestCase(name="test_a", covers=["R9"], code="def check_a():\n    assert f(1)\n"),
        artifacts.TestCase(name="test_b", covers=["R1"], code="def one():\n    pass\n\ndef two():\n    pass\n"),
        artifacts.TestCase(name="test_b", covers=["R2"], code="def test_b(:\n"),
    ])
    assert broken.tests[0].code.startswith("def test_a():")  # a single misnamed function is renamed
    issues = "\n".join(artifacts.validate_test_suite(broken, 3, 5))
    for expected in ("only 3 tests", "duplicate test name 'test_b'",
                     "unknown requirement id(s) ['R9']", "must define a function named test_b",
                     "not valid Python"):
        assert expected in issues, expected


def test_trivial_tests_fail_the_empty_code_check():
    all_pass = {"test_t1": True, "test_t2": True, "test_t3": True}
    issues = artifacts.validate_test_suite(TRIVIAL_SUITE, 3, 3, lambda tests: all_pass)
    assert len(issues) == 1 and "3 of 3 tests pass even with no implementation" in issues[0]
    # one trivial test of three is under the 0.5 threshold: allowed; two of three is not
    assert artifacts.validate_test_suite(GOOD_SUITE, 3, 3, empty_code_results({"test_zero"})) == []
    assert artifacts.validate_test_suite(GOOD_SUITE, 3, 3, empty_code_results({"test_zero", "test_negative"}))


def test_requirement_and_design_validation():
    assert artifacts.validate_requirements(artifacts.Requirements(requirements=["", " "]))
    assert artifacts.validate_requirements(REQUIREMENTS) == []
    assert artifacts.validate_design(artifacts.Design(functions=[], approach="x"))
    not_python = artifacts.Design(functions=[
        artifacts.FunctionSpec(name="f", signature="int f(int x)", behavior="b"),
        artifacts.FunctionSpec(name="f", signature="int f(int x)", behavior="b")], approach="x")
    assert artifacts.validate_design(not_python) == []  # formatting never fails the run...
    warnings = artifacts.design_warnings(not_python)
    assert len(warnings) == 1 and "not a valid Python signature" in warnings[0]  # ...it is a warning, once
    assert artifacts.validate_design(DESIGN) == [] and artifacts.design_warnings(DESIGN) == []


def test_signatures_missing_def_or_with_a_colon_are_normalized_not_rejected():
    for written in ("fibonacci(n: int) -> int", "def fibonacci(n: int) -> int:", " fibonacci(n) "):
        spec = artifacts.FunctionSpec(name="fibonacci", signature=written, behavior="b")
        assert spec.signature.startswith("def fibonacci(n"), written
        assert artifacts.validate_design(artifacts.Design(functions=[spec], approach="x")) == []
    assert artifacts.normalize_signature("class Stack") == "class Stack"
    assert artifacts.normalize_signature("async def fetch(url: str) -> bytes") == "async def fetch(url: str) -> bytes"
    assert artifacts.normalize_signature("") is None
    # methods qualified with their class, as Qwen writes them
    assert artifacts.normalize_signature("ProjectGrouperAdmin.add_project(self, name: str) -> None") == (
        "def add_project(self, name: str) -> None")
    assert artifacts.normalize_signature("def Stack.push(self, item)") == "def push(self, item)"


def test_test_cases_written_as_bare_asserts_or_misnamed_are_normalized():
    # the Kaggle baseline: body only, no def -> "code must define a function named test_empty_list"
    body_only = artifacts.TestCase(name="test_empty_list", covers=["R1"], code="assert f([]) == []\n")
    assert body_only.code == "def test_empty_list():\n    assert f([]) == []\n"
    spaced = artifacts.TestCase(name="Test with empty list", covers=["r2", "Requirement R-3", "4"],
                                code="def test_empty():\n    assert f([]) == []\n")
    assert spaced.name == "test_with_empty_list" and spaced.code.startswith("def test_with_empty_list():")
    assert spaced.covers == ["R2", "R3", "R4"]
    unchanged = artifacts.TestCase(name="test_a", covers=["R1"], code="def helper():\n    pass\n\n"
                                   "def test_a():\n    assert helper() is None\n")
    assert unchanged.code.count("def ") == 2 and "def test_a():" in unchanged.code
    suite = artifacts.TestSuite(tests=[body_only, spaced, artifacts.TestCase(
        name="test_three", covers=["R1"], code="assert f([3]) == [9]")])
    assert artifacts.validate_test_suite(suite, 4, 3) == []


def test_uncovered_requirements_are_relaxed_only_when_asked():
    assert artifacts.uncovered_requirements(UNCOVERED_SUITE, 3) == ["R3"]
    assert artifacts.validate_test_suite(UNCOVERED_SUITE, 3, 3, max_uncovered_fraction=0.5) == []
    assert artifacts.validate_test_suite(UNCOVERED_SUITE, 6, 3, max_uncovered_fraction=0.5)  # 4 of 6 untested


def test_schemas_produce_json_schemas_for_guided_decoding():
    for schema in artifacts.SCHEMA_BY_STAGE.values():
        json_schema = schema.model_json_schema()
        assert json_schema["type"] == "object" and "properties" in json_schema
    severity = artifacts.Review.model_json_schema()["$defs"]["Finding"]["properties"]["severity"]
    assert severity["enum"] == ["Blocker", "Major", "Nit"]
