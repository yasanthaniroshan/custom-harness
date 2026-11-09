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
        artifacts.TestCase(name="check_a", covers=["R9"], code="def check_a():\n    assert f(1)\n"),
        artifacts.TestCase(name="test_b", covers=["R1"], code="def other():\n    pass\n"),
        artifacts.TestCase(name="test_b", covers=["R2"], code="def test_b(:\n"),
    ])
    issues = "\n".join(artifacts.validate_test_suite(broken, 3, 5))
    for expected in ("only 3 tests", "duplicate test name 'test_b'", "must start with 'test_'",
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
    not_python = artifacts.Design(
        functions=[artifacts.FunctionSpec(name="f", signature="f(x)", behavior="b")], approach="x")
    assert "must be a Python signature" in artifacts.validate_design(not_python)[0]
    assert artifacts.validate_design(DESIGN) == []


def test_schemas_produce_json_schemas_for_guided_decoding():
    for schema in artifacts.SCHEMA_BY_STAGE.values():
        json_schema = schema.model_json_schema()
        assert json_schema["type"] == "object" and "properties" in json_schema
    severity = artifacts.Review.model_json_schema()["$defs"]["Finding"]["properties"]["severity"]
    assert severity["enum"] == ["Blocker", "Major", "Nit"]
