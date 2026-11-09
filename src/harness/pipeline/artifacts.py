"""Typed SDLC artifacts, how they render into the next stage's prompt, and their validation.

Requirements get IDs (R1, R2, ...) when rendered, and every test declares which IDs it covers,
so requirement coverage is checked by code instead of trusted from the model. Code itself is
never JSON-wrapped: the implementer writes it through tool calls.
"""
from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Requirements(BaseModel):
    requirements: list[str] = Field(
        description="Short, testable functional requirements, one per item.")
    edge_cases: list[str] = Field(
        default_factory=list,
        description="Edge cases and failure behavior: empty or invalid input, boundaries, duplicates.")
    assumptions: list[str] = Field(
        default_factory=list, description="Every assumption made, stated explicitly.")


_QUALIFIED_NAME = re.compile(r"^((?:async\s+)?(?:def\s+)?)(?:\w+\.)+(\w+\s*\()")


def normalize_signature(signature: str) -> str | None:
    """A Python `def`/`class` header for `signature`, or None if it is not one.

    Models often leave out the `def` keyword, add a trailing colon, or qualify a method with its
    class (`Stack.push(self, item)`); all are fixed here, so a usable signature is never
    rejected over formatting.
    """
    header = signature.strip().removesuffix(":").strip()
    if not header:
        return None
    # `def Stack.push(...)` / `Stack.push(...)` -> `def push(...)`: keep only the method name
    header = _QUALIFIED_NAME.sub(lambda match: match.group(1) + match.group(2), header, count=1)
    has_keyword = header.startswith(("def ", "async def ", "class "))
    candidates = [header] if has_keyword else [f"def {header}", f"class {header}"]
    for candidate in candidates:
        try:
            ast.parse(candidate + ":\n    pass\n")
        except SyntaxError:
            continue
        return candidate
    return None


class FunctionSpec(BaseModel):
    name: str = Field(description="Function name.")
    signature: str = Field(
        description="Python signature with types, starting with def, "
                    "e.g. 'def merge(xs: list[int]) -> list[int]'.")
    behavior: str = Field(description="What it does, including how edge cases behave.")
    raises: list[str] = Field(default_factory=list, description="Exceptions it raises, and when.")

    @field_validator("signature")
    @classmethod
    def _normalize_signature(cls, signature: str) -> str:
        return normalize_signature(signature) or signature  # invalid ones are reported by validate_design


class Design(BaseModel):
    functions: list[FunctionSpec] = Field(description="Every function the implementation needs.")
    approach: str = Field(description="The algorithm/approach in a few lines.")
    decisions: list[str] = Field(
        default_factory=list,
        description="Design decisions made here so the implementer has none left to make.")


_REQUIREMENT_ID = re.compile(r"\bR\s*-?\s*(\d+)\b", re.IGNORECASE)


def normalize_test_name(name: str) -> str:
    """`'Test with empty list'` -> `test_with_empty_list`; `empty_list` -> `test_empty_list`."""
    slug = name.strip()
    if not slug.isidentifier():
        slug = re.sub(r"\W+", "_", slug).strip("_").lower()
    if not slug:
        return name
    return slug if slug.startswith("test_") else "test_" + slug.removeprefix("test").lstrip("_")


def normalize_requirement_ids(covers: list[str]) -> list[str]:
    """`['r1', 'R-2', 'Requirement R3', '4']` -> `['R1', 'R2', 'R3', 'R4']` (order kept, no duplicates)."""
    ids: list[str] = []
    for item in covers:
        text = str(item).strip()
        found = [f"R{number}" for number in _REQUIREMENT_ID.findall(text)]
        if not found and text.isdigit():
            found = [f"R{text}"]
        for requirement_id in found or [text]:
            if requirement_id not in ids:
                ids.append(requirement_id)
    return ids


def normalize_test_code(name: str, code: str) -> str:
    """Make `code` define a function called `name`, fixing the two ways models get it wrong.

    - only the body was written (`assert f(1) == 3`): it is wrapped in `def name():`
    - the one function it defines has another name: that function is renamed to `name`
    Code that cannot be fixed safely is returned unchanged, and the gate reports it.
    """
    source = textwrap.dedent(code).strip("\n")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return code
    function_names = [node.name for node in tree.body
                      if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]
    has_class = any(isinstance(node, ast.ClassDef) for node in tree.body)
    if name in function_names or has_class or not name.isidentifier():
        return code
    if not function_names:
        return f"def {name}():\n" + textwrap.indent(source, "    ") + "\n"
    test_like = [function_name for function_name in function_names if function_name.startswith("test")]
    candidates = test_like if test_like else function_names
    if len(candidates) != 1:
        return code
    return re.sub(rf"\bdef\s+{re.escape(candidates[0])}\s*\(", f"def {name}(", source, count=1) + "\n"


class TestCase(BaseModel):
    __test__ = False  # not a pytest class

    name: str = Field(description="Function name, must start with 'test_'.")
    covers: list[str] = Field(description="Requirement IDs this test covers, e.g. ['R1', 'R3'].")
    code: str = Field(
        description="Complete source of the test function: def test_x(): with plain asserts.")

    @model_validator(mode="after")
    def _normalize(self) -> TestCase:
        """Fix formatting slips (name, IDs, missing `def`) so good tests are not rejected over them."""
        self.name = normalize_test_name(self.name)
        self.covers = normalize_requirement_ids(self.covers)
        self.code = normalize_test_code(self.name, self.code)
        return self


class TestSuite(BaseModel):
    __test__ = False

    tests: list[TestCase]


class Finding(BaseModel):
    severity: Literal["Blocker", "Major", "Nit"]
    location: str = Field(description="Function or line the finding is about.")
    problem: str
    fix: str = Field(description="Concrete fix.")


class Review(BaseModel):
    summary: str = Field(description="1-2 sentence paraphrase of what the code does.")
    findings: list[Finding] = Field(default_factory=list)


# Structured stages and the schema their completion must match.
SCHEMA_BY_STAGE: dict[str, type[BaseModel]] = {
    "requirements": Requirements, "design": Design, "write_tests": TestSuite, "review": Review,
}


# ---- rendering: artifact -> prompt text for the next stage -------------------------------

def render_requirements(requirements: Requirements) -> str:
    lines = [f"R{number}: {text}" for number, text in enumerate(requirements.requirements, 1)]
    if requirements.edge_cases:
        lines += ["", "Edge cases:"] + [f"- {edge_case}" for edge_case in requirements.edge_cases]
    if requirements.assumptions:
        lines += ["", "Assumptions:"] + [f"- {assumption}" for assumption in requirements.assumptions]
    return "\n".join(lines)


def render_design(design: Design) -> str:
    lines = []
    for function in design.functions:
        lines.append(f"{function.signature}\n  {function.behavior}")
        if function.raises:
            lines.append("  Raises: " + "; ".join(function.raises))
    lines += ["", f"Approach: {design.approach}"]
    if design.decisions:
        lines += ["Decisions:"] + [f"- {decision}" for decision in design.decisions]
    return "\n".join(lines)


def render_tests(suite: TestSuite) -> str:
    return "\n\n".join(test.code.strip("\n") for test in suite.tests) + "\n"


def render_review(review: Review) -> str:
    """One `[Severity] location - problem Fix: ...` line per finding (the trust filter reads it)."""
    lines = [f"[{finding.severity}] {finding.location} - {finding.problem} Fix: {finding.fix}"
             for finding in review.findings]
    return review.summary + "\n" + ("\n".join(lines) if lines else "No findings.")


# ---- validation --------------------------------------------------------------------------

def validate_requirements(requirements: Requirements) -> list[str]:
    if not [text for text in requirements.requirements if text.strip()]:
        return ["no requirements were produced"]
    return []


def validate_design(design: Design) -> list[str]:
    """Problems that make the design unusable (they fail the run).

    Signature formatting is deliberately NOT one of them: signatures are normalized where
    possible, and one the implementer can still read (e.g. `int f(int x)`) is reported by
    design_warnings instead of throwing the whole run away.
    """
    if not design.functions:
        return ["the design lists no functions"]
    return []


def design_warnings(design: Design) -> list[str]:
    """Non-fatal problems: signatures that are not valid Python, each reported once."""
    warnings: list[str] = []
    for function in design.functions:
        if normalize_signature(function.signature) is None:
            warning = (f"signature of {function.name!r} is not a valid Python signature: "
                       f"{function.signature[:80]!r}")
            if warning not in warnings:
                warnings.append(warning)
    return warnings


StubRunner = Callable[[str], dict[str, bool]]  # rendered tests -> {test name: passed}


def uncovered_requirements(suite: TestSuite, requirement_count: int) -> list[str]:
    """Requirement IDs (R1..Rn) that no test in `suite` declares it covers, in order."""
    covered_ids = {requirement_id for test in suite.tests for requirement_id in test.covers}
    return [f"R{number}" for number in range(1, requirement_count + 1) if f"R{number}" not in covered_ids]


def validate_test_suite(
    suite: TestSuite,
    requirement_count: int,
    min_tests: int,
    run_against_empty_code: StubRunner | None = None,
    max_stub_pass_fraction: float = 0.5,
    max_uncovered_fraction: float = 0.0,
) -> list[str]:
    """Problems with a generated test suite; an empty list means it may be used.

    run_against_empty_code(rendered_tests) -> {test_name: passed} with an EMPTY implementation.
    Real tests call the function under test, so they fail there; tests that pass anyway
    (`assert True`) prove nothing.
    max_uncovered_fraction: share of requirements allowed to have no test (0 = every one needs
    a test). The pipeline relaxes it on the last attempt; see uncovered_requirements.
    """
    issues: list[str] = []
    if len(suite.tests) < min_tests:
        issues.append(f"only {len(suite.tests)} tests; write at least {min_tests}")
    names = [test.name for test in suite.tests]
    for duplicate in sorted({name for name in names if names.count(name) > 1}):
        issues.append(f"duplicate test name {duplicate!r}")
    valid_ids = {f"R{number}" for number in range(1, requirement_count + 1)}
    for test in suite.tests:
        if not test.name.startswith("test_"):
            issues.append(f"test name {test.name!r} must start with 'test_'")
        try:
            tree = ast.parse(test.code)
        except SyntaxError as error:
            issues.append(f"{test.name}: code is not valid Python (line {error.lineno}: {error.msg})")
            continue
        defined_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        if test.name not in defined_functions:
            issues.append(f"{test.name}: code must define a function named {test.name}")
        unknown_ids = [requirement_id for requirement_id in test.covers if requirement_id not in valid_ids]
        if unknown_ids:
            issues.append(f"{test.name}: covers unknown requirement id(s) {unknown_ids}")
    uncovered_ids = uncovered_requirements(suite, requirement_count)
    if uncovered_ids and len(uncovered_ids) > max_uncovered_fraction * requirement_count:
        issues.append(f"no test covers requirement(s) {', '.join(uncovered_ids)}")
    if run_against_empty_code is not None and not issues:
        results = run_against_empty_code(render_tests(suite))
        trivially_passing = sorted(name for name, passed in results.items() if passed)
        if results and len(trivially_passing) / len(results) > max_stub_pass_fraction:
            issues.append(
                f"{len(trivially_passing)} of {len(results)} tests pass even with no implementation "
                f"({', '.join(trivially_passing[:3])}); each test must call the function under test "
                "and assert on its result")
    return issues
