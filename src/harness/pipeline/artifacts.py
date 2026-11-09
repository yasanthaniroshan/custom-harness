"""Typed SDLC artifacts, how they render into the next stage's prompt, and their validation.

Requirements get IDs (R1, R2, ...) when rendered, and every test declares which IDs it covers,
so requirement coverage is checked by code instead of trusted from the model. Code itself is
never JSON-wrapped: the implementer writes it through tool calls.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, field_validator


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


class TestCase(BaseModel):
    __test__ = False  # not a pytest class

    name: str = Field(description="Function name, must start with 'test_'.")
    covers: list[str] = Field(description="Requirement IDs this test covers, e.g. ['R1', 'R3'].")
    code: str = Field(
        description="Complete source of the test function: def test_x(): with plain asserts.")


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


def validate_test_suite(
    suite: TestSuite,
    requirement_count: int,
    min_tests: int,
    run_against_empty_code: StubRunner | None = None,
    max_stub_pass_fraction: float = 0.5,
) -> list[str]:
    """Problems with a generated test suite; an empty list means it may be used.

    run_against_empty_code(rendered_tests) -> {test_name: passed} with an EMPTY implementation.
    Real tests call the function under test, so they fail there; tests that pass anyway
    (`assert True`) prove nothing.
    """
    issues: list[str] = []
    if len(suite.tests) < min_tests:
        issues.append(f"only {len(suite.tests)} tests; write at least {min_tests}")
    names = [test.name for test in suite.tests]
    for duplicate in sorted({name for name in names if names.count(name) > 1}):
        issues.append(f"duplicate test name {duplicate!r}")
    valid_ids = {f"R{number}" for number in range(1, requirement_count + 1)}
    covered_ids: set[str] = set()
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
        covered_ids.update(requirement_id for requirement_id in test.covers if requirement_id in valid_ids)
    uncovered_ids = sorted(valid_ids - covered_ids, key=lambda requirement_id: int(requirement_id[1:]))
    if uncovered_ids:
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
