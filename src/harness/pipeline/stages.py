"""Declarative SDLC stages: what each stage sees and what it is asked to do.

Test-driven and linear: requirements -> design -> write_tests -> implement -> review.
Each stage reads ONLY the artifacts listed in `inputs` and writes exactly one new artifact.
Change a stage's `inputs` here to change what it sees. The implementer never sees the tests:
it can only run them and read a sanitized pass/fail report.

Prompt techniques adapted (Apache-2.0) from google/elephant-goldfish-model: explicit
assumptions (design_critic), implementation-readiness design (implementation_readiness), and
paraphrase-first, severity-tagged review (comprehension).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    inputs: tuple[str, ...]  # artifacts this stage may see
    output: str  # artifact it produces
    instruction: str  # appended after the input artifacts


REQUIREMENTS = Stage(
    name="requirements",
    inputs=("task",),
    output="requirements",
    instruction=(
        "List the functional requirements as short, testable bullets. Then list edge cases and "
        "failure behavior (empty or invalid input, boundaries, duplicates). State every "
        "assumption you make explicitly instead of leaving it implied."
    ),
)

DESIGN = Stage(
    name="design",
    inputs=("requirements",),
    output="design",
    instruction=(
        "Describe the function signatures (names, parameters, types, return values, errors "
        "raised) and the approach in a few lines. Make every design decision yourself (data "
        "structures, error handling, edge-case behavior) so an implementer with no other "
        "context has no choices left to make."
    ),
)

WRITE_TESTS = Stage(
    name="write_tests",
    inputs=("requirements", "design"),
    output="tests",
    instruction=(
        "Write the test functions. Each test is a complete `def test_<behavior>():` function "
        "using plain `assert` statements (no pytest) and lists in `covers` the requirement IDs "
        "(R1, R2, ...) it checks. Cover every requirement and every edge case. Assume the "
        "implementation is defined above the tests and matches the signatures in the design."
    ),
)

IMPLEMENT = Stage(
    name="implement",
    inputs=("design",),
    output="code",
    instruction=(
        "Implement this design in the workspace, in {solution_file}, using write_file (no tests). "
        "Call run_tests to check your work against hidden tests: you see pass/fail per test, not "
        "the tests themselves. Fix failures with str_replace. When all tests pass, reply with a "
        "short summary and no tool call."
    ),
)

REVIEW = Stage(
    name="review",
    inputs=("code", "test_output"),
    output="review",
    instruction=(
        "Summarize in 1-2 sentences what this code does. Then list findings with a severity "
        "(Blocker only for real correctness problems), the location, the problem and a "
        "concrete fix, using the test results. Return no findings if there are none."
    ),
)

STAGES: tuple[Stage, ...] = (REQUIREMENTS, DESIGN, WRITE_TESTS, IMPLEMENT, REVIEW)
STAGES_BY_NAME: dict[str, Stage] = {stage.name: stage for stage in STAGES}
