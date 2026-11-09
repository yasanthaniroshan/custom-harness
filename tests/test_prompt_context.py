import json

from harness.config import ContextConfig
from harness.monitoring import Tracer
from harness.pipeline.stages import DESIGN, IMPLEMENT, REQUIREMENTS, REVIEW, STAGES
from harness.prompt_context import (
    FewShotStore,
    PromptBuilder,
    PromptSection,
    approximate_token_count,
    assemble_sections,
    truncate_text,
)


def test_truncate_modes_fit_and_mark():
    text = "\n".join(f"line {number}" for number in range(200))
    for mode in ("head", "middle", "tail"):
        truncated = truncate_text(text, 50, approximate_token_count, mode)
        assert approximate_token_count(truncated) <= 50 and "tokens truncated" in truncated
    assert truncate_text(text, 50, approximate_token_count, "head").startswith("line 0")
    assert truncate_text(text, 50, approximate_token_count, "tail").rstrip().endswith("line 199")
    middle = truncate_text(text, 50, approximate_token_count, "middle")
    assert middle.startswith("line 0") and middle.rstrip().endswith("line 199")
    assert truncate_text("short", 50, approximate_token_count) == "short"  # untouched when it fits


def test_assembly_shrinks_least_important_first_and_never_touches_priority_zero():
    sections = [
        PromptSection("instruction", "DO THE THING", priority=0),
        PromptSection("artifact", "a" * 800, priority=2),
        PromptSection("example", "e" * 800, priority=4, mode="drop"),
    ]
    prompt, events = assemble_sections(sections, budget_tokens=260, count_tokens=approximate_token_count)
    assert "DO THE THING" in prompt and "e" * 50 not in prompt  # the example goes first
    assert "a" * 700 in prompt  # the artifact survived
    assert events[0]["section"] == "example" and events[0]["dropped"]
    assert approximate_token_count(prompt) <= 260


def test_assembly_truncates_the_body_but_keeps_code_fences():
    sections = [PromptSection("code", "x = 1\n" * 500, "Code:\n```python\n", "\n```", priority=2)]
    prompt, _ = assemble_sections(sections, budget_tokens=100, count_tokens=approximate_token_count)
    assert prompt.startswith("Code:\n```python\n") and prompt.endswith("\n```")
    assert approximate_token_count(prompt) <= 100


def test_overflow_is_reported_when_only_protected_text_remains():
    _, events = assemble_sections([PromptSection("instruction", "x" * 4000, priority=0)], 100,
                                  approximate_token_count)
    assert events == [{"kind": "context_overflow", "tokens": 1000, "budget": 100}]


def test_each_stage_sees_only_its_declared_inputs():
    artifacts = {name: f"<<{name}>>" for name in ("task", "requirements", "design", "code", "tests", "test_output")}
    builder = PromptBuilder()
    for stage in STAGES:
        prompt = builder.build_prompt(stage, artifacts)
        for name in artifacts:
            assert (f"<<{name}>>" in prompt) == (name in stage.inputs), (stage.name, name)


def test_prompt_is_artifacts_then_instruction_without_budget_pressure():
    prompt = PromptBuilder().build_prompt(IMPLEMENT, {"design": "DESIGN"}, "Do it.")
    assert prompt == "Design:\nDESIGN\nDo it."


def test_budget_applies_and_emits_events_tagged_with_the_stage():
    tracer = Tracer()
    builder = PromptBuilder(ContextConfig(max_prompt_tokens=300, artifact_max_tokens=250), tracer=tracer)
    artifacts = {"code": "def f():\n    return 1\n" * 200, "test_output": "FAIL\n" * 1000}
    prompt = builder.build_prompt(REVIEW, artifacts)
    assert approximate_token_count(prompt) <= 300 and REVIEW.instruction in prompt
    truncated_sections = {event["section"] for event in tracer.events if event["kind"] == "context_truncated"}
    assert truncated_sections == {"artifact:code", "artifact:test_output"}
    assert all(event["stage"] == "review" for event in tracer.events)


def test_conventions_are_injected_per_stage_and_for_all_stages():
    builder = PromptBuilder(ContextConfig(conventions={"*": "Use type hints.", "design": "Prefer pure functions."}))
    design_prompt = builder.build_prompt(DESIGN, {"requirements": "R"})
    assert "Use type hints." in design_prompt and "Prefer pure functions." in design_prompt
    assert "Prefer pure functions." not in builder.build_prompt(REQUIREMENTS, {"task": "T"})


def sft_row(stage, prompt, completion, task_id=1):
    return {"stage": stage, "task_id": task_id,
            "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": completion}]}


def test_few_shot_retrieves_similar_examples_per_stage(tmp_path):
    rows = [
        sft_row("design", "Requirements: merge overlapping intervals sorted by start", "DESIGN-INTERVALS"),
        sft_row("design", "Requirements: parse csv rows into dictionaries", "DESIGN-CSV"),
        sft_row("requirements", "Task: merge intervals", "REQ-INTERVALS"),
    ]
    path = tmp_path / "sft.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    store = FewShotStore.from_jsonl(path)
    assert store.retrieve("design", "merge overlapping intervals", 1)[0][1] == "DESIGN-INTERVALS"
    assert store.retrieve("design", "zzz unrelated", 3) == []  # no zero-score filler
    assert store.retrieve("review", "merge intervals", 3) == []  # per stage only

    builder = PromptBuilder(ContextConfig(few_shot_count=1), few_shot_store=store)
    prompt = builder.build_prompt(DESIGN, {"requirements": "merge overlapping intervals"})
    assert "Example input:" in prompt and "DESIGN-INTERVALS" in prompt and "DESIGN-CSV" not in prompt


def test_oversized_examples_are_skipped_and_examples_go_first_under_pressure():
    big_store = FewShotStore([sft_row("design", "intervals " * 400, "DESIGN-BIG")])
    builder = PromptBuilder(ContextConfig(few_shot_count=1, few_shot_max_tokens=100), few_shot_store=big_store)
    assert "DESIGN-BIG" not in builder.build_prompt(DESIGN, {"requirements": "intervals"})
    small_store = FewShotStore([sft_row("design", "intervals", "DESIGN-SMALL")])
    tight = PromptBuilder(ContextConfig(few_shot_count=1, max_prompt_tokens=60), few_shot_store=small_store)
    assert "DESIGN-SMALL" not in tight.build_prompt(DESIGN, {"requirements": "intervals " * 10})
