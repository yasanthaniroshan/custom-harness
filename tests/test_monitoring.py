import json

from harness.monitoring import Tracer, format_report, read_events, stage_scope, summarize_events


def test_tracer_writes_jsonl_and_tags_the_current_stage(tmp_path):
    path = tmp_path / "events.jsonl"
    tracer = Tracer(path)
    with stage_scope("design"):
        tracer.emit("x", value=1)
    tracer.emit("y")
    events = read_events(path)
    assert [(event["kind"], event["stage"]) for event in events] == [("x", "design"), ("y", None)]
    assert tracer.events == []  # not kept in memory when writing to disk


def test_summary_counts_truncation_latency_tools_context_and_runs():
    events = [
        {"kind": "llm_call", "stage": "implement", "prompt_tokens": 10, "completion_tokens": 2048,
         "truncated": True, "seconds": 2.0},
        {"kind": "llm_call", "stage": "implement", "prompt_tokens": 20, "completion_tokens": 100,
         "truncated": False, "seconds": 1.0},
        {"kind": "tool_call", "stage": "implement", "tool": "grep", "ok": False, "output_tokens": 5},
        {"kind": "tool_call", "stage": "implement", "tool": "grep", "ok": True, "output_tokens": 15},
        {"kind": "context_truncated", "stage": "review", "section": "artifact:code"},
        {"kind": "run_end", "passed": True, "steps": 4, "tool_errors": 1, "stop_reason": "finished"},
        {"kind": "run_end", "passed": False, "steps": 12, "tool_errors": 3, "stop_reason": "max_steps"},
    ]
    summary = summarize_events(events)
    implement = summary["stages"]["implement"]
    assert implement["truncated"] == 1 and implement["truncation_rate"] == 0.5 and implement["seconds_mean"] == 1.5
    assert summary["tools"]["grep"] == {"calls": 2, "error_rate": 0.5, "output_tokens_mean": 10.0}
    assert summary["context"] == {"context_truncated:artifact:code": 1}
    assert summary["runs"] == {"count": 2, "pass_rate": 0.5, "mean_steps": 8.0, "mean_tool_errors": 2.0,
                               "stop_reasons": {"finished": 1, "max_steps": 1}}
    json.dumps(summary)  # report data is serializable
    report = format_report(summary)
    assert "implement" in report and "Runs: 2, pass rate 0.5" in report and "grep x2 (50% err)" in report
