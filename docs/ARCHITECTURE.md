# Architecture

How the pieces fit, where to make common changes, and the rules the code relies on.

## Data flow
```
seed tasks (training_data/seed_tasks.py)
   |
   v
pipeline/service.py ── builds PipelineDependencies, opens the graph, calls the runner
   |
   v
pipeline/runner.py ── runs many (task, sample) pairs concurrently against vLLM
   |                    appends one run record per line to runs.jsonl (resumable)
   v
pipeline/graph.py ── one run:
   requirements -> design -> write_tests -> validate_tests -> implement x N -> select -> review
                                  ^              |
                                  +-- retry -----+          implement = pipeline/implementer.py
   |
   v
runs.jsonl ──> training_data/trajectory_filter.py ──> sft.jsonl
                                                         |
               training_data/validation.py <─────────────+  (harness validate-data)
                                                         |
               training_data/sft_dataset.py  ────────────+  split by task, render, tokenize
               training_data/chat_rendering.py              (real chat template)
                                                         |
                                                         v
                                            training/trainer.py ──> adapter/
                                                         |
          evaluation/ (harness_metrics, comparison, humaneval) <── eval runs with the adapter
```

## Modules
| Module | Responsibility |
|---|---|
| `pipeline/stages.py` | What each stage sees (`inputs`) and is told (`instruction`). Change prompts here. |
| `pipeline/artifacts.py` | Pydantic schemas for requirements, design, tests, review; how each renders for the next stage; the test-suite gate. |
| `pipeline/graph.py` | The LangGraph graph, structured-output calls, the gate's retry routing, best-of-N selection. |
| `pipeline/implementer.py` | The tool-using implement subgraph (model -> ToolNode -> model), its temporary workspace and the hidden-test `run_tests` tool. |
| `pipeline/runner.py` | Concurrency, resume (by output file and by checkpoint), the run-record format. |
| `pipeline/service.py` | Entry points: `generate_runs`, `evaluate_on_tasks`, `run_single_task`. |
| `pipeline/tracing.py` | LangChain callback -> `events.jsonl`; LangSmith run config and feedback. |
| `pipeline/messages.py` | Message -> chat-template dict; conversation compaction for the tool loop. |
| `pipeline/tool_adapters.py` | Workspace tools as LangChain `StructuredTool`s. |
| `pipeline/chat_models.py` | `ChatOpenAI` per temperature, pointed at vLLM. |
| `tools/` | Workspace path jail and hidden paths, tool registry (validation, loop detection, truncation), search and edit tools, tolerant patching. |
| `sandbox.py` | Runs code or a workspace against tests in a subprocess; the sanitized report. |
| `prompt_context.py` | Builds every stage prompt inside the token budget; few-shot retrieval. |
| `training_data/` | Trust filter, SFT-row validation, chat-template rendering, dataset building, seed tasks. |
| `training/` | Model loading (4-bit), SFT trainer, train-log summary. |
| `evaluation/` | Held-out metrics, statistics, A/B comparison, HumanEval. |
| `serving.py` | The vLLM command line, readiness check, background start and stop. |
| `cli.py` | Argument parsing; each command is a short function that calls the modules above. |

Dependencies point one way: `cli` -> `pipeline` / `training` / `evaluation` -> `training_data`,
`tools`, `sandbox`, `prompt_context`, `monitoring`, `config`. Heavy libraries (torch, transformers,
datasets, TRL) are imported inside the functions that need them, so the pipeline and its tests
run without the training stack.

## Invariants (tests enforce each of these)
1. **The implementer never sees the tests.** Tests reach only `run_tests` and the final
   verification, in memory; the report has names, PASS/FAIL and error types only. Tested in
   `test_pipeline_graph.py` (prompts) and `test_validation.py` (`hidden_test_leak` in training rows).
2. **Each stage sees only its declared inputs** (`stages.py`). Tested in `test_prompt_context.py`.
3. **Training text equals inference text.** Rows are rendered with the model's own chat template;
   tool-call arguments stay dicts (a JSON string is double-encoded by Qwen's template).
4. **Only trusted, accepted outputs become training data**: never a rejected test suite, never a
   losing candidate's trajectory, never a failed run.
5. **Validation is split by task**, so validation loss is never measured on a task seen in training.
6. **Monitoring never breaks a run** (the tracing callback and LangSmith feedback swallow errors).

## Common changes
- **Change a prompt or what a stage sees**: `pipeline/stages.py`.
- **Change an artifact's fields**: `pipeline/artifacts.py` (schema + `render_*`); the JSON schema
  sent to vLLM and the validator's schema check follow automatically.
- **Add a tool**: write it in `tools/search_tools.py` or `tools/edit_tools.py` (a function plus a
  `Tool(...)` entry); it is exposed to the model automatically.
- **Add a trust rule**: `training_data/trajectory_filter.rejection_reason`.
- **Add a data check**: `training_data/validation.py` (`collector.add("error" | "warning", ...)`).
- **Add a CLI command**: a `command_<name>` function plus its parser in `cli.build_parser`, and
  an entry in `COMMANDS`.
- **Add a setting**: a field in `config.py` and the same key in `configs/default.yaml`
  (`test_kaggle.py` fails if they differ).

## Run record (one line of runs.jsonl)
```
task_id, sample, system_prompt,
artifacts: {task, requirements, design, tests, code, diff, test_output, review},
llm_calls: [{stage, attempt, messages, completion, [completion_message, tool_schemas, passed]}],
passed, test_count, steps, tool_calls, tool_errors, stop_reason,
tests_valid, failure_reason, candidate_count, candidates_passed
```
An SFT row (one line of sft.jsonl) is `{stage, task_id, prompt: [messages], completion:
[assistant message], tools?: [schemas]}`.
