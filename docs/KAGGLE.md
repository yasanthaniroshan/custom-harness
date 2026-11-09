# Fine-tuning on Kaggle (2x T4)

The loop: **generate** data with the harness -> **filter** it -> **validate** it -> **train** (QLoRA)
-> **evaluate** against the baseline. `notebooks/kaggle_pipeline.ipynb` runs it in stages.

## Getting the code into Kaggle
**Recommended: a GitHub repository, pinned to a tag or commit.**
- Every round records exactly which code produced it (`outputs/roundN/code_version.txt`).
- Updating means `git push` + changing `GIT_REF`; there is no zip to rebuild and re-upload.
- Private repos work: create a fine-grained GitHub token with **read-only Contents** access to
  this one repo and add it as the Kaggle secret `GITHUB_TOKEN`. The notebook uses it for the
  clone, never prints it, and removes it from the clone's remote URL.

Pin `GIT_REF` to a tag (`git tag v0.3.0 && git push --tags`) or a commit SHA, not a branch:
a branch moves, so re-running an old notebook version would silently use newer code.

**Fallback: a Kaggle Dataset** (no internet in the notebook, or no GitHub account):
`python scripts/bundle.py` -> `harness-src.zip`, upload it under *Datasets -> New Dataset*,
leave `REPO_URL` empty. Each new upload is a dataset version; note which one each round used.

## One-time setup
1. Push the code to GitHub (above) and set `REPO_URL` and `GIT_REF` in the notebook settings cell.
2. New notebook -> upload `notebooks/kaggle_pipeline.ipynb` (File -> Import).
3. Settings: **Accelerator GPU T4 x2**, **Internet on**.
4. Add the **Qwen2.5-Coder 7B Instruct** Kaggle Model (the path in `configs/kaggle.yaml`; edit
   it if yours differs).
5. Add-ons -> Secrets: `GITHUB_TOKEN` (private repo only), `HF_TOKEN` (optional, faster
   downloads) and, only for LangSmith tracing, `LANGSMITH_API_KEY` plus
   `langsmith.enabled: true` in `configs/kaggle.yaml`.

## Running a round (round 0)
Run the notebook with `STAGE` set to each value in turn (or `all` for a short pilot):

| STAGE | Check this before moving on |
|---|---|
| `setup` | `nvidia-smi` lists 2 GPUs |
| `baseline` | `harness report`: **truncated** column ~0, tool-error rate low. If truncation is high, raise `sampling.max_new_tokens`; if tool errors are high, the 7B model is not coping with tool calls and more data will not fix it |
| `generate` | the **filter stats**: if few runs survive, the base model's tests or format are the bottleneck. Then `validate-data` must say *OK to train* |
| `train` | `train-report`: `overfitting: false`; per-stage validation loss all decreasing |
| `evaluate` | the three `compare` outputs (below) |
| `save` | Save Version -> Save & Run All (Commit) to keep `/kaggle/working` |

### Reading the three comparisons
- **self-eval** (held-out tasks, judged by tests the model wrote): a noisy proxy. Look at
  `trusted`, not `passed`.
- **HumanEval, harness**: the external check that the pipeline improved.
- **HumanEval, direct**: the *regression* check. If this says `WORSE`, fine-tuning damaged general
  coding ability: lower the learning rate, train fewer epochs, or raise `data.magicoder_mix`.

Every comparison prints a 95% interval and a paired p-value. With 40 problems almost nothing is
significant. The tool says "too few paired tasks" rather than guessing; scale `EVAL_TASKS` and
`BENCH_LIMIT` for a real decision.

## Round 1+
Round 1 trains **from the base model** on all accumulated data (avoids compounding drift), but
generates with the previous adapter. Download round 0's results, add them to the notebook as a
Dataset, and set:
```python
ROUND = 1
PREV_SFT = ["/kaggle/input/harness-r0/round0/sft.jsonl"]
PREV_ADAPTER = "/kaggle/input/harness-r0/train0/adapter"
```
Compare round 1 against the **round 0 baseline**, not against round 0's tuned model.

## Budget and quotas
Sessions last <= 12 h and the weekly GPU quota is limited. vLLM (both GPUs) and training (one GPU)
cannot run together, so stages are sequential; the filtered data and adapter persist in
`/kaggle/working`. Generation resumes where it stopped (`runs.jsonl` is appended per run, and
`graph.checkpoint_db` resumes runs interrupted mid-graph), so a timed-out session loses little.
To re-run a stage from scratch (e.g. after a code change), delete its output directory: the
checkpoints are tied to it through `<runs file>.checkpoint-scope`, so they are discarded too.
**Run the pilot first and time each stage** before scaling `NUM_TASKS`: I could not measure
throughput without the hardware.

## Known cost: loss tokens are a small share
Each tool-loop turn is its own training row and re-sends the tool schemas plus the whole
conversation, so only a few percent of tokens carry loss (`validate-data` prints
`loss_token_share`; the fake pipeline measured ~3.5%). Training is therefore slower per useful
token than it looks. Mitigations, in order: lower `agent.max_agent_steps`, keep trajectories that
finish quickly, or (code change) train one sequence per trajectory with loss on every assistant
turn.

## If something fails
- **OOM during training**: `train.max_length` 2048 -> 1536, keep `batch_size: 1`, keep
  `gradient_checkpointing`, use `optimizer: paged_adamw_8bit`, or lower `lora.rank`.
- **vLLM will not start on the T4**: pin an older release with `VLLM_SPEC` in the notebook;
  `max_model_len` / `gpu_memory_utilization` down if it runs out of memory. Check
  `outputs/vllm_server.log`.
- **Training says the data failed validation**: read the ERRORS block. `hidden_test_leak` means
  an assertion from the hidden tests reached the implementer; fix the cause, do not use
  `--skip-validation`.
- **401 from Hugging Face**: a stale token. Re-add the `HF_TOKEN` secret, or remove it for public
  models.
