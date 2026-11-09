import pytest

from harness.config import Config
from harness.training_data.chat_rendering import RenderError, render_row
from harness.training_data.sft_dataset import (
    is_validation_task,
    magicoder_count_for_mix,
    prepare_training_data,
    split_rows,
    tokenize_rows,
)
from tests.fakes import cached_qwen_tokenizer, make_sft_rows, write_jsonl


@pytest.fixture(scope="module")
def rows():
    return make_sft_rows(6)


@pytest.fixture(scope="module")
def tokenizer():
    return cached_qwen_tokenizer()


def test_render_matches_the_real_template_and_masks_only_the_completion(rows, tokenizer):
    row = next(row for row in rows if row["stage"] == "implement" and "tool_calls" in row["completion"][0])
    rendered = render_row(tokenizer, row)
    assert len(rendered.input_ids) == len(rendered.completion_mask) and 0 < rendered.completion_token_count < rendered.token_count
    # prompt tokens are masked out, completion tokens are not
    n_prompt = rendered.token_count - rendered.completion_token_count
    assert rendered.completion_mask[:n_prompt] == [0] * n_prompt and set(rendered.completion_mask[n_prompt:]) == {1}
    assert rendered.completion_text.startswith("<tool_call>") and rendered.completion_text.endswith("<|im_end|>")
    assert "<tools>" in rendered.prompt_text and '"name": "read_file"' in rendered.prompt_text  # tools block
    assert tokenizer.decode(rendered.input_ids[-1:]) == "<|im_end|>"  # the model learns to stop: EOS last, no stray newline
    assert rendered.prompt_text.endswith("<|im_start|>assistant\n")


def test_structured_stage_and_tool_result_turns_render(rows, tokenizer):
    requirements_row = next(row for row in rows if row["stage"] == "requirements")
    assert render_row(tokenizer, requirements_row).completion_text.startswith('{"requirements"')
    after_tool_result = next(row for row in rows if row["stage"] == "implement" and row["prompt"][-1]["role"] == "tool")
    assert "<tool_response>" in render_row(tokenizer, after_tool_result).prompt_text


def test_unrenderable_rows_raise(tokenizer):
    with pytest.raises(RenderError):
        render_row(tokenizer, {"prompt": [{"role": "user", "content": "x"}],
                         "completion": [{"role": "assistant", "content": ""}]})


def test_split_is_by_task_deterministic_and_roughly_the_right_size(rows):
    seed = next(seed for seed in range(50) if all(split_rows(rows, 0.5, seed)))  # a seed with both sides non-empty
    train, validation = split_rows(rows, 0.5, seed)
    assert {row["task_id"] for row in train}.isdisjoint({row["task_id"] for row in validation})
    assert len(train) + len(validation) == len(rows)
    assert split_rows(rows, 0.5, seed) == (train, validation)  # deterministic
    assert split_rows(rows, 0.0, 7) == (rows, [])
    many = [{"task_id": number, "stage": "x"} for number in range(4000)]
    fraction = len(split_rows(many, 0.1, 1)[1]) / 4000
    assert 0.08 < fraction < 0.12
    assert is_validation_task(5, 0.5, 1) == is_validation_task(5, 0.5, 1)


def test_tokenize_drops_overlong_rows_instead_of_truncating_them(rows, tokenizer):
    _, stats = tokenize_rows(rows, tokenizer, 100_000)
    assert stats["kept"] == len(rows) and not stats["dropped_overlong"]
    short, stats = tokenize_rows(rows, tokenizer, 800)
    assert stats["dropped_overlong"] > 0 and stats["kept"] + stats["dropped_overlong"] == len(rows)
    assert all(len(input_ids) <= 800 for input_ids in short.input_ids)
    _, stats = tokenize_rows(rows, tokenizer, 800)  # typed stages are short and survive
    assert stats["kept:requirements"] == 6


def test_prepare_data_builds_datasets_and_per_stage_validation(tmp_path, rows, tokenizer):
    path = write_jsonl(tmp_path / "sft.jsonl", rows)
    config = Config()
    config.data.sft_paths, config.data.magicoder_mix = [str(path)], 0.0  # no Magicoder: no download
    config.train.validation_fraction, config.train.max_length = 0.4, 8192
    data = prepare_training_data(config, tokenizer)
    assert data.train.column_names == ["input_ids", "completion_mask"]
    assert data.validation is not None and len(data.train) + len(data.validation) == len(rows)
    assert set(data.validation_by_stage) <= {"requirements", "design", "write_tests", "implement", "review"}
    assert sum(len(dataset) for dataset in data.validation_by_stage.values()) == len(data.validation)
    assert data.stats["own_rows"] == len(rows)


def test_magicoder_mix_arithmetic():
    assert magicoder_count_for_mix(300, 0.25) == 100 and magicoder_count_for_mix(300, 0.0) == 0
    with pytest.raises(ValueError):
        magicoder_count_for_mix(1, 1.0)
