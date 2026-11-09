import json

import pytest

from harness.cli import COMMANDS, build_parser, main


def test_every_command_has_a_parser_and_a_handler():
    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    assert set(subcommands) == set(COMMANDS)


def test_report_and_train_report_read_files_without_the_training_stack(tmp_path, capsys):
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"kind": "llm_call", "stage": "design", "prompt_tokens": 10,
                                  "completion_tokens": 5, "truncated": False, "seconds": 0.5}) + "\n")
    main(["report", str(events)])
    assert "design" in capsys.readouterr().out
    (tmp_path / "train_log.jsonl").write_text(json.dumps({"step": 1, "loss": 2.0}) + "\n")
    main(["train-report", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["first_loss"] == 2.0


def test_commands_that_need_a_server_fail_cleanly_without_one(tmp_path):
    config = tmp_path / "unreachable.yaml"
    config.write_text("serve:\n  port: 9\n")
    with pytest.raises(SystemExit, match="No vLLM server"):
        main(["--config", str(config), "bench", "--mode", "direct", "--limit", "1"])
