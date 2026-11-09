"""Start, check and stop the vLLM OpenAI-compatible server the pipeline talks to."""
from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from harness.config import Config
from harness.pipeline.chat_models import ADAPTER_MODEL_NAME

MIN_LORA_RANK_SLOTS = 16
READY_POLL_SECONDS = 3
STOP_POLL_SECONDS = 0.5


class ServerNotRunning(RuntimeError):
    """No vLLM server is reachable (an expected, user-fixable condition)."""


def build_server_command(config: Config, adapter_path: str | None = None) -> list[str]:
    serve, vllm = config.serve, config.vllm
    command = [
        serve.vllm_executable, "serve", config.model.path,
        "--served-model-name", serve.served_model_name,
        "--host", serve.host, "--port", str(serve.port),
        "--dtype", vllm.dtype,
        "--tensor-parallel-size", str(vllm.tensor_parallel_size),
        "--max-model-len", str(vllm.max_model_len),
        "--gpu-memory-utilization", str(vllm.gpu_memory_utilization),
        # native tool calling for Qwen2.5: <tool_call> JSON -> OpenAI tool_calls
        "--enable-auto-tool-choice", "--tool-call-parser", serve.tool_call_parser,
    ]
    if vllm.enforce_eager:
        command.append("--enforce-eager")
    if adapter_path:
        command += ["--enable-lora", "--max-lora-rank", str(max(config.lora.rank, MIN_LORA_RANK_SLOTS)),
                    "--lora-modules", f"{ADAPTER_MODEL_NAME}={adapter_path}"]
    return command


def server_ready(base_url: str, timeout_seconds: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout_seconds) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def require_server(config: Config) -> None:
    if not server_ready(config.serve.base_url):
        raise ServerNotRunning(
            f"No vLLM server at {config.serve.base_url}. Start one first: "
            "`harness serve --background` (add --adapter DIR to serve a trained LoRA).")


def wait_until_ready(base_url: str, timeout_seconds: float = 900.0,
                     process: subprocess.Popen | None = None) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server_ready(base_url):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {process.returncode}; see its log.")
        time.sleep(READY_POLL_SECONDS)
    raise TimeoutError(f"vLLM server not ready after {timeout_seconds:.0f}s at {base_url}")


def pid_file_for(log_path: Path) -> Path:
    return log_path.with_suffix(".pid")


def start_in_background(config: Config, adapter_path: str | None, log_path: Path) -> subprocess.Popen:
    """Start the server detached (for notebooks): output goes to log_path, the pid to <log>.pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        process = subprocess.Popen(build_server_command(config, adapter_path), stdout=log_file,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    pid_file_for(log_path).write_text(str(process.pid))
    return process


def _process_alive(pid: int) -> bool:
    """True if the process is running. A child of ours that already exited is a zombie, which
    still answers kill(pid, 0), so it is reaped first."""
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False  # it was our child and has exited
    except ChildProcessError:
        pass  # not our child (the normal case: started by another CLI process)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_server(log_path: Path, timeout_seconds: float = 60.0) -> bool:
    """Stop a server started by start_in_background and wait for it to release the GPUs.

    Training and the vLLM server cannot share the GPUs, so this must finish before `train`.
    Returns False if no server was recorded.
    """
    pid_file = pid_file_for(log_path)
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text())
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)  # the whole group: vLLM spawns workers
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return True
    deadline = time.monotonic() + timeout_seconds
    while _process_alive(pid) and time.monotonic() < deadline:
        time.sleep(STOP_POLL_SECONDS)
    if _process_alive(pid):  # did not exit in time
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # exited between the check and the kill
    pid_file.unlink(missing_ok=True)
    return True
