"""The real ChatOpenAI client against a local OpenAI-compatible fake server.

This is the part of the integration that can be checked without a GPU: what actually goes over
the wire (guided JSON schema, tools, sampling params) and how replies are parsed.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from harness.config import Config
from harness.pipeline import artifacts
from harness.pipeline.chat_models import ChatModelFactory
from harness.pipeline.dependencies import PipelineDependencies
from harness.pipeline.graph import CUT_OFF_REASON, build_pipeline_graph, call_structured
from harness.prompt_context import PromptBuilder
from harness.serving import server_ready
from tests.fakes import REQUIREMENTS


class FakeVLLM:
    """Records requests; replies with whatever `reply` returns for the request body."""

    def __init__(self, reply):
        self.requests: list[dict] = []
        fake_server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *arguments):
                pass

            def do_GET(self):
                self._send(200, {"object": "list", "data": [{"id": "harness", "object": "model"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake_server.requests.append(body)
                message, finish = reply(body)
                self._send(200, {
                    "id": "c1", "object": "chat.completion", "created": 1, "model": body["model"],
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})

            def _send(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def config():
    test_config = Config()
    test_config.serve.max_retries = 0
    return test_config


def dependencies_for(config, server):
    config.serve.port = server.port
    return PipelineDependencies(config, ChatModelFactory(config), PromptBuilder(config.context))


def test_structured_output_sends_a_json_schema_and_parses_the_reply(config):
    server = FakeVLLM(lambda request_body: ({"role": "assistant", "content": REQUIREMENTS.model_dump_json()}, "stop"))
    try:
        assert server_ready(f"http://127.0.0.1:{server.port}/v1")
        parsed, _raw, error = call_structured(
            dependencies_for(config, server), artifacts.Requirements, [("system", "be brief"), ("user", "triple it")], 0.3, None)
    finally:
        server.close()
    assert error is None and parsed.requirements[0] == "f(x) returns x tripled"
    body = server.requests[0]
    assert body["model"] == "harness" and body["temperature"] == 0.3
    response_format = body["response_format"]  # this is what vLLM turns into guided decoding
    assert response_format["type"] == "json_schema" and response_format["json_schema"]["name"] == "Requirements"
    assert "requirements" in response_format["json_schema"]["schema"]["properties"]
    assert body.get("max_tokens") == config.sampling.max_new_tokens or \
        body.get("max_completion_tokens") == config.sampling.max_new_tokens
    assert body["top_p"] == config.sampling.top_p


def test_truncated_structured_output_is_reported_as_cut_off(config):
    server = FakeVLLM(lambda request_body: ({"role": "assistant", "content": '{"requirements": ["a", "b'}, "length"))
    try:
        parsed, raw_reply, error = call_structured(
            dependencies_for(config, server), artifacts.Requirements, [("user", "x")], 0.3, None)
    finally:
        server.close()
    assert parsed is None and error == CUT_OFF_REASON
    assert raw_reply.content == '{"requirements": ["a", "b'  # the partial text is kept for the record


def test_native_tool_calls_are_sent_and_parsed(config):
    import tempfile

    from langchain_core.tools import StructuredTool

    from harness.pipeline.tool_adapters import to_langchain_tools
    from harness.tools import Workspace, build_tool_registry

    call = {"id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "a.py", "start_line": 3})}}
    server = FakeVLLM(lambda request_body: ({"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"))
    with tempfile.TemporaryDirectory() as directory:
        tools = to_langchain_tools(build_tool_registry(Workspace(directory)))
        assert all(isinstance(t, StructuredTool) for t in tools)
        try:
            chat_model = ChatModelFactory(_with_port(config, server)).get(0.2).bind_tools(tools)
            reply = chat_model.invoke([("system", "s"), ("user", "u")])
        finally:
            server.close()
    assert reply.tool_calls == [{"name": "read_file", "args": {"path": "a.py", "start_line": 3},
                              "id": "call_1", "type": "tool_call"}]
    sent = {tool["function"]["name"]: tool["function"] for tool in server.requests[0]["tools"]}
    assert {"read_file", "grep", "str_replace", "apply_patch"} <= set(sent)
    properties = sent["read_file"]["parameters"]["properties"]
    assert set(properties) == {"path", "start_line", "end_line"} and sent["read_file"]["parameters"]["required"] == ["path"]
    assert "state" not in properties and "registry" not in json.dumps(sent)  # nothing internal leaks


def _with_port(config, server):
    config.serve.port = server.port
    return config


def test_adapter_models_use_the_adapter_name(config):
    server = FakeVLLM(lambda request_body: ({"role": "assistant", "content": "hi"}, "stop"))
    try:
        config.serve.port = server.port
        ChatModelFactory(config, adapter_path="/some/adapter").get(0.1).invoke("hello")
    finally:
        server.close()
    assert server.requests[0]["model"] == "adapter"


def test_unreachable_server_is_detected():
    assert not server_ready("http://127.0.0.1:9/v1", timeout_seconds=0.5)


def test_a_truncated_stage_fails_the_run_with_a_reason_and_keeps_the_partial_output(config):
    server = FakeVLLM(lambda request_body: ({"role": "assistant", "content": '{"requirements": ["a'}, "length"))
    try:
        config.serve.port = server.port
        graph = build_pipeline_graph(dependencies_for(config, server))
        state = graph.invoke({"task": "t"})
    finally:
        server.close()
    assert state["failure_reason"] == f"requirements: {CUT_OFF_REASON}"
    assert state["llm_calls"][0]["completion"] == '{"requirements": ["a' and "design" not in state
