"""管线与路由选择的单元测试"""

import json

import pytest

from kllmgate.models import ProtocolFormat, ProviderConfig
from kllmgate.errors import ConfigError
from kllmgate.pipeline import (
    get_tool_adapter,
    get_converter,
    process_request,
    resolve_provider_and_model,
)
from kllmgate.toolcall.standard import StandardToolAdapter
from kllmgate.toolcall.minimax_xml import MinimaxXmlToolAdapter
from kllmgate.toolcall.anthropic import AnthropicToolAdapter
from kllmgate.converters.passthrough import PassthroughConverter
from kllmgate.converters.openai_responses_to_openai_chat import (
    OpenaiResponsesToOpenaiChatConverter,
)
from kllmgate.converters.openai_chat_tool_adapt import (
    OpenaiChatToolAdaptConverter,
)

PF = ProtocolFormat


def _cfg(**overrides) -> ProviderConfig:
    defaults = {
        "name": "test",
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test",
        "protocol": "openai",
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


class TestGetToolAdapter:

    def test_standard_openai(self):
        cfg = _cfg(protocol="openai", tool_style="standard")
        adapter = get_tool_adapter(cfg)
        assert isinstance(adapter, StandardToolAdapter)

    def test_minimax_xml(self):
        cfg = _cfg(protocol="openai", tool_style="minimax_xml")
        adapter = get_tool_adapter(cfg)
        assert isinstance(adapter, MinimaxXmlToolAdapter)

    def test_anthropic_always_anthropic_adapter(self):
        cfg = _cfg(
            protocol="anthropic", wire_api="messages",
            tool_style="standard",
        )
        adapter = get_tool_adapter(cfg)
        assert isinstance(adapter, AnthropicToolAdapter)

    def test_unknown_tool_style_raises(self):
        cfg = _cfg(protocol="openai", tool_style="unknown")
        with pytest.raises(ConfigError, match="tool_style"):
            get_tool_adapter(cfg)


class TestGetConverter:

    def test_same_protocol_standard_is_passthrough(self):
        adapter = StandardToolAdapter()
        conv = get_converter(PF.OPENAI_CHAT, PF.OPENAI_CHAT, adapter)
        assert isinstance(conv, PassthroughConverter)

    def test_same_protocol_nonstandard_is_tool_adapt(self):
        adapter = MinimaxXmlToolAdapter()
        conv = get_converter(PF.OPENAI_CHAT, PF.OPENAI_CHAT, adapter)
        assert isinstance(conv, OpenaiChatToolAdaptConverter)

    def test_cross_protocol_converter(self):
        adapter = StandardToolAdapter()
        conv = get_converter(PF.OPENAI_RESPONSES, PF.OPENAI_CHAT, adapter)
        assert isinstance(conv, OpenaiResponsesToOpenaiChatConverter)

    def test_anthropic_same_protocol_with_anthropic_adapter_is_passthrough(
        self,
    ):
        adapter = AnthropicToolAdapter()
        conv = get_converter(
            PF.ANTHROPIC_MESSAGES, PF.ANTHROPIC_MESSAGES, adapter,
        )
        assert isinstance(conv, PassthroughConverter)


class TestResolveProviderAndModel:

    def test_provider_slash_model(self):
        p, m = resolve_provider_and_model("scnet/MiniMax-M2.5", None, {})
        assert (p, m) == ("scnet", "MiniMax-M2.5")

    def test_model_alias(self):
        aliases = {"MiniMax-M2.5": "scnet/MiniMax-M2.5"}
        p, m = resolve_provider_and_model("MiniMax-M2.5", None, aliases)
        assert (p, m) == ("scnet", "MiniMax-M2.5")

    def test_header_provider(self):
        p, m = resolve_provider_and_model("MiniMax-M2.5", "scnet", {})
        assert (p, m) == ("scnet", "MiniMax-M2.5")

    def test_slash_takes_priority_over_header(self):
        p, m = resolve_provider_and_model(
            "openai/gpt-4.1", "scnet", {},
        )
        assert (p, m) == ("openai", "gpt-4.1")

    def test_alias_takes_priority_over_header(self):
        aliases = {"gpt-4": "openai/gpt-4"}
        p, m = resolve_provider_and_model("gpt-4", "scnet", aliases)
        assert (p, m) == ("openai", "gpt-4")

    def test_bare_model_no_header_raises(self):
        with pytest.raises(ConfigError, match="cannot determine provider"):
            resolve_provider_and_model("MiniMax-M2.5", None, {})

    def test_empty_model_raises(self):
        with pytest.raises(ConfigError, match="required"):
            resolve_provider_and_model("", None, {})

    def test_multi_segment_model(self):
        p, m = resolve_provider_and_model("org/ns/model-v2", None, {})
        assert (p, m) == ("org", "ns/model-v2")

    def test_default_provider_fallback(self):
        p, m = resolve_provider_and_model(
            "gpt-4", None, {}, default_provider="openai",
        )
        assert (p, m) == ("openai", "gpt-4")

    def test_header_takes_priority_over_default_provider(self):
        p, m = resolve_provider_and_model(
            "gpt-4", "scnet", {}, default_provider="openai",
        )
        assert (p, m) == ("scnet", "gpt-4")

    def test_alias_takes_priority_over_default_provider(self):
        aliases = {"gpt-4": "anthropic/gpt-4"}
        p, m = resolve_provider_and_model(
            "gpt-4", None, aliases, default_provider="openai",
        )
        assert (p, m) == ("anthropic", "gpt-4")

    def test_bare_model_no_default_provider_raises(self):
        with pytest.raises(ConfigError, match="cannot determine provider"):
            resolve_provider_and_model(
                "gpt-4", None, {}, default_provider=None,
            )


class TestProcessRequest:

    @pytest.mark.asyncio
    async def test_invalid_model_format_raises(self):
        with pytest.raises(ConfigError, match="cannot determine provider"):
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "no-slash"},
                {},
                {},
            )

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(ConfigError, match="unknown provider"):
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "nonexistent/gpt-4"},
                {},
                {},
            )

    @pytest.mark.asyncio
    async def test_model_not_in_whitelist_raises(self):
        providers = {
            "test": _cfg(models=["gpt-4.1"]),
        }
        with pytest.raises(ConfigError, match="not supported"):
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "test/gpt-3.5"},
                providers,
                {},
            )

    @pytest.mark.asyncio
    async def test_missing_client_raises(self):
        providers = {"test": _cfg()}
        with pytest.raises(ConfigError, match="not initialized"):
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "test/gpt-4", "messages": []},
                providers,
                {},
            )

    @pytest.mark.asyncio
    async def test_non_stream_success(self, httpx_mock):
        from kllmgate.upstream.client import UpstreamClient

        providers = {"test": _cfg()}
        client = UpstreamClient(providers["test"])
        clients = {"test": client}

        httpx_mock.add_response(json={
            "id": "chatcmpl-1",
            "choices": [{
                "message": {"content": "hi"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        })

        resp = await process_request(
            PF.OPENAI_CHAT,
            {"model": "test/gpt-4", "messages": [
                {"role": "user", "content": "hello"},
            ]},
            providers,
            clients,
        )
        assert resp.status_code == 200
        await client.close()

    @pytest.mark.asyncio
    async def test_stream_returns_streaming_response(self):
        """验证流式请求返回 StreamingResponse（不实际发 HTTP）"""
        from unittest.mock import MagicMock

        providers = {"test": _cfg()}

        async def _fake_stream(body, extra_headers=None, target=None):
            return
            yield

        mock_client = MagicMock()
        mock_client.send_stream = _fake_stream
        clients = {"test": mock_client}

        resp = await process_request(
            PF.OPENAI_CHAT,
            {"model": "test/gpt-4", "stream": True, "messages": [
                {"role": "user", "content": "hello"},
            ]},
            providers,
            clients,
        )
        from starlette.responses import StreamingResponse
        assert isinstance(resp, StreamingResponse)

    @pytest.mark.asyncio
    async def test_stream_preflight_error_raises_before_response(self):
        from kllmgate.errors import UpstreamHTTPError

        providers = {"test": _cfg()}

        async def _failing_stream(body, extra_headers=None, target=None):
            raise UpstreamHTTPError(401, "Unauthorized")
            yield

        class _Client:
            def send_stream(self, body, extra_headers=None, target=None):
                return _failing_stream(
                    body, extra_headers=extra_headers, target=target,
                )

        with pytest.raises(UpstreamHTTPError):
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "test/gpt-4", "stream": True, "messages": [
                    {"role": "user", "content": "hello"},
                ]},
                providers,
                {"test": _Client()},
            )

    @pytest.mark.asyncio
    async def test_strip_inbound_system_prompt_openai_chat(self, httpx_mock):
        from kllmgate.upstream.client import UpstreamClient

        providers = {"test": _cfg(strip_system_prompt=True)}
        client = UpstreamClient(providers["test"])
        clients = {"test": client}

        httpx_mock.add_response(json={
            "id": "chatcmpl-1",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        })

        req_body = {
            "model": "test/gpt-4",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "developer", "content": "dev"},
                {"role": "user", "content": "hello"},
            ],
        }
        await process_request(
            PF.OPENAI_CHAT, req_body, providers, clients,
        )
        assert len(req_body["messages"]) == 1
        assert req_body["messages"][0]["role"] == "user"
        await client.close()

    @pytest.mark.asyncio
    async def test_strip_inbound_system_prompt_openai_responses(self, httpx_mock):
        from kllmgate.upstream.client import UpstreamClient

        providers = {"test": _cfg(
            protocol="openai", wire_api="responses",
            strip_system_prompt=True,
        )}
        client = UpstreamClient(providers["test"])
        clients = {"test": client}

        httpx_mock.add_response(json={
            "id": "resp-1",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        })

        req_body = {
            "model": "test/gpt-4",
            "instructions": "sys desc",
            "input": [
                {"role": "system", "content": "sys"},
                {"role": "developer", "content": "dev"},
                {"role": "user", "content": "hello"},
                "direct text",
            ],
        }
        await process_request(
            PF.OPENAI_RESPONSES, req_body, providers, clients,
        )
        assert "instructions" not in req_body
        assert len(req_body["input"]) == 2
        assert req_body["input"][0].get("role") == "user"
        assert req_body["input"][1] == "direct text"
        await client.close()

    @pytest.mark.asyncio
    async def test_non_json_upstream_raises_conversion_error(
        self, httpx_mock,
    ):
        from kllmgate.errors import ConversionError
        from kllmgate.upstream.client import UpstreamClient

        providers = {"test": _cfg()}
        client = UpstreamClient(providers["test"])
        clients = {"test": client}

        httpx_mock.add_response(
            status_code=200,
            text="not-json-at-all",
            headers={"content-type": "text/plain"},
        )

        with pytest.raises(ConversionError) as exc_info:
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "test/gpt-4", "messages": [
                    {"role": "user", "content": "hello"},
                ]},
                providers,
                clients,
            )
        assert exc_info.value.status_code == 502
        assert exc_info.value.error_type == "conversion_error"
        assert "JSON" in exc_info.value.message or "json" in (
            exc_info.value.message.lower()
        )
        await client.close()

    @pytest.mark.asyncio
    async def test_invalid_utf8_upstream_raises_conversion_error(
        self, httpx_mock,
    ):
        from kllmgate.errors import ConversionError
        from kllmgate.upstream.client import UpstreamClient

        providers = {"test": _cfg()}
        client = UpstreamClient(providers["test"])
        clients = {"test": client}

        httpx_mock.add_response(
            status_code=200,
            content=b"\xff\xfe not utf-8",
            headers={"content-type": "application/json"},
        )

        with pytest.raises(ConversionError) as exc_info:
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "test/gpt-4", "messages": [
                    {"role": "user", "content": "hello"},
                ]},
                providers,
                clients,
            )
        assert exc_info.value.status_code == 502
        assert exc_info.value.error_type == "conversion_error"
        assert exc_info.value.code == "invalid_upstream_json"
        await client.close()

    @pytest.mark.asyncio
    async def test_unexpected_exception_wrapped_as_internal_error(self):
        from kllmgate.errors import InternalError

        providers = {"test": _cfg()}

        class _BoomClient:
            async def send(self, body, extra_headers=None, target=None):
                raise RuntimeError("simulated bug with secret path")

        with pytest.raises(InternalError) as exc_info:
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "test/gpt-4", "messages": [
                    {"role": "user", "content": "hello"},
                ]},
                providers,
                {"test": _BoomClient()},
            )
        err = exc_info.value
        assert err.status_code == 500
        assert err.error_type == "server_error"
        assert err.code == "internal_error"
        assert err.message == "internal server error"
        assert "secret" not in err.message
        assert "simulated bug" not in err.message


class TestMakeStreamErrorEvents:

    def test_openai_chat_includes_error_signal(self):
        from kllmgate.pipeline import _make_stream_error_events

        events = _make_stream_error_events(
            PF.OPENAI_CHAT, "upstream boom",
        )
        assert len(events) >= 2
        assert events[-1] == "data: [DONE]\n\n"
        payload = json.loads(events[0].removeprefix("data: ").strip())
        assert payload["choices"][0]["finish_reason"] == "error"
        assert payload["error"]["message"] == "upstream boom"
        assert payload["error"]["code"] == "stream_interrupted"

    def test_openai_responses_incomplete_status(self):
        from kllmgate.pipeline import _make_stream_error_events

        events = _make_stream_error_events(
            PF.OPENAI_RESPONSES, "upstream boom",
        )
        assert len(events) == 1
        assert "incomplete" in events[0]
        assert "stream_interrupted" in events[0]


class TestLoggedStreamDisconnect:

    @pytest.mark.asyncio
    async def test_logs_when_client_disconnected(self, caplog):
        import logging
        from unittest.mock import AsyncMock, MagicMock

        from kllmgate.pipeline import _logged_stream

        closed = {"done": False}

        async def _chunks():
            try:
                yield "data: one\n\n"
                yield "data: two\n\n"
            finally:
                closed["done"] = True

        http_request = MagicMock()
        http_request.is_disconnected = AsyncMock(
            side_effect=[False, True],
        )

        with caplog.at_level(logging.INFO, logger="kllmgate.pipeline"):
            out = [
                chunk async for chunk in _logged_stream(
                    _chunks(),
                    "req123",
                    PF.OPENAI_CHAT,
                    "test",
                    "gpt-4",
                    {},
                    0.0,
                    http_request=http_request,
                )
            ]

        assert out == ["data: one\n\n"]
        assert closed["done"] is True
        assert any(
            "Client disconnected mid-stream" in r.message
            and "req123" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_unexpected_stream_error_sanitized_but_logged(
        self, caplog,
    ):
        import logging

        from kllmgate.pipeline import _logged_stream

        async def _chunks():
            yield "data: partial\n\n"
            raise RuntimeError("secret stack detail")

        with caplog.at_level(logging.ERROR, logger="kllmgate.pipeline"):
            out = [
                chunk async for chunk in _logged_stream(
                    _chunks(),
                    "req456",
                    PF.OPENAI_CHAT,
                    "test",
                    "gpt-4",
                    {},
                    0.0,
                )
            ]

        full = "".join(out)
        assert "secret stack detail" not in full
        assert "internal server error" in full
        assert any(
            "secret stack detail" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.ERROR
        )

    @pytest.mark.asyncio
    async def test_upstream_stream_error_message_still_forwarded(self):
        from kllmgate.errors import UpstreamError
        from kllmgate.pipeline import _logged_stream

        async def _chunks():
            raise UpstreamError(
                "upstream timeout after retries",
                code="upstream_request_error",
            )
            yield  # pragma: no cover

        out = [
            chunk async for chunk in _logged_stream(
                _chunks(),
                "req789",
                PF.OPENAI_CHAT,
                "test",
                "gpt-4",
                {},
                0.0,
            )
        ]
        full = "".join(out)
        assert "upstream timeout after retries" in full

    @pytest.mark.asyncio
    async def test_conversion_stream_error_message_still_forwarded(self):
        from kllmgate.errors import ConversionError
        from kllmgate.pipeline import _logged_stream

        async def _chunks():
            raise ConversionError(
                "malformed tool call xml",
                code="malformed_tool_call",
            )
            yield  # pragma: no cover

        out = [
            chunk async for chunk in _logged_stream(
                _chunks(),
                "req-conv",
                PF.OPENAI_CHAT,
                "test",
                "gpt-4",
                {},
                0.0,
            )
        ]
        full = "".join(out)
        assert "malformed tool call xml" in full
        assert "internal server error" not in full


class TestAutoProtocolProcessRequest:

    def _dual_cfg(self, **overrides):
        from kllmgate.models import ProtocolEndpointConfig

        defaults = {
            "name": "dual",
            "protocol": "auto",
            "api_key": "sk-top",
            "openai": ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
                wire_api="chat",
            ),
            "anthropic": ProtocolEndpointConfig(
                base_url="https://anthropic.example.com",
                wire_api="messages",
            ),
        }
        defaults.update(overrides)
        return ProviderConfig(**defaults)

    @pytest.mark.asyncio
    async def test_openai_inbound_posts_to_openai_url(self, httpx_mock):
        from kllmgate.upstream.client import UpstreamClient

        provider = self._dual_cfg()
        client = UpstreamClient(provider)
        httpx_mock.add_response(json={
            "id": "chatcmpl-1",
            "choices": [{
                "message": {"content": "hi"},
                "finish_reason": "stop",
            }],
            "usage": {},
        })

        await process_request(
            PF.OPENAI_CHAT,
            {"model": "dual/m1", "messages": [
                {"role": "user", "content": "hello"},
            ]},
            {"dual": provider},
            {"dual": client},
        )
        request = httpx_mock.get_request()
        assert str(request.url) == (
            "https://openai.example.com/v1/chat/completions"
        )
        assert request.headers["Authorization"] == "Bearer sk-top"
        await client.close()

    @pytest.mark.asyncio
    async def test_openai_stream_posts_to_openai_url(self, httpx_mock):
        from kllmgate.upstream.client import UpstreamClient
        from starlette.responses import StreamingResponse

        provider = self._dual_cfg()
        client = UpstreamClient(provider)
        sse_body = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"choices":[{"index":0,"delta":{"content":"hi"},'
            '"finish_reason":null}]}\n\n'
            "data: [DONE]\n\n"
        )
        httpx_mock.add_response(
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )

        resp = await process_request(
            PF.OPENAI_CHAT,
            {
                "model": "dual/m1",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "hello"},
                ],
            },
            {"dual": provider},
            {"dual": client},
        )
        assert isinstance(resp, StreamingResponse)
        chunks = [chunk async for chunk in resp.body_iterator]
        assert chunks
        request = httpx_mock.get_request()
        assert str(request.url) == (
            "https://openai.example.com/v1/chat/completions"
        )
        assert request.headers["Authorization"] == "Bearer sk-top"
        await client.close()

    @pytest.mark.asyncio
    async def test_anthropic_stream_posts_to_anthropic_url(
        self, httpx_mock,
    ):
        from kllmgate.upstream.client import UpstreamClient
        from starlette.responses import StreamingResponse

        provider = self._dual_cfg()
        client = UpstreamClient(provider)
        sse_body = (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"id":"msg_1",'
            '"type":"message","role":"assistant","content":[],'
            '"model":"m1","stop_reason":null,"usage":'
            '{"input_tokens":1,"output_tokens":0}}}\n\n'
            "event: content_block_start\n"
            'data: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"text","text":""}}\n\n'
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"hi"}}\n\n'
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n\n'
        )
        httpx_mock.add_response(
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )

        resp = await process_request(
            PF.ANTHROPIC_MESSAGES,
            {
                "model": "dual/m1",
                "stream": True,
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "hello"},
                ],
            },
            {"dual": provider},
            {"dual": client},
        )
        assert isinstance(resp, StreamingResponse)
        chunks = [chunk async for chunk in resp.body_iterator]
        assert chunks
        request = httpx_mock.get_request()
        assert str(request.url) == (
            "https://anthropic.example.com/v1/messages"
        )
        assert request.headers["x-api-key"] == "sk-top"
        await client.close()

    @pytest.mark.asyncio
    async def test_anthropic_inbound_posts_to_anthropic_url(
        self, httpx_mock,
    ):
        from kllmgate.upstream.client import UpstreamClient

        provider = self._dual_cfg()
        client = UpstreamClient(provider)
        httpx_mock.add_response(json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": "m1",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

        await process_request(
            PF.ANTHROPIC_MESSAGES,
            {
                "model": "dual/m1",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "hello"},
                ],
            },
            {"dual": provider},
            {"dual": client},
        )
        request = httpx_mock.get_request()
        assert str(request.url) == (
            "https://anthropic.example.com/v1/messages"
        )
        assert request.headers["x-api-key"] == "sk-top"
        await client.close()

    @pytest.mark.asyncio
    async def test_missing_subsection_raises(self):
        from kllmgate.models import ProtocolEndpointConfig

        provider = ProviderConfig(
            name="dual",
            protocol="auto",
            api_key="sk-top",
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
            ),
        )
        with pytest.raises(ConfigError, match="anthropic"):
            await process_request(
                PF.ANTHROPIC_MESSAGES,
                {
                    "model": "dual/m1",
                    "max_tokens": 16,
                    "messages": [
                        {"role": "user", "content": "hello"},
                    ],
                },
                {"dual": provider},
                {"dual": object()},
            )

    @pytest.mark.asyncio
    async def test_responses_inbound_converts_to_chat_wire(
        self, httpx_mock,
    ):
        from kllmgate.upstream.client import UpstreamClient

        provider = self._dual_cfg()
        client = UpstreamClient(provider)
        httpx_mock.add_response(json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        })

        resp = await process_request(
            PF.OPENAI_RESPONSES,
            {
                "model": "dual/m1",
                "input": "hello",
            },
            {"dual": provider},
            {"dual": client},
        )
        assert resp.status_code == 200
        request = httpx_mock.get_request()
        assert str(request.url) == (
            "https://openai.example.com/v1/chat/completions"
        )
        await client.close()

    @pytest.mark.asyncio
    async def test_models_override_per_protocol(self):
        from kllmgate.models import ProtocolEndpointConfig

        provider = ProviderConfig(
            name="dual",
            protocol="auto",
            api_key="sk-top",
            models=["shared"],
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
                models=["openai-only"],
            ),
            anthropic=ProtocolEndpointConfig(
                base_url="https://anthropic.example.com",
            ),
        )
        with pytest.raises(ConfigError, match="not supported"):
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "dual/shared", "messages": [
                    {"role": "user", "content": "hello"},
                ]},
                {"dual": provider},
                {"dual": object()},
            )
        with pytest.raises(ConfigError, match="not supported"):
            await process_request(
                PF.ANTHROPIC_MESSAGES,
                {
                    "model": "dual/openai-only",
                    "max_tokens": 16,
                    "messages": [
                        {"role": "user", "content": "hello"},
                    ],
                },
                {"dual": provider},
                {"dual": object()},
            )

    @pytest.mark.asyncio
    async def test_empty_models_list_rejects_all_models(self):
        from kllmgate.models import ProtocolEndpointConfig

        provider = ProviderConfig(
            name="dual",
            protocol="auto",
            api_key="sk-top",
            models=["shared"],
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
                models=[],
            ),
        )
        with pytest.raises(ConfigError, match="not supported"):
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "dual/shared", "messages": [
                    {"role": "user", "content": "hello"},
                ]},
                {"dual": provider},
                {"dual": object()},
            )
