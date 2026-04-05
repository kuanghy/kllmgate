"""管线与路由选择的单元测试"""

import pytest

from kllmgate.models import ProtocolFormat, ProviderConfig
from kllmgate.errors import ConfigError
from kllmgate.pipeline import (
    get_tool_adapter,
    get_converter,
    process_request,
)
from kllmgate.tools.standard import StandardToolAdapter
from kllmgate.tools.minimax_xml import MinimaxXmlToolAdapter
from kllmgate.tools.anthropic import AnthropicToolAdapter
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

    def test_anthropic_same_protocol_standard_is_passthrough(self):
        adapter = StandardToolAdapter()
        conv = get_converter(
            PF.ANTHROPIC_MESSAGES, PF.ANTHROPIC_MESSAGES, adapter,
        )
        assert isinstance(conv, PassthroughConverter)


class TestProcessRequest:

    @pytest.mark.asyncio
    async def test_invalid_model_format_raises(self):
        with pytest.raises(ConfigError, match="provider/model"):
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

        async def _fake_stream(body):
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

        async def _failing_stream(body):
            raise UpstreamHTTPError(401, "Unauthorized")
            yield

        class _Client:
            def send_stream(self, body):
                return _failing_stream(body)

        with pytest.raises(UpstreamHTTPError):
            await process_request(
                PF.OPENAI_CHAT,
                {"model": "test/gpt-4", "stream": True, "messages": [
                    {"role": "user", "content": "hello"},
                ]},
                providers,
                {"test": _Client()},
            )
