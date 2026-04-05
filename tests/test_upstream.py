"""上游客户端的单元测试"""

import json

import httpx
import pytest

from kllmgate.models import ProviderConfig
from kllmgate.upstream.client import UpstreamClient
from kllmgate.errors import UpstreamError, UpstreamHTTPError


def _make_config(**overrides) -> ProviderConfig:
    defaults = {
        "name": "test",
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test",
        "protocol": "openai",
        "wire_api": "chat",
        "timeout_seconds": 10,
        "max_retries": 1,
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


class TestUpstreamClientEndpoint:

    def test_openai_chat_endpoint(self):
        cfg = _make_config(protocol="openai", wire_api="chat")
        client = UpstreamClient(cfg)
        assert client._endpoint == "https://api.example.com/v1/chat/completions"

    def test_openai_responses_endpoint(self):
        cfg = _make_config(protocol="openai", wire_api="responses")
        client = UpstreamClient(cfg)
        assert client._endpoint == "https://api.example.com/v1/responses"

    def test_anthropic_messages_endpoint(self):
        cfg = _make_config(
            protocol="anthropic",
            wire_api="messages",
            base_url="https://api.anthropic.com",
        )
        client = UpstreamClient(cfg)
        assert client._endpoint == "https://api.anthropic.com/v1/messages"


class TestUpstreamClientHeaders:

    def test_openai_auth_header(self):
        cfg = _make_config(protocol="openai")
        client = UpstreamClient(cfg)
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert "x-api-key" not in headers

    def test_anthropic_auth_header(self):
        cfg = _make_config(
            protocol="anthropic", wire_api="messages",
            base_url="https://api.anthropic.com",
        )
        client = UpstreamClient(cfg)
        headers = client._build_headers()
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in headers


class TestUpstreamClientExtraHeaders:

    def test_anthropic_extra_headers_merged(self):
        cfg = _make_config(
            protocol="anthropic", wire_api="messages",
            base_url="https://api.anthropic.com",
        )
        client = UpstreamClient(cfg)
        headers = client._build_headers(
            extra_headers={"anthropic-beta": "output-128k-2025-02-19"},
        )
        assert headers["anthropic-beta"] == "output-128k-2025-02-19"
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == "2023-06-01"

    def test_anthropic_extra_headers_override_version(self):
        cfg = _make_config(
            protocol="anthropic", wire_api="messages",
            base_url="https://api.anthropic.com",
        )
        client = UpstreamClient(cfg)
        headers = client._build_headers(
            extra_headers={"anthropic-version": "2024-10-22"},
        )
        assert headers["anthropic-version"] == "2024-10-22"

    def test_openai_ignores_extra_headers(self):
        cfg = _make_config(protocol="openai")
        client = UpstreamClient(cfg)
        headers = client._build_headers(
            extra_headers={"anthropic-beta": "some-beta"},
        )
        assert "anthropic-beta" not in headers
        assert headers["Authorization"] == "Bearer sk-test"

    def test_anthropic_none_extra_headers(self):
        cfg = _make_config(
            protocol="anthropic", wire_api="messages",
            base_url="https://api.anthropic.com",
        )
        client = UpstreamClient(cfg)
        headers = client._build_headers(extra_headers=None)
        assert headers["anthropic-version"] == "2023-06-01"
        assert "anthropic-beta" not in headers


class TestUpstreamClientSend:

    @pytest.mark.asyncio
    async def test_send_success(self, httpx_mock):
        cfg = _make_config()
        client = UpstreamClient(cfg)

        response_body = {
            "id": "chatcmpl-1",
            "choices": [{"message": {"content": "hi"}}],
        }
        httpx_mock.add_response(json=response_body)

        result = await client.send({"model": "test", "messages": []})
        assert result["id"] == "chatcmpl-1"
        await client.close()

    @pytest.mark.asyncio
    async def test_send_http_error(self, httpx_mock):
        cfg = _make_config(max_retries=0)
        client = UpstreamClient(cfg)

        httpx_mock.add_response(status_code=401, text="Unauthorized")

        with pytest.raises(UpstreamHTTPError) as exc_info:
            await client.send({"model": "test"})
        assert exc_info.value.upstream_status == 401
        await client.close()

    @pytest.mark.asyncio
    async def test_send_retryable_status(self, httpx_mock):
        cfg = _make_config(max_retries=1)
        client = UpstreamClient(cfg)

        httpx_mock.add_response(status_code=429, text="rate limited")
        httpx_mock.add_response(
            json={"id": "chatcmpl-2", "choices": []},
        )

        result = await client.send({"model": "test"})
        assert result["id"] == "chatcmpl-2"
        await client.close()

    @pytest.mark.asyncio
    async def test_send_non_retryable_status(self, httpx_mock):
        cfg = _make_config(max_retries=2)
        client = UpstreamClient(cfg)

        httpx_mock.add_response(status_code=400, text="bad request")

        with pytest.raises(UpstreamHTTPError) as exc_info:
            await client.send({"model": "test"})
        assert exc_info.value.upstream_status == 400
        await client.close()


class TestUpstreamClientStream:

    @pytest.mark.asyncio
    async def test_send_stream_success(self, httpx_mock):
        cfg = _make_config()
        client = UpstreamClient(cfg)

        sse_body = (
            'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        httpx_mock.add_response(
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )

        events = []
        async for ev in client.send_stream(
            {"model": "test", "stream": True}
        ):
            events.append(ev)

        assert len(events) == 2
        assert '"hi"' in events[0].data
        assert events[1].data == "[DONE]"
        await client.close()

    @pytest.mark.asyncio
    async def test_send_stream_http_error(self, httpx_mock):
        cfg = _make_config(max_retries=0)
        client = UpstreamClient(cfg)

        httpx_mock.add_response(status_code=500, text="server error")

        with pytest.raises(UpstreamHTTPError):
            async for _ in client.send_stream({"model": "test"}):
                pass
        await client.close()

    @pytest.mark.asyncio
    async def test_send_stream_retries_retryable_status(self, httpx_mock):
        cfg = _make_config(max_retries=1)
        client = UpstreamClient(cfg)

        sse_body = (
            'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        httpx_mock.add_response(status_code=503, text="unavailable")
        httpx_mock.add_response(
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )

        events = []
        async for ev in client.send_stream({"model": "test", "stream": True}):
            events.append(ev)

        assert len(events) == 2
        assert events[-1].data == "[DONE]"
        await client.close()
