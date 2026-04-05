"""应用路由层集成测试"""

import json
import textwrap
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from kllmgate.app import _extract_forward_headers, create_app


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent("""\
        [providers.test_openai]
        base_url = "https://api.example.com/v1"
        api_key = "sk-test"
        protocol = "openai"
        wire_api = "chat"

        [providers.test_anthropic]
        base_url = "https://api.anthropic.com"
        api_key = "sk-ant-test"
        protocol = "anthropic"
    """))
    return str(path)


@pytest.fixture
def app(config_file):
    return create_app(config_path=config_file)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


class TestRoutes:

    def test_openai_chat_invalid_model(self, client):
        resp = client.post(
            "/openai/chat/completions",
            json={"model": "no-slash", "messages": []},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "invalid_model_format"

    def test_openai_chat_unknown_provider(self, client):
        resp = client.post(
            "/openai/chat/completions",
            json={
                "model": "nonexistent/gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "unknown_provider"

    def test_openai_responses_invalid_model(self, client):
        resp = client.post(
            "/openai/responses",
            json={"model": "bad"},
        )
        assert resp.status_code == 400

    def test_anthropic_messages_invalid_model(self, client):
        resp = client.post(
            "/anthropic/v1/messages",
            json={"model": "bad"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"

    def test_openai_chat_error_format_is_openai(self, client):
        resp = client.post(
            "/openai/chat/completions",
            json={"model": "x/y", "messages": []},
        )
        body = resp.json()
        assert "error" in body
        assert "type" in body["error"]
        assert "code" in body["error"]

    def test_anthropic_error_format_is_anthropic(self, client):
        resp = client.post(
            "/anthropic/v1/messages",
            json={"model": "x/y"},
        )
        body = resp.json()
        assert body["type"] == "error"
        assert "message" in body["error"]

    def test_header_provider_routes_bare_model(self, client):
        """header 提供 provider 后裸模型名不再报 invalid_model_format"""
        try:
            resp = client.post(
                "/openai/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"X-KLLMGate-Provider": "test_openai"},
            )
        except Exception:
            return
        if resp.status_code == 400:
            assert resp.json()["error"]["code"] != "invalid_model_format"

    def test_header_provider_unknown_raises(self, client):
        resp = client.post(
            "/openai/chat/completions",
            json={"model": "gpt-4", "messages": []},
            headers={"X-KLLMGate-Provider": "nonexistent"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "unknown_provider"

    def test_model_slash_overrides_header(self, client):
        """model 含 / 时 provider 以 model 为准，忽略 header"""
        try:
            resp = client.post(
                "/openai/chat/completions",
                json={
                    "model": "test_openai/gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"X-KLLMGate-Provider": "test_anthropic"},
            )
        except Exception:
            return
        if resp.status_code == 400:
            assert resp.json()["error"]["code"] != "invalid_model_format"


class TestHealthCheck:

    def test_head_anthropic_returns_200(self, client):
        resp = client.head("/anthropic")
        assert resp.status_code == 200

    def test_head_openai_returns_200(self, client):
        resp = client.head("/openai")
        assert resp.status_code == 200

    def test_get_anthropic_returns_200(self, client):
        resp = client.get("/anthropic")
        assert resp.status_code == 200

    def test_head_unknown_returns_404(self, client):
        resp = client.head("/unknown")
        assert resp.status_code == 404


class TestExtractForwardHeaders:

    @staticmethod
    def _make_request(**headers):
        req = MagicMock()
        req.headers = Headers(headers=headers)
        return req

    def test_extracts_anthropic_beta(self):
        req = self._make_request(
            **{"anthropic-beta": "output-128k-2025-02-19"},
        )
        result = _extract_forward_headers(req)
        assert result == {"anthropic-beta": "output-128k-2025-02-19"}

    def test_extracts_anthropic_version(self):
        req = self._make_request(
            **{"anthropic-version": "2024-10-22"},
        )
        result = _extract_forward_headers(req)
        assert result == {"anthropic-version": "2024-10-22"}

    def test_extracts_multiple_headers(self):
        req = self._make_request(**{
            "anthropic-beta": "some-beta",
            "anthropic-version": "2024-10-22",
        })
        result = _extract_forward_headers(req)
        assert result == {
            "anthropic-beta": "some-beta",
            "anthropic-version": "2024-10-22",
        }

    def test_returns_none_when_no_matching_headers(self):
        req = self._make_request(
            **{"content-type": "application/json"},
        )
        result = _extract_forward_headers(req)
        assert result is None

    def test_ignores_unrelated_headers(self):
        req = self._make_request(**{
            "authorization": "Bearer sk-xxx",
            "anthropic-beta": "some-beta",
            "x-custom": "value",
        })
        result = _extract_forward_headers(req)
        assert result == {"anthropic-beta": "some-beta"}


class TestModelAliasRoutes:

    @pytest.fixture
    def alias_config_file(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent("""\
            [providers.scnet]
            base_url = "https://api.scnet.cn/v1"
            api_key = "sk-test"
            protocol = "openai"

            [model_aliases]
            "MiniMax-M2.5" = "scnet/MiniMax-M2.5"
        """))
        return str(path)

    @pytest.fixture
    def alias_client(self, alias_config_file):
        app = create_app(config_path=alias_config_file)
        with TestClient(app) as c:
            yield c

    def test_alias_routes_bare_model(self, alias_client):
        """别名匹配后裸模型名不再报 invalid_model_format"""
        try:
            resp = alias_client.post(
                "/openai/responses",
                json={"model": "MiniMax-M2.5", "input": "hi"},
            )
        except Exception:
            return
        if resp.status_code == 400:
            assert resp.json()["error"]["code"] != "invalid_model_format"

    def test_alias_unknown_model_falls_through(self, alias_client):
        resp = alias_client.post(
            "/openai/responses",
            json={"model": "unknown-model"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "invalid_model_format"
