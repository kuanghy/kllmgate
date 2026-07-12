"""应用路由层集成测试"""

import json
import textwrap
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from kllmgate.app import _extract_forward_headers, create_app
from kllmgate.config import load_config
from kllmgate.errors import ConfigError


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent("""\
        [providers.test_openai]
        base_url = "https://api.example.com/v1"
        api_key = "sk-test"
        protocol = "openai"
        wire_api = "chat"
        models = ["gpt-4.1"]

        [providers.test_anthropic]
        base_url = "https://api.anthropic.com"
        api_key = "sk-ant-test"
        protocol = "anthropic"

        [model_aliases]
        "MiniMax-M2.5" = "test_openai/gpt-4.1"
    """))
    return str(path)


@pytest.fixture
def app(config_file):
    return create_app(load_config(config_file))


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


class TestOpenAIModels:

    def test_openai_models_lists_aliases_and_provider_models(self, client):
        resp = client.get("/openai/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        model_ids = {item["id"] for item in body["data"]}
        assert "MiniMax-M2.5" in model_ids
        assert "test_openai/gpt-4.1" in model_ids
        assert "gpt-4.1" not in model_ids

    def test_openai_v1_models_matches_openai_models(self, client):
        base = client.get("/openai/models")
        v1 = client.get("/openai/v1/models")
        assert base.status_code == 200
        assert v1.status_code == 200
        assert base.json() == v1.json()

    def test_openai_models_filters_by_server_list(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent("""\
            [server]
            models_list = ["MiniMax-M2.5"]

            [providers.test_openai]
            base_url = "https://api.example.com/v1"
            api_key = "sk-test"
            protocol = "openai"
            wire_api = "chat"
            models = ["gpt-4.1"]

            [model_aliases]
            "MiniMax-M2.5" = "test_openai/gpt-4.1"
        """))
        with TestClient(create_app(load_config(str(path)))) as c:
            resp = c.get("/openai/models")
        assert resp.status_code == 200
        model_ids = {item["id"] for item in resp.json()["data"]}
        assert model_ids == {"MiniMax-M2.5"}


class TestAnthropicModels:

    def test_anthropic_models_lists_aliases_and_provider_models(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent("""\
            [providers.test_anthropic]
            base_url = "https://api.anthropic.com"
            api_key = "sk-ant-test"
            protocol = "anthropic"
            models = ["claude-sonnet-4-20250514"]

            [providers.test_openai]
            base_url = "https://api.example.com/v1"
            api_key = "sk-test"
            protocol = "openai"
            models = ["gpt-4.1"]

            [model_aliases]
            "Claude-Sonnet" = "test_anthropic/claude-sonnet-4-20250514"
        """))
        with TestClient(create_app(load_config(str(path)))) as c:
            resp = c.get("/anthropic/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        model_ids = {item["id"] for item in body["data"]}
        assert model_ids == {
            "Claude-Sonnet",
            "test_anthropic/claude-sonnet-4-20250514",
            "test_openai/gpt-4.1",
        }
        assert body["has_more"] is False
        assert body["data"][0]["type"] == "model"
        assert "capabilities" in body["data"][0]

    def test_models_list_unknown_entry_rejected_at_load(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent("""\
            [server]
            models_list = ["missing-from-candidates"]

            [providers.test_openai]
            base_url = "https://api.example.com/v1"
            api_key = "sk-test"
            protocol = "openai"
            models = ["gpt-4.1"]
        """))
        with pytest.raises(ConfigError, match="missing-from-candidates"):
            load_config(str(path))

    def test_empty_models_list_returns_empty_endpoints(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent("""\
            [server]
            models_list = []

            [providers.test_openai]
            base_url = "https://api.example.com/v1"
            api_key = "sk-test"
            protocol = "openai"
            models = ["gpt-4.1"]
        """))
        with TestClient(create_app(load_config(str(path)))) as c:
            openai_body = c.get("/openai/models").json()
            anthropic_body = c.get("/anthropic/v1/models").json()
        assert openai_body["data"] == []
        assert anthropic_body["data"] == []
        assert anthropic_body["has_more"] is False

    def test_openai_and_anthropic_models_share_same_ids(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent("""\
            [providers.test_anthropic]
            base_url = "https://api.anthropic.com"
            api_key = "sk-ant-test"
            protocol = "anthropic"
            models = ["claude-sonnet-4-20250514"]

            [providers.test_openai]
            base_url = "https://api.example.com/v1"
            api_key = "sk-test"
            protocol = "openai"
            models = ["gpt-4.1"]

            [model_aliases]
            "Claude-Sonnet" = "test_anthropic/claude-sonnet-4-20250514"
        """))
        with TestClient(create_app(load_config(str(path)))) as c:
            openai_ids = {
                item["id"] for item in c.get("/openai/models").json()["data"]
            }
            anthropic_ids = {
                item["id"] for item in c.get("/anthropic/v1/models").json()["data"]
            }
        assert openai_ids == anthropic_ids

    def test_anthropic_models_filters_by_server_list(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent("""\
            [server]
            models_list = ["Claude-Sonnet"]

            [providers.test_anthropic]
            base_url = "https://api.anthropic.com"
            api_key = "sk-ant-test"
            protocol = "anthropic"
            models = ["claude-sonnet-4-20250514"]

            [model_aliases]
            "Claude-Sonnet" = "test_anthropic/claude-sonnet-4-20250514"
        """))
        with TestClient(create_app(load_config(str(path)))) as c:
            resp = c.get("/anthropic/v1/models")
        assert resp.status_code == 200
        model_ids = {item["id"] for item in resp.json()["data"]}
        assert model_ids == {"Claude-Sonnet"}


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

    def test_health_ready_reports_providers(self, client, httpx_mock):
        # 任意 HTTP 响应（含 401）均视为上游可达
        httpx_mock.add_response(status_code=200)
        httpx_mock.add_response(status_code=401)
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert "test_openai" in body["providers"]
        assert "test_anthropic" in body["providers"]
        assert body["providers"]["test_openai"]["ok"] is True
        assert body["providers"]["test_anthropic"]["ok"] is True
        assert body["providers"]["test_openai"]["detail"].startswith("http_")
        assert body["providers"]["test_anthropic"]["detail"].startswith("http_")

    def test_health_ready_503_when_all_unreachable(self, client, httpx_mock):
        import httpx

        httpx_mock.add_exception(httpx.ConnectError("fail"))
        httpx_mock.add_exception(httpx.ConnectError("fail"))
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["providers"]["test_openai"]["ok"] is False
        assert body["providers"]["test_anthropic"]["ok"] is False

    def test_health_ready_probes_providers_in_parallel(self, app):
        import asyncio
        import time
        from unittest.mock import AsyncMock

        async def _slow_probe(*_args, **_kwargs):
            await asyncio.sleep(0.25)
            return True, "HTTP 200"

        with TestClient(app) as client:
            for upstream in client.app.state.upstream_clients.values():
                upstream.check_reachable = AsyncMock(side_effect=_slow_probe)
            # 若串行约 0.5s+；并行约 0.25s
            started = time.monotonic()
            resp = client.get("/health/ready")
            elapsed = time.monotonic() - started

        assert resp.status_code == 200
        assert elapsed < 0.45

    def test_health_ready_isolates_probe_exceptions(self, app, caplog):
        import logging
        from unittest.mock import AsyncMock

        with TestClient(app) as client:
            clients = client.app.state.upstream_clients
            names = list(clients.keys())
            assert len(names) >= 2
            clients[names[0]].check_reachable = AsyncMock(
                side_effect=RuntimeError("boom probe secret-host"),
            )
            clients[names[1]].check_reachable = AsyncMock(
                return_value=(True, "http_200"),
            )
            with caplog.at_level(logging.WARNING, logger="kllmgate.app"):
                resp = client.get("/health/ready")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["providers"][names[0]]["ok"] is False
        assert body["providers"][names[0]]["detail"] == "probe_error"
        assert "secret-host" not in resp.text
        assert body["providers"][names[1]]["ok"] is True
        assert any(
            "secret-host" in r.getMessage() for r in caplog.records
        )

    def test_health_ready_unreachable_detail_is_sanitized(
        self, client, httpx_mock,
    ):
        import httpx

        httpx_mock.add_exception(
            httpx.ConnectError('连接失败: host="internal.svc.local"'),
        )
        httpx_mock.add_exception(
            httpx.ConnectError('连接失败: host="internal.svc.local"'),
        )
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["providers"]["test_openai"]["detail"] == "connect_error"
        assert "internal.svc.local" not in resp.text


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
        app = create_app(load_config(alias_config_file))
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
