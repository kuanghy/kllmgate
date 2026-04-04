"""错误类型层次的单元测试"""

import pytest

from kllmgate.errors import (
    GatewayError,
    ConfigError,
    ProtocolError,
    UpstreamError,
    UpstreamHTTPError,
    ConversionError,
    format_error_response,
)
from kllmgate.models import ProtocolFormat


class TestErrorHierarchy:

    def test_gateway_error_defaults(self):
        e = GatewayError("something went wrong")
        assert e.status_code == 500
        assert e.error_type == "server_error"
        assert e.code == "internal_error"
        assert e.message == "something went wrong"
        assert str(e) == "something went wrong"

    def test_config_error(self):
        e = ConfigError("provider 'xxx' not found")
        assert e.status_code == 400
        assert e.error_type == "invalid_request_error"
        assert e.code == "config_error"
        assert isinstance(e, GatewayError)

    def test_config_error_custom_code(self):
        e = ConfigError("not found", code="unknown_provider")
        assert e.code == "unknown_provider"

    def test_protocol_error(self):
        e = ProtocolError("missing model field")
        assert e.status_code == 400
        assert e.error_type == "invalid_request_error"
        assert e.code == "invalid_request"
        assert isinstance(e, GatewayError)

    def test_upstream_error(self):
        e = UpstreamError("connection refused")
        assert e.status_code == 502
        assert e.error_type == "upstream_error"
        assert e.code == "upstream_error"
        assert isinstance(e, GatewayError)

    def test_upstream_http_error(self):
        e = UpstreamHTTPError(
            upstream_status=429,
            upstream_body='{"error": "rate limited"}',
        )
        assert e.status_code == 502
        assert e.upstream_status == 429
        assert e.upstream_body == '{"error": "rate limited"}'
        assert isinstance(e, UpstreamError)
        assert isinstance(e, GatewayError)

    def test_conversion_error(self):
        e = ConversionError("invalid JSON in response")
        assert e.status_code == 502
        assert e.error_type == "conversion_error"
        assert e.code == "conversion_error"
        assert isinstance(e, GatewayError)

    def test_all_errors_are_exceptions(self):
        for cls in (GatewayError, ConfigError, ProtocolError,
                    UpstreamError, ConversionError):
            e = cls("test")
            assert isinstance(e, Exception)


class TestFormatErrorResponse:

    def test_openai_chat_format(self):
        e = ConfigError("provider 'xxx' not found", code="unknown_provider")
        resp = format_error_response(e, ProtocolFormat.OPENAI_CHAT)
        assert resp.status_code == 400
        import json
        body = json.loads(resp.body)
        assert body == {
            "error": {
                "type": "invalid_request_error",
                "code": "unknown_provider",
                "message": "provider 'xxx' not found",
            }
        }

    def test_openai_responses_format(self):
        e = UpstreamError("timeout")
        resp = format_error_response(e, ProtocolFormat.OPENAI_RESPONSES)
        import json
        body = json.loads(resp.body)
        assert "error" in body
        assert body["error"]["type"] == "upstream_error"
        assert body["error"]["code"] == "upstream_error"
        assert resp.status_code == 502

    def test_anthropic_format(self):
        e = ConfigError("provider 'xxx' not found", code="unknown_provider")
        resp = format_error_response(e, ProtocolFormat.ANTHROPIC_MESSAGES)
        import json
        body = json.loads(resp.body)
        assert body == {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "provider 'xxx' not found",
            }
        }
        assert resp.status_code == 400

    def test_upstream_http_error_format(self):
        e = UpstreamHTTPError(401, "unauthorized")
        resp = format_error_response(e, ProtocolFormat.OPENAI_CHAT)
        import json
        body = json.loads(resp.body)
        assert body["error"]["type"] == "upstream_error"
        assert resp.status_code == 502
