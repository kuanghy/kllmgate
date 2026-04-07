"""错误类型层次的单元测试"""

import json

import pytest

from kllmgate.errors import (
    GatewayError,
    ConfigError,
    ProtocolError,
    UpstreamError,
    UpstreamHTTPError,
    ConversionError,
    _extract_upstream_error_fields,
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

    def test_upstream_http_error_preserves_status_code(self):
        e = UpstreamHTTPError(
            429,
            '{"error": {"message": "Rate limit exceeded"}}',
        )
        assert e.status_code == 429
        assert e.upstream_status == 429
        assert e.message == "Rate limit exceeded"
        assert isinstance(e, UpstreamError)

    def test_upstream_http_error_fallback_message(self):
        e = UpstreamHTTPError(503, "")
        assert e.status_code == 503
        assert e.message == "upstream returned HTTP 503"

    def test_upstream_http_error_non_json_body(self):
        e = UpstreamHTTPError(502, "<html>Bad Gateway</html>")
        assert e.status_code == 502
        assert e.message == "<html>Bad Gateway</html>"

    def test_upstream_http_error_invalid_status_defaults_to_502(self):
        e = UpstreamHTTPError(999, '{"error": {"message": "weird"}}')
        assert e.status_code == 502

    def test_upstream_http_error_stores_headers(self):
        hdrs = {"retry-after": "30", "x-request-id": "abc"}
        e = UpstreamHTTPError(429, "", upstream_headers=hdrs)
        assert e.upstream_headers == hdrs

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


class TestExtractUpstreamErrorFields:

    def test_openai_format(self):
        raw = json.dumps({
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
                "param": None,
            },
        })
        fields = _extract_upstream_error_fields(raw)
        assert fields["message"] == "Invalid API key"
        assert fields["type"] == "invalid_request_error"
        assert fields["code"] == "invalid_api_key"
        assert "param" not in fields

    def test_anthropic_format(self):
        raw = json.dumps({
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "invalid x-api-key",
            },
        })
        fields = _extract_upstream_error_fields(raw)
        assert fields["message"] == "invalid x-api-key"
        assert fields["type"] == "authentication_error"

    def test_non_json(self):
        fields = _extract_upstream_error_fields("<html>err</html>")
        assert fields["message"] == "<html>err</html>"

    def test_non_json_truncated(self):
        long_html = "x" * 500
        fields = _extract_upstream_error_fields(long_html)
        assert len(fields["message"]) == 200

    def test_empty_body(self):
        assert _extract_upstream_error_fields("") == {}

    def test_json_with_string_error(self):
        raw = json.dumps({"error": "something broke"})
        fields = _extract_upstream_error_fields(raw)
        assert fields["message"] == "something broke"

    def test_json_with_toplevel_message(self):
        raw = json.dumps({"message": "top level msg"})
        fields = _extract_upstream_error_fields(raw)
        assert fields["message"] == "top level msg"


class TestFormatErrorResponse:

    def test_openai_chat_format(self):
        e = ConfigError(
            "provider 'xxx' not found", code="unknown_provider",
        )
        resp = format_error_response(e, ProtocolFormat.OPENAI_CHAT)
        assert resp.status_code == 400
        body = json.loads(resp.body)
        assert body == {
            "error": {
                "type": "invalid_request_error",
                "code": "unknown_provider",
                "message": "provider 'xxx' not found",
            },
        }

    def test_openai_responses_format(self):
        e = UpstreamError("timeout")
        resp = format_error_response(
            e, ProtocolFormat.OPENAI_RESPONSES,
        )
        body = json.loads(resp.body)
        assert "error" in body
        assert body["error"]["type"] == "upstream_error"
        assert body["error"]["code"] == "upstream_error"
        assert resp.status_code == 502

    def test_anthropic_format(self):
        e = ConfigError(
            "provider 'xxx' not found", code="unknown_provider",
        )
        resp = format_error_response(
            e, ProtocolFormat.ANTHROPIC_MESSAGES,
        )
        body = json.loads(resp.body)
        assert body == {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "provider 'xxx' not found",
            },
        }
        assert resp.status_code == 400

    def test_upstream_http_error_preserves_status_and_message(self):
        e = UpstreamHTTPError(
            401,
            json.dumps({
                "error": {
                    "message": "Incorrect API key",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                },
            }),
        )
        resp = format_error_response(e, ProtocolFormat.OPENAI_CHAT)
        body = json.loads(resp.body)
        assert resp.status_code == 401
        assert body["error"]["message"] == "Incorrect API key"
        assert body["error"]["type"] == "authentication_error"
        assert body["error"]["code"] == "invalid_api_key"

    def test_upstream_http_error_source_header(self):
        e = UpstreamHTTPError(429, '{"error": {"message": "x"}}')
        resp = format_error_response(e, ProtocolFormat.OPENAI_CHAT)
        assert resp.headers["x-kllmgate-error-source"] == "upstream"

    def test_upstream_http_error_forwards_headers(self):
        e = UpstreamHTTPError(
            429, '{"error": {"message": "rate limited"}}',
            upstream_headers={"retry-after": "60"},
        )
        resp = format_error_response(e, ProtocolFormat.OPENAI_CHAT)
        assert resp.headers["retry-after"] == "60"

    def test_upstream_http_error_anthropic_format(self):
        e = UpstreamHTTPError(
            401,
            json.dumps({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid x-api-key",
                },
            }),
        )
        resp = format_error_response(
            e, ProtocolFormat.ANTHROPIC_MESSAGES,
        )
        body = json.loads(resp.body)
        assert resp.status_code == 401
        assert body["type"] == "error"
        assert body["error"]["message"] == "invalid x-api-key"
        assert body["error"]["type"] == "authentication_error"

    def test_upstream_http_error_fallback_fields(self):
        e = UpstreamHTTPError(500, "not json at all")
        resp = format_error_response(e, ProtocolFormat.OPENAI_CHAT)
        body = json.loads(resp.body)
        assert resp.status_code == 500
        assert body["error"]["message"] == "not json at all"
        assert body["error"]["type"] == "upstream_error"
        assert body["error"]["code"] == "upstream_http_error"
