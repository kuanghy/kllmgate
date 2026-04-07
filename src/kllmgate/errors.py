"""错误类型定义与错误响应格式化"""

from __future__ import annotations

import json

from fastapi.responses import JSONResponse

from .models import ProtocolFormat


class GatewayError(Exception):
    """网关错误基类"""

    status_code: int = 500
    error_type: str = "server_error"
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class ConfigError(GatewayError):
    """配置类错误：provider 不存在、模型不支持、API key 缺失"""

    status_code = 400
    error_type = "invalid_request_error"
    code = "config_error"


class ProtocolError(GatewayError):
    """协议类错误：请求体结构非法、必填字段缺失"""

    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request"


class UpstreamError(GatewayError):
    """上游类错误：连接失败、超时、重试耗尽"""

    status_code = 502
    error_type = "upstream_error"
    code = "upstream_error"


def _extract_upstream_error_fields(raw: str) -> dict:
    """从上游错误响应体中提取结构化字段

    支持 OpenAI 和 Anthropic 两种错误格式，非 JSON 响应截断返回。
    """
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"message": raw[:200].strip()}

    error = data.get("error")
    if isinstance(error, dict):
        return {
            k: v for k, v in {
                "message": error.get("message", ""),
                "type": error.get("type", ""),
                "code": error.get("code"),
                "param": error.get("param"),
            }.items() if v
        }
    if isinstance(error, str):
        return {"message": error}
    if "message" in data:
        return {"message": data["message"]}
    return {}


class UpstreamHTTPError(UpstreamError):
    """上游 HTTP 错误，携带原始状态码、响应体和响应头"""

    def __init__(
        self,
        upstream_status: int,
        upstream_body: str,
        upstream_headers: dict[str, str] | None = None,
    ):
        self.upstream_status = upstream_status
        self.upstream_body = upstream_body
        self.upstream_headers = upstream_headers or {}
        self.upstream_fields = _extract_upstream_error_fields(upstream_body)
        msg = (
            self.upstream_fields.get("message")
            or f"upstream returned HTTP {upstream_status}"
        )
        super().__init__(msg, code="upstream_http_error")
        self.status_code = (
            upstream_status if 400 <= upstream_status < 600 else 502
        )


class ConversionError(GatewayError):
    """转换类错误：响应解析失败、流中断、工具调用语义非法"""

    status_code = 502
    error_type = "conversion_error"
    code = "conversion_error"


_ERROR_SOURCE_HEADER = "X-KLLMGate-Error-Source"


def format_error_response(
    error: GatewayError,
    inbound_format: ProtocolFormat,
) -> JSONResponse:
    response_headers: dict[str, str] = {}

    if isinstance(error, UpstreamHTTPError):
        fields = error.upstream_fields
        message = fields.get("message") or error.message
        error_type = fields.get("type") or error.error_type
        error_code = str(fields.get("code") or error.code)
        status_code = error.status_code
        response_headers[_ERROR_SOURCE_HEADER] = "upstream"
        for k, v in error.upstream_headers.items():
            response_headers[k] = v
    else:
        message = error.message
        error_type = error.error_type
        error_code = error.code
        status_code = error.status_code

    if inbound_format == ProtocolFormat.ANTHROPIC_MESSAGES:
        body: dict = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }
    else:
        body = {
            "error": {
                "type": error_type,
                "code": error_code,
                "message": message,
            },
        }

    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=response_headers or None,
    )
