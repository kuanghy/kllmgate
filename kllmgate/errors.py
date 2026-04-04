"""错误类型定义与错误响应格式化"""

from __future__ import annotations

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


class UpstreamHTTPError(UpstreamError):
    """上游 HTTP 错误，携带原始状态码和响应体"""

    def __init__(self, upstream_status: int, upstream_body: str):
        self.upstream_status = upstream_status
        self.upstream_body = upstream_body
        super().__init__(
            f"upstream returned HTTP {upstream_status}",
            code="upstream_http_error",
        )


class ConversionError(GatewayError):
    """转换类错误：响应解析失败、流中断、工具调用语义非法"""
    status_code = 502
    error_type = "conversion_error"
    code = "conversion_error"


def format_error_response(
    error: GatewayError,
    inbound_format: ProtocolFormat,
) -> JSONResponse:
    if inbound_format == ProtocolFormat.ANTHROPIC_MESSAGES:
        body = {
            "type": "error",
            "error": {
                "type": error.error_type,
                "message": error.message,
            },
        }
    else:
        body = {
            "error": {
                "type": error.error_type,
                "code": error.code,
                "message": error.message,
            },
        }
    return JSONResponse(status_code=error.status_code, content=body)
