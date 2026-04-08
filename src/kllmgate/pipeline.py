"""请求处理管线"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

from .converters import Converter
from .converters.passthrough import PassthroughConverter
from .converters.openai_chat_tool_adapt import OpenaiChatToolAdaptConverter
from .converters.openai_responses_tool_adapt import (
    OpenaiResponsesToolAdaptConverter,
)
from .converters.openai_responses_to_openai_chat import (
    OpenaiResponsesToOpenaiChatConverter,
)
from .converters.openai_chat_to_openai_responses import (
    OpenaiChatToOpenaiResponsesConverter,
)
from .converters.openai_chat_to_anthropic_messages import (
    OpenaiChatToAnthropicMessagesConverter,
)
from .converters.anthropic_messages_to_openai_chat import (
    AnthropicMessagesToOpenaiChatConverter,
)
from .converters.openai_responses_to_anthropic_messages import (
    OpenaiResponsesToAnthropicMessagesConverter,
)
from .converters.anthropic_messages_to_openai_responses import (
    AnthropicMessagesToOpenaiResponsesConverter,
)
from .errors import (
    ConfigError, ConversionError, GatewayError,
    UpstreamError, UpstreamHTTPError,
)
from .models import ProtocolFormat, ProviderConfig
from .sse import SseEvent, format_data_only_sse, format_named_sse
from .toolcall import ToolAdapter
from .toolcall.standard import StandardToolAdapter
from .toolcall.minimax_xml import MinimaxXmlToolAdapter
from .toolcall.anthropic import AnthropicToolAdapter
from .upstream.client import UpstreamClient
from .utils import text_shorten

PF = ProtocolFormat


def _strip_inbound_system_prompt(body: dict, fmt: ProtocolFormat) -> None:
    """按入站协议就地移除请求体中的系统指令"""
    if fmt == PF.OPENAI_CHAT:
        body["messages"] = [
            m for m in body.get("messages", [])
            if m.get("role") not in ("system", "developer")
        ]
    elif fmt == PF.OPENAI_RESPONSES:
        body.pop("instructions", None)
        if isinstance(body.get("input"), list):
            body["input"] = [
                item for item in body["input"]
                if not (
                    isinstance(item, dict)
                    and item.get("role") in ("system", "developer")
                )
            ]
    elif fmt == PF.ANTHROPIC_MESSAGES:
        body.pop("system", None)


CONVERTER_REGISTRY: dict[
    tuple[ProtocolFormat, ProtocolFormat],
    type[Converter],
] = {
    (PF.OPENAI_CHAT, PF.OPENAI_RESPONSES): OpenaiChatToOpenaiResponsesConverter,
    (PF.OPENAI_CHAT, PF.ANTHROPIC_MESSAGES): OpenaiChatToAnthropicMessagesConverter,
    (PF.OPENAI_RESPONSES, PF.OPENAI_CHAT): OpenaiResponsesToOpenaiChatConverter,
    (PF.OPENAI_RESPONSES, PF.ANTHROPIC_MESSAGES): OpenaiResponsesToAnthropicMessagesConverter,
    (PF.ANTHROPIC_MESSAGES, PF.OPENAI_CHAT): AnthropicMessagesToOpenaiChatConverter,
    (PF.ANTHROPIC_MESSAGES, PF.OPENAI_RESPONSES): AnthropicMessagesToOpenaiResponsesConverter,
    (PF.OPENAI_CHAT, PF.OPENAI_CHAT): OpenaiChatToolAdaptConverter,
    (PF.OPENAI_RESPONSES, PF.OPENAI_RESPONSES): OpenaiResponsesToolAdaptConverter,
}


def get_tool_adapter(provider: ProviderConfig) -> ToolAdapter:
    if provider.protocol == "anthropic":
        return AnthropicToolAdapter()
    match provider.tool_style:
        case "minimax_xml":
            return MinimaxXmlToolAdapter()
        case "standard":
            return StandardToolAdapter()
        case _:
            raise ConfigError(
                f"unknown tool_style for provider {provider.name!r}: "
                f"{provider.tool_style!r}",
                code="invalid_tool_style",
            )


def get_converter(
    inbound_format: ProtocolFormat,
    upstream_format: ProtocolFormat,
    tool_adapter: ToolAdapter,
) -> Converter:
    if (
        inbound_format == upstream_format
        and isinstance(tool_adapter, StandardToolAdapter)
    ):
        return PassthroughConverter(tool_adapter)

    key = (inbound_format, upstream_format)
    converter_cls = CONVERTER_REGISTRY.get(key)
    if converter_cls is None:
        raise ConfigError(
            f"no converter for {inbound_format} -> {upstream_format}",
            code="unsupported_conversion",
        )
    return converter_cls(tool_adapter)


async def _replay_prefetched_stream(
    first_event,
    upstream_events: AsyncIterator,
) -> AsyncIterator:
    yield first_event
    async for event in upstream_events:
        yield event


PROVIDER_HEADER = "X-KLLMGate-Provider"


def resolve_provider_and_model(
    model_ref: str,
    header_provider: str | None,
    model_aliases: dict[str, str],
    default_provider: str | None = None,
) -> tuple[str, str]:
    """解析 provider 和上游模型名

    优先级：
    1. model 含 "/" → 直接拆分
    2. model 命中 model_aliases → 展开后拆分
    3. X-KLLMGate-Provider header → header 做 provider，model 原样做上游模型名
    4. default_provider 配置 → 兜底路由
    5. 均不满足 → 抛出 ConfigError

    当 model 含 "/" 且 header 也存在但两者 provider 不一致时，以 model 为准。
    """
    if not model_ref:
        raise ConfigError(
            "model field is required",
            code="invalid_model_format",
        )

    if "/" in model_ref:
        provider_name, upstream_model = model_ref.split("/", 1)
        if (
            header_provider
            and header_provider != provider_name
        ):
            logger.debug(
                "model provider %r overrides header provider %r",
                provider_name, header_provider,
            )
        return provider_name, upstream_model

    if model_ref in model_aliases:
        return model_aliases[model_ref].split("/", 1)

    if header_provider:
        return header_provider, model_ref

    if default_provider:
        return default_provider, model_ref

    raise ConfigError(
        f"cannot determine provider for model {model_ref!r}: "
        f"use provider/model format, configure a model_aliases entry, "
        f"or set the {PROVIDER_HEADER} header",
        code="invalid_model_format",
    )


def _normalize_usage(usage: dict) -> dict:
    """将不同协议的 usage 格式统一为 input/output/total tokens"""
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_t = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_t = usage.get(
        "output_tokens", usage.get("completion_tokens", 0),
    )
    total_t = usage.get("total_tokens", 0) or (input_t + output_t)
    return {
        "input_tokens": input_t,
        "output_tokens": output_t,
        "total_tokens": total_t,
    }


def _log_request(
    request_id: str,
    inbound_format: ProtocolFormat,
    provider: str,
    model: str,
    stream: bool,
    start_time: float,
    usage: dict,
    status: str = "ok",
    error_type: str | None = None,
) -> None:
    """记录结构化请求完成日志"""
    duration_ms = int((time.monotonic() - start_time) * 1000)
    normalized = _normalize_usage(usage)
    parts = [
        f"request_id={request_id}",
        f"protocol={inbound_format.value}",
        f"provider={provider}",
        f"model={model}",
        f"stream={stream}",
        f"duration_ms={duration_ms}",
        f"input_tokens={normalized['input_tokens']}",
        f"output_tokens={normalized['output_tokens']}",
        f"total_tokens={normalized['total_tokens']}",
        f"status={status}",
    ]
    if error_type:
        parts.append(f"error_type={error_type}")
    logger.info("Request completed: %s", " ".join(parts))


async def _track_usage(
    upstream_events: AsyncIterator[SseEvent],
    usage: dict,
) -> AsyncIterator[SseEvent]:
    """包装上游 SSE 事件流，从中提取 token 用量"""
    async for event in upstream_events:
        if event.data and event.data != "[DONE]":
            try:
                data = json.loads(event.data)
                # OpenAI Chat/Responses: 顶层 usage 字段
                if isinstance(data.get("usage"), dict):
                    usage.update(data["usage"])
                # OpenAI Responses: response.completed 中的 usage
                resp = data.get("response")
                if isinstance(resp, dict) and isinstance(
                    resp.get("usage"), dict,
                ):
                    usage.update(resp["usage"])
                # Anthropic: message_start 中的 message.usage
                msg = data.get("message")
                if isinstance(msg, dict) and isinstance(
                    msg.get("usage"), dict,
                ):
                    usage.update(msg["usage"])
            except (json.JSONDecodeError, ValueError):
                pass
        yield event


def _make_stream_error_events(
    inbound_format: ProtocolFormat,
    error_msg: str,
) -> list[str]:
    """生成协议对应的流式错误 SSE 事件"""
    if inbound_format == PF.OPENAI_RESPONSES:
        return [format_named_sse("response.completed", {
            "type": "response.completed",
            "response": {
                "id": f"resp_{uuid.uuid4().hex[:24]}",
                "object": "response",
                "created_at": int(time.time()),
                "status": "incomplete",
                "output": [],
                "error": {
                    "type": "server_error",
                    "code": "stream_interrupted",
                    "message": error_msg,
                },
            },
        })]
    if inbound_format == PF.OPENAI_CHAT:
        return [
            format_data_only_sse({
                "choices": [{"delta": {}, "finish_reason": None}],
            }),
            "data: [DONE]\n\n",
        ]
    if inbound_format == PF.ANTHROPIC_MESSAGES:
        return [format_named_sse("error", {
            "type": "error",
            "error": {
                "type": "server_error",
                "message": error_msg,
            },
        })]
    return []


async def _logged_stream(
    converter_stream: AsyncIterator[str],
    request_id: str,
    inbound_format: ProtocolFormat,
    provider: str,
    model: str,
    stream_usage: dict,
    start_time: float,
) -> AsyncIterator[str]:
    """包装 converter 输出流，提供错误处理与请求日志"""
    status = "ok"
    error_type_val: str | None = None
    try:
        async for chunk in converter_stream:
            yield chunk
    except Exception as e:
        status = "error"
        error_type_val = getattr(e, "error_type", "server_error")
        if isinstance(e, UpstreamHTTPError):
            logger.warning(
                "%s: %s | upstream_body=%s",
                type(e).__name__, e.message,
                text_shorten(e.upstream_body, 200) if e.upstream_body else "",
            )
        elif isinstance(e, (UpstreamError, ConversionError)):
            logger.warning("%s: %s", type(e).__name__, e.message)
        else:
            logger.error("Stream error: %s", e, exc_info=True)
        for event_text in _make_stream_error_events(
            inbound_format, str(e),
        ):
            yield event_text
    finally:
        _log_request(
            request_id, inbound_format, provider, model,
            True, start_time, stream_usage, status, error_type_val,
        )


async def process_request(
    inbound_format: ProtocolFormat,
    body: dict,
    providers: dict[str, ProviderConfig],
    upstream_clients: dict[str, UpstreamClient],
    header_provider: str | None = None,
    model_aliases: dict[str, str] | None = None,
    default_provider: str | None = None,
    forward_headers: dict[str, str] | None = None,
) -> JSONResponse | StreamingResponse:
    request_id = uuid.uuid4().hex
    start_time = time.monotonic()
    provider_name = ""
    upstream_model = ""
    is_stream = body.get("stream", False)

    try:
        model_ref = body.get("model", "")
        provider_name, upstream_model = resolve_provider_and_model(
            model_ref, header_provider, model_aliases or {},
            default_provider=default_provider,
        )

        if provider_name not in providers:
            raise ConfigError(
                f"unknown provider: {provider_name!r}",
                code="unknown_provider",
            )
        provider = providers[provider_name]

        if provider.models and upstream_model not in provider.models:
            raise ConfigError(
                f"model {upstream_model!r} not supported by "
                f"provider {provider_name!r}",
                code="model_not_supported",
            )

        upstream_format = provider.protocol_format
        tool_adapter = get_tool_adapter(provider)
        converter = get_converter(
            inbound_format, upstream_format, tool_adapter,
        )

        if provider.strip_system_prompt:
            _strip_inbound_system_prompt(body, inbound_format)
        upstream_body = converter.convert_request(body, upstream_model)
        logger.debug(
            "Upstream request: request_id=%s provider=%s model=%s "
            "converter=%s",
            request_id, provider_name, upstream_model,
            type(converter).__name__,
        )

        client = upstream_clients.get(provider_name)
        if client is None:
            raise ConfigError(
                f"upstream client not initialized for "
                f"{provider_name!r}",
                code="client_not_found",
            )

        if is_stream:
            stream_usage: dict = {}
            upstream_events = client.send_stream(
                upstream_body, extra_headers=forward_headers,
            )
            try:
                first_event = await anext(upstream_events)
            except StopAsyncIteration:
                async def _empty_events():
                    return
                    yield

                source_events = _empty_events()
            else:
                source_events = _replay_prefetched_stream(
                    first_event, upstream_events,
                )

            tracked = _track_usage(source_events, stream_usage)
            event_stream = converter.convert_stream(tracked)
            return StreamingResponse(
                _logged_stream(
                    event_stream, request_id, inbound_format,
                    provider_name, upstream_model,
                    stream_usage, start_time,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        upstream_response = await client.send(
            upstream_body, extra_headers=forward_headers,
        )
        converted = converter.convert_response(upstream_response)
        _log_request(
            request_id, inbound_format, provider_name,
            upstream_model, False, start_time,
            upstream_response.get("usage", {}),
        )
        return JSONResponse(converted)

    except GatewayError as e:
        _log_request(
            request_id, inbound_format, provider_name,
            upstream_model, is_stream, start_time,
            {}, "error", e.error_type,
        )
        if isinstance(e, UpstreamHTTPError):
            logger.warning(
                "%s: %s | upstream_body=%s",
                type(e).__name__, e.message,
                text_shorten(e.upstream_body, 200) if e.upstream_body else "",
            )
        elif isinstance(e, (UpstreamError, ConversionError)):
            logger.warning(
                "%s: %s", type(e).__name__, e.message,
            )
        raise
    except Exception as e:
        _log_request(
            request_id, inbound_format, provider_name,
            upstream_model, is_stream, start_time,
            {}, "error", "server_error",
        )
        logger.error("Unexpected error: %s", e, exc_info=True)
        raise
