"""请求处理管线"""

from __future__ import annotations

from fastapi.responses import JSONResponse, StreamingResponse

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
from .errors import ConfigError
from .models import ProtocolFormat, ProviderConfig
from .tools import ToolAdapter
from .tools.standard import StandardToolAdapter
from .tools.minimax_xml import MinimaxXmlToolAdapter
from .tools.anthropic import AnthropicToolAdapter
from .upstream.client import UpstreamClient

PF = ProtocolFormat

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


async def process_request(
    inbound_format: ProtocolFormat,
    body: dict,
    providers: dict[str, ProviderConfig],
    upstream_clients: dict[str, UpstreamClient],
) -> JSONResponse | StreamingResponse:
    model_ref = body.get("model", "")
    if not model_ref or "/" not in model_ref:
        raise ConfigError(
            f"model must be in provider/model format: {model_ref!r}",
            code="invalid_model_format",
        )

    provider_name, upstream_model = model_ref.split("/", 1)

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
    converter = get_converter(inbound_format, upstream_format, tool_adapter)

    upstream_body = converter.convert_request(body, upstream_model)

    client = upstream_clients.get(provider_name)
    if client is None:
        raise ConfigError(
            f"upstream client not initialized for {provider_name!r}",
            code="client_not_found",
        )

    stream = body.get("stream", False)

    if stream:
        upstream_events = client.send_stream(upstream_body)
        event_stream = converter.convert_stream(upstream_events)
        return StreamingResponse(
            event_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    upstream_response = await client.send(upstream_body)
    return JSONResponse(converter.convert_response(upstream_response))
