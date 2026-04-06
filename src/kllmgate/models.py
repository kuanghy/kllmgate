"""核心数据模型"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class ProtocolFormat(str, Enum):
    OPENAI_CHAT = "openai.chat"
    OPENAI_RESPONSES = "openai.responses"
    ANTHROPIC_MESSAGES = "anthropic.messages"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8500
    log_level: str = "info"
    default_provider: str | None = None


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    protocol: str
    api_key: str | None = None
    env_key: str | None = None
    wire_api: str | None = None
    tool_style: str = "standard"
    timeout_seconds: int = 120
    max_retries: int = 2
    models: list[str] | None = None

    def __post_init__(self):
        if self.wire_api is None:
            self.wire_api = (
                "messages" if self.protocol == "anthropic" else "chat"
            )

    @property
    def protocol_format(self) -> ProtocolFormat:
        if self.protocol == "anthropic":
            return ProtocolFormat.ANTHROPIC_MESSAGES
        if self.wire_api == "responses":
            return ProtocolFormat.OPENAI_RESPONSES
        return ProtocolFormat.OPENAI_CHAT

    def resolve_api_key(self) -> str:
        """解析 provider 的 API key

        优先使用显式配置的 api_key；未配置时再读取 env_key 对应的环境变量。
        若两者都不可用，则抛出 ConfigError。
        """
        from .errors import ConfigError

        if self.api_key:
            return self.api_key
        if self.env_key:
            value = os.environ.get(self.env_key, "")
            if value:
                return value
        raise ConfigError(
            f"API key not configured for provider {self.name!r}: "
            f"set api_key or env_key in config",
            code="api_key_not_configured",
        )


@dataclass
class GatewayConfig:
    server: ServerConfig
    providers: dict[str, ProviderConfig]
    model_aliases: dict[str, str]

    @property
    def default_provider(self) -> str | None:
        return self.server.default_provider
