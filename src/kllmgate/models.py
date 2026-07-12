"""核心数据模型"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    models_list: list[str] | None = None


@dataclass
class ProtocolEndpointConfig:
    """auto 模式下单个协议子表配置"""

    base_url: str
    api_key: str | None = None
    env_key: str | None = None
    wire_api: str | None = None
    tool_style: str = "standard"
    thinking_style: str = "disabled"
    models: list[str] | None = None


@dataclass
class ResolvedUpstream:
    """一次请求解析出的上游目标"""

    protocol: str
    wire_api: str
    base_url: str
    api_key: str | None = None
    env_key: str | None = None
    fallback_api_key: str | None = None
    fallback_env_key: str | None = None
    tool_style: str = "standard"
    thinking_style: str = "disabled"
    models: list[str] | None = None
    provider_name: str = ""

    @property
    def protocol_format(self) -> ProtocolFormat:
        if self.protocol == "anthropic":
            return ProtocolFormat.ANTHROPIC_MESSAGES
        if self.wire_api == "responses":
            return ProtocolFormat.OPENAI_RESPONSES
        return ProtocolFormat.OPENAI_CHAT

    def resolve_api_key(self) -> str:
        """解析上游 API key

        优先级：显式 api_key → env_key → fallback_api_key → fallback_env_key。
        子表只配 env_key 且环境变量为空时，可回退到顶层 key。
        """
        from .errors import ConfigError

        if self.api_key:
            return self.api_key
        if self.env_key:
            value = os.environ.get(self.env_key, "")
            if value:
                return value
        if self.fallback_api_key:
            return self.fallback_api_key
        if self.fallback_env_key:
            value = os.environ.get(self.fallback_env_key, "")
            if value:
                return value
        raise ConfigError(
            f"API key not configured for provider {self.provider_name!r}: "
            f"set api_key or env_key in config",
            code="api_key_not_configured",
        )


@dataclass
class ProviderConfig:
    name: str
    base_url: str = ""
    protocol: str = "openai"
    api_key: str | None = None
    env_key: str | None = None
    wire_api: str | None = None
    tool_style: str = "standard"
    thinking_style: str = "disabled"
    timeout_seconds: int = 120
    max_retries: int = 2
    models: list[str] | None = None
    strip_system_prompt: bool = False
    openai: ProtocolEndpointConfig | None = None
    anthropic: ProtocolEndpointConfig | None = None

    def __post_init__(self):
        if self.protocol == "auto":
            return
        if self.wire_api is None:
            self.wire_api = (
                "messages" if self.protocol == "anthropic" else "chat"
            )

    @property
    def protocol_format(self) -> ProtocolFormat:
        if self.protocol == "auto":
            from .errors import ConfigError

            raise ConfigError(
                "protocol_format is undefined for protocol='auto'; "
                "use resolve_for_inbound()",
                code="auto_protocol_format_undefined",
            )
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

    def resolve_for_inbound(
        self, inbound_format: ProtocolFormat,
    ) -> ResolvedUpstream:
        """按入站协议解析上游目标

        auto：入站协议必须有对应子表，否则报错；openai 子表内
        chat↔responses 不一致时仍走该子表（由 converter 转换）。
        单协议：始终使用 provider 自身配置。
        """
        from .errors import ConfigError

        if self.protocol == "auto":
            return self._resolve_auto(inbound_format)

        return ResolvedUpstream(
            protocol=self.protocol,
            wire_api=self.wire_api or "chat",
            base_url=self.base_url,
            api_key=self.api_key,
            env_key=self.env_key,
            tool_style=self.tool_style,
            thinking_style=self.thinking_style,
            models=self.models,
            provider_name=self.name,
        )

    def _resolve_auto(
        self, inbound_format: ProtocolFormat,
    ) -> ResolvedUpstream:
        from .errors import ConfigError

        if inbound_format == ProtocolFormat.ANTHROPIC_MESSAGES:
            if self.anthropic is None:
                raise ConfigError(
                    f"provider {self.name!r} protocol=auto has no "
                    f"anthropic subsection for inbound "
                    f"{inbound_format.value}",
                    code="unsupported_inbound_protocol",
                )
            return self._from_endpoint("anthropic", self.anthropic)

        if inbound_format in (
            ProtocolFormat.OPENAI_CHAT,
            ProtocolFormat.OPENAI_RESPONSES,
        ):
            if self.openai is None:
                raise ConfigError(
                    f"provider {self.name!r} protocol=auto has no "
                    f"openai subsection for inbound "
                    f"{inbound_format.value}",
                    code="unsupported_inbound_protocol",
                )
            return self._from_endpoint("openai", self.openai)

        raise ConfigError(
            f"provider {self.name!r}: unsupported inbound format "
            f"{inbound_format.value}",
            code="unsupported_inbound_protocol",
        )

    def _from_endpoint(
        self,
        protocol: str,
        endpoint: ProtocolEndpointConfig,
    ) -> ResolvedUpstream:
        if protocol == "anthropic":
            wire_api = endpoint.wire_api or "messages"
            tool_style = "standard"
            thinking_style = "disabled"
        else:
            wire_api = endpoint.wire_api or "chat"
            tool_style = endpoint.tool_style
            thinking_style = endpoint.thinking_style

        models = (
            endpoint.models
            if endpoint.models is not None
            else self.models
        )
        if endpoint.api_key:
            api_key, env_key = endpoint.api_key, None
            fallback_api_key, fallback_env_key = None, None
        elif endpoint.env_key:
            # 子表 env_key 优先；为空时回退顶层 api_key/env_key
            api_key, env_key = None, endpoint.env_key
            fallback_api_key = self.api_key
            fallback_env_key = self.env_key
        else:
            api_key, env_key = self.api_key, self.env_key
            fallback_api_key, fallback_env_key = None, None
        return ResolvedUpstream(
            protocol=protocol,
            wire_api=wire_api,
            base_url=endpoint.base_url,
            api_key=api_key,
            env_key=env_key,
            fallback_api_key=fallback_api_key,
            fallback_env_key=fallback_env_key,
            tool_style=tool_style,
            thinking_style=thinking_style,
            models=models,
            provider_name=self.name,
        )

    def validate_api_keys(self) -> None:
        """启动时校验所有会用到的 API key 均可解析"""
        if self.protocol != "auto":
            self.resolve_api_key()
            return

        for protocol, endpoint in (
            ("openai", self.openai),
            ("anthropic", self.anthropic),
        ):
            if endpoint is None:
                continue
            self._from_endpoint(protocol, endpoint).resolve_api_key()

    def exposed_model_ids(self) -> set[str] | None:
        """返回可暴露的上游模型名集合；None 表示不限制（不自动列出）"""
        if self.protocol != "auto":
            if self.models is None:
                return None
            return set(self.models)

        collected: set[str] = set()
        unrestricted = False
        has_endpoint = False
        for endpoint in (self.openai, self.anthropic):
            if endpoint is None:
                continue
            has_endpoint = True
            effective = (
                endpoint.models
                if endpoint.models is not None
                else self.models
            )
            if effective is None:
                unrestricted = True
                continue
            collected.update(effective)
        if not has_endpoint:
            if self.models is None:
                return None
            return set(self.models)
        if unrestricted:
            return None
        return collected

    def exposed_model_ids_for_family(
        self, family: str,
    ) -> set[str] | None:
        """按入站协议族返回可暴露模型；None 表示不限制

        family: "openai" | "anthropic"
        auto 且无对应子表时返回空集合（该协议族不可路由）。
        单协议 provider 因可跨协议转换，两族共用同一白名单。
        """
        if self.protocol == "auto":
            endpoint = (
                self.openai if family == "openai" else self.anthropic
            )
            if endpoint is None:
                return set()
            effective = (
                endpoint.models
                if endpoint.models is not None
                else self.models
            )
            if effective is None:
                return None
            return set(effective)

        if self.models is None:
            return None
        return set(self.models)

    def allows_model_for_alias(self, upstream_model: str) -> bool:
        """model_aliases 目标模型是否可被该 provider 至少一条路径接受"""
        if self.protocol != "auto":
            return (
                self.models is None
                or upstream_model in self.models
            )

        for endpoint in (self.openai, self.anthropic):
            if endpoint is None:
                continue
            effective = (
                endpoint.models
                if endpoint.models is not None
                else self.models
            )
            if effective is None or upstream_model in effective:
                return True
        return False

    def allows_model_for_family(
        self, upstream_model: str, family: str,
    ) -> bool:
        """模型是否可在指定入站协议族下路由"""
        exposed = self.exposed_model_ids_for_family(family)
        if exposed is None:
            return True
        return upstream_model in exposed


@dataclass
class GatewayConfig:
    server: ServerConfig
    providers: dict[str, ProviderConfig]
    model_aliases: dict[str, str]

    @property
    def default_provider(self) -> str | None:
        return self.server.default_provider
