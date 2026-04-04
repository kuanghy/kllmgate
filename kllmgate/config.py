"""TOML 配置加载与校验"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .errors import ConfigError
from .models import ProviderConfig

_VALID_PROTOCOLS = {"openai", "anthropic"}
_VALID_WIRE_APIS = {"chat", "responses", "messages"}


def load_config(path: str) -> dict[str, ProviderConfig]:
    """加载并校验 TOML 配置文件，返回 provider 名称到配置的映射"""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(
            f"config file not found: {path}",
            code="config_file_not_found",
        )

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    if "providers" not in raw:
        raise ConfigError(
            "config file missing 'providers' section",
            code="missing_providers",
        )

    providers_raw = raw["providers"]
    if not providers_raw:
        raise ConfigError(
            "no provider configured in config file",
            code="empty_providers",
        )

    providers: dict[str, ProviderConfig] = {}
    for name, section in providers_raw.items():
        providers[name] = _parse_provider(name, section)

    return providers


def _parse_provider(name: str, section: dict) -> ProviderConfig:
    if "base_url" not in section:
        raise ConfigError(
            f"provider {name!r}: missing required field 'base_url'",
            code="missing_required_field",
        )
    if "protocol" not in section:
        raise ConfigError(
            f"provider {name!r}: missing required field 'protocol'",
            code="missing_required_field",
        )

    protocol = section["protocol"]
    if protocol not in _VALID_PROTOCOLS:
        raise ConfigError(
            f"provider {name!r}: invalid protocol {protocol!r}, "
            f"must be one of {_VALID_PROTOCOLS}",
            code="invalid_protocol",
        )

    wire_api = section.get("wire_api")
    if protocol == "anthropic" and wire_api and wire_api != "messages":
        raise ConfigError(
            f"provider {name!r}: anthropic protocol does not support "
            f"wire_api={wire_api!r}, must be 'messages' or omitted",
            code="invalid_wire_api",
        )

    api_key = section.get("api_key")
    env_key = section.get("env_key")
    if not api_key and not env_key:
        raise ConfigError(
            f"provider {name!r}: API key not configured, "
            f"set 'api_key' or 'env_key'",
            code="api_key_not_configured",
        )

    base_url = section["base_url"].rstrip("/")

    return ProviderConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        env_key=env_key,
        protocol=protocol,
        wire_api=wire_api,
        tool_style=section.get("tool_style", "standard"),
        timeout_seconds=section.get("timeout_seconds", 120),
        max_retries=section.get("max_retries", 2),
        models=section.get("models"),
    )
