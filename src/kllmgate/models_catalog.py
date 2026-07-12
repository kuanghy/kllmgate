"""模型列表收集与 /models 端点响应构造"""

from __future__ import annotations

from .models import ProviderConfig

_ANTHROPIC_MODELS_CREATED_AT = "1970-01-01T00:00:00Z"
_UNSUPPORTED_CAPABILITY = {"supported": False}


def _is_routable_target(
    target: str,
    providers: dict[str, ProviderConfig],
    protocol_family: str | None = None,
) -> bool:
    """alias 目标是否在 provider 可路由范围内"""
    provider_name, upstream_model = target.split("/", 1)
    provider = providers[provider_name]
    if protocol_family is None:
        return provider.allows_model_for_alias(upstream_model)
    return provider.allows_model_for_family(upstream_model, protocol_family)


def collect_available_model_ids(
    providers: dict[str, ProviderConfig],
    model_aliases: dict[str, str],
    protocol_family: str | None = None,
) -> set[str]:
    """汇总 /models 可暴露的模型 ID

    protocol_family 为 None 时汇总全部路径（供 models_list 启动校验）。
    为 "openai" / "anthropic" 时仅包含该入站协议族可路由的模型。
    """
    models: set[str] = set()
    for alias, target in model_aliases.items():
        if _is_routable_target(
            target, providers, protocol_family=protocol_family,
        ):
            models.add(alias)
    for provider_name, provider in providers.items():
        if protocol_family is None:
            exposed = provider.exposed_model_ids()
        else:
            exposed = provider.exposed_model_ids_for_family(protocol_family)
        if exposed is None:
            continue
        for model_id in exposed:
            models.add(f"{provider_name}/{model_id}")
    return models


def filter_model_ids(
    model_ids: set[str],
    models_list: list[str] | None,
) -> list[str]:
    if models_list is not None:
        model_ids &= set(models_list)
    return sorted(model_ids)


def collect_listed_models(
    providers: dict[str, ProviderConfig],
    model_aliases: dict[str, str],
    models_list: list[str] | None = None,
    protocol_family: str | None = None,
) -> list[str]:
    """汇总网关对外暴露的模型 ID，并应用 models_list 白名单"""
    model_ids = collect_available_model_ids(
        providers, model_aliases, protocol_family=protocol_family,
    )
    return filter_model_ids(model_ids, models_list)


def openai_models_payload(
    providers: dict[str, ProviderConfig],
    model_aliases: dict[str, str],
    models_list: list[str] | None = None,
) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "kllmgate",
            }
            for model_id in collect_listed_models(
                providers,
                model_aliases,
                models_list=models_list,
                protocol_family="openai",
            )
        ],
    }


def _anthropic_model_entry(model_id: str) -> dict:
    return {
        "id": model_id,
        "type": "model",
        "display_name": model_id,
        "created_at": _ANTHROPIC_MODELS_CREATED_AT,
        "max_input_tokens": 0,
        "max_tokens": 0,
        "capabilities": {
            "batch": _UNSUPPORTED_CAPABILITY,
            "citations": _UNSUPPORTED_CAPABILITY,
            "code_execution": _UNSUPPORTED_CAPABILITY,
            "image_input": _UNSUPPORTED_CAPABILITY,
            "pdf_input": _UNSUPPORTED_CAPABILITY,
            "structured_outputs": _UNSUPPORTED_CAPABILITY,
            "thinking": {
                "supported": False,
                "types": {
                    "adaptive": _UNSUPPORTED_CAPABILITY,
                    "enabled": _UNSUPPORTED_CAPABILITY,
                },
            },
        },
    }


def anthropic_models_payload(
    providers: dict[str, ProviderConfig],
    model_aliases: dict[str, str],
    models_list: list[str] | None = None,
) -> dict:
    data = [
        _anthropic_model_entry(model_id)
        for model_id in collect_listed_models(
            providers,
            model_aliases,
            models_list=models_list,
            protocol_family="anthropic",
        )
    ]
    if not data:
        return {
            "data": [],
            "first_id": None,
            "last_id": None,
            "has_more": False,
        }
    return {
        "data": data,
        "first_id": data[0]["id"],
        "last_id": data[-1]["id"],
        "has_more": False,
    }
