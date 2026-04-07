"""FastAPI 实例创建与路由注册"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from .errors import GatewayError, ProtocolError, format_error_response
from .models import GatewayConfig, ProtocolFormat
from .pipeline import PROVIDER_HEADER, process_request
from .upstream.client import UpstreamClient

logger = logging.getLogger(__name__)

_FORWARD_HEADER_NAMES = frozenset({"anthropic-beta", "anthropic-version"})


def _extract_forward_headers(request: Request) -> dict[str, str] | None:
    """提取需要透传给上游的请求头（anthropic-beta 等）"""
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() in _FORWARD_HEADER_NAMES
    }
    return headers or None


def create_app(config: GatewayConfig) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        clients: dict[str, UpstreamClient] = {}
        for name, cfg in config.providers.items():
            clients[name] = UpstreamClient(cfg)
            logger.info(
                "Initialized upstream client: %s (%s)",
                name, cfg.protocol_format.value,
            )
        app.state.providers = config.providers
        app.state.upstream_clients = clients
        app.state.model_aliases = config.model_aliases
        app.state.default_provider = config.default_provider
        if config.model_aliases:
            logger.info("Loaded %d model alias(es)", len(config.model_aliases))
        if config.default_provider:
            logger.info(
                "Default provider: %s", config.default_provider,
            )
        yield
        for client in clients.values():
            await client.close()

    app = FastAPI(title="kllmgate", lifespan=lifespan)

    @app.post("/openai/chat/completions")
    async def openai_chat(request: Request):
        try:
            try:
                body = await request.json()
            except json.JSONDecodeError as e:
                raise ProtocolError(f"invalid JSON body: {e}") from e
            return await process_request(
                ProtocolFormat.OPENAI_CHAT,
                body,
                request.app.state.providers,
                request.app.state.upstream_clients,
                header_provider=request.headers.get(PROVIDER_HEADER),
                model_aliases=request.app.state.model_aliases,
                default_provider=request.app.state.default_provider,
                forward_headers=_extract_forward_headers(request),
            )
        except GatewayError as e:
            return format_error_response(
                e, ProtocolFormat.OPENAI_CHAT,
            )

    @app.post("/openai/responses")
    async def openai_responses(request: Request):
        try:
            try:
                body = await request.json()
            except json.JSONDecodeError as e:
                raise ProtocolError(f"invalid JSON body: {e}") from e
            return await process_request(
                ProtocolFormat.OPENAI_RESPONSES,
                body,
                request.app.state.providers,
                request.app.state.upstream_clients,
                header_provider=request.headers.get(PROVIDER_HEADER),
                model_aliases=request.app.state.model_aliases,
                default_provider=request.app.state.default_provider,
                forward_headers=_extract_forward_headers(request),
            )
        except GatewayError as e:
            return format_error_response(
                e, ProtocolFormat.OPENAI_RESPONSES,
            )

    @app.post("/anthropic/v1/messages")
    async def anthropic_messages(request: Request):
        try:
            try:
                body = await request.json()
            except json.JSONDecodeError as e:
                raise ProtocolError(f"invalid JSON body: {e}") from e
            return await process_request(
                ProtocolFormat.ANTHROPIC_MESSAGES,
                body,
                request.app.state.providers,
                request.app.state.upstream_clients,
                header_provider=request.headers.get(PROVIDER_HEADER),
                model_aliases=request.app.state.model_aliases,
                default_provider=request.app.state.default_provider,
                forward_headers=_extract_forward_headers(request),
            )
        except GatewayError as e:
            return format_error_response(
                e, ProtocolFormat.ANTHROPIC_MESSAGES,
            )

    @app.api_route(
        "/{prefix}",
        methods=["HEAD", "GET"],
        include_in_schema=False,
    )
    async def health_check(prefix: str):
        """Claude Code 等客户端启动时会对 base URL 发 HEAD 探测"""
        if prefix not in ("anthropic", "openai"):
            return Response(status_code=404)
        return Response(status_code=200)

    return app
