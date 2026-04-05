"""FastAPI 实例创建与路由注册"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import load_config
from .errors import GatewayError, ProtocolError, format_error_response
from .models import ProtocolFormat
from .pipeline import PROVIDER_HEADER, process_request
from .upstream.client import UpstreamClient

logger = logging.getLogger(__name__)


def create_app(config_path: str = "config.toml") -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        providers, model_aliases = load_config(config_path)
        clients: dict[str, UpstreamClient] = {}
        for name, cfg in providers.items():
            clients[name] = UpstreamClient(cfg)
            logger.info(
                "Initialized upstream client: %s (%s)",
                name, cfg.protocol_format.value,
            )
        app.state.providers = providers
        app.state.upstream_clients = clients
        app.state.model_aliases = model_aliases
        if model_aliases:
            logger.info("Loaded %d model alias(es)", len(model_aliases))
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
            )
        except GatewayError as e:
            return format_error_response(
                e, ProtocolFormat.ANTHROPIC_MESSAGES,
            )

    return app
