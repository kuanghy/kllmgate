"""上游 HTTP 客户端"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx

from ..errors import UpstreamError, UpstreamHTTPError
from ..models import ProviderConfig
from ..sse import SseEvent, parse_sse_events

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}

_ENDPOINT_MAP = {
    ("openai", "chat"): "/chat/completions",
    ("openai", "responses"): "/responses",
    ("anthropic", "messages"): "/v1/messages",
}


class UpstreamClient:

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.timeout_seconds, connect=10.0,
            ),
        )
        key = (config.protocol, config.wire_api)
        suffix = _ENDPOINT_MAP.get(key, "/chat/completions")
        self._endpoint = f"{config.base_url}{suffix}"

    def _build_headers(self) -> dict[str, str]:
        api_key = self.config.resolve_api_key()
        headers = {"Content-Type": "application/json"}
        if self.config.protocol == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def send(self, body: dict) -> dict:
        """发送非流式请求，返回解析后的 JSON 响应"""
        headers = self._build_headers()
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self._client.post(
                    self._endpoint, json=body, headers=headers,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < self.config.max_retries:
                    wait = self._backoff(attempt)
                    logger.warning(
                        "Connect error (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1,
                        self.config.max_retries + 1,
                        wait, e,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise UpstreamError(
                    f"connect failed after {self.config.max_retries + 1}"
                    f" attempts: {e}",
                    code="upstream_connect_error",
                ) from e

            if resp.status_code in _RETRYABLE_STATUS_CODES:
                if attempt < self.config.max_retries:
                    wait = self._backoff(attempt)
                    logger.warning(
                        "Retryable HTTP %d (attempt %d/%d), "
                        "retrying in %.1fs",
                        resp.status_code,
                        attempt + 1,
                        self.config.max_retries + 1,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise UpstreamHTTPError(resp.status_code, resp.text)

            if resp.status_code >= 400:
                raise UpstreamHTTPError(resp.status_code, resp.text)

            return resp.json()

        raise UpstreamError(
            f"exhausted retries: {last_error}",
            code="upstream_connect_error",
        )

    async def send_stream(
        self, body: dict,
    ) -> AsyncIterator[SseEvent]:
        """发送流式请求，yield 已完成分帧的 SSE 事件"""
        headers = self._build_headers()

        resp = await self._client.send(
            self._client.build_request(
                "POST", self._endpoint, json=body, headers=headers,
            ),
            stream=True,
        )

        try:
            if resp.status_code >= 400:
                body_text = (await resp.aread()).decode(
                    errors="replace",
                )
                raise UpstreamHTTPError(resp.status_code, body_text)

            async def _line_iter() -> AsyncIterator[str]:
                async for line in resp.aiter_lines():
                    yield line

            async for event in parse_sse_events(_line_iter()):
                yield event
        finally:
            await resp.aclose()

    async def close(self):
        await self._client.aclose()

    @staticmethod
    def _backoff(attempt: int) -> float:
        if attempt == 0:
            return 0
        return min(2 ** (attempt - 1), 8)
