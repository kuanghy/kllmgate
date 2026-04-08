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

_RETRYABLE_STATUS_CODES = {500, 502, 503}
_MAX_429_RETRY_AFTER = 60

_FORWARD_UPSTREAM_HEADERS = frozenset({
    "retry-after",
    "x-request-id",
    "request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
})

_ERROR_LOG_HEADERS = (
    "x-request-id",
    "request-id",
    "cf-ray",
    "openai-processing-ms",
)

_RATELIMIT_LOG_HEADERS = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-tokens-remaining",
    "anthropic-ratelimit-requests-reset",
    "anthropic-ratelimit-tokens-reset",
)

_ENDPOINT_MAP = {
    ("openai", "chat"): "/chat/completions",
    ("openai", "responses"): "/responses",
    ("anthropic", "messages"): "/v1/messages",
}


def _collect_forward_headers(
    resp: httpx.Response,
) -> dict[str, str]:
    return {
        k: v for k, v in resp.headers.items()
        if k.lower() in _FORWARD_UPSTREAM_HEADERS
    }


def _parse_retry_after(value: str | None) -> float | None:
    """解析 Retry-After 头，返回等待秒数，无效值返回 None"""
    if not value:
        return None
    try:
        seconds = float(value)
        return seconds if seconds >= 0 else None
    except ValueError:
        return None


def _log_error_headers(resp: httpx.Response) -> None:
    """记录有助于问题定位的上游响应头"""
    names = _ERROR_LOG_HEADERS
    if resp.status_code == 429:
        names = names + _RATELIMIT_LOG_HEADERS
    parts = [
        f"{name}={resp.headers[name]}"
        for name in names
        if name in resp.headers
    ]
    if parts:
        logger.warning(
            "Upstream HTTP %d response headers: %s",
            resp.status_code, ", ".join(parts),
        )


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

    def _build_headers(
        self, extra_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        api_key = self.config.resolve_api_key()
        headers = {"Content-Type": "application/json"}
        if self.config.protocol == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
            if extra_headers:
                headers.update(extra_headers)
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def send(
        self, body: dict,
        extra_headers: dict[str, str] | None = None,
    ) -> dict:
        """发送非流式请求，返回解析后的 JSON 响应"""
        headers = self._build_headers(extra_headers)
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self._client.post(
                    self._endpoint, json=body, headers=headers,
                )
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    wait = self._backoff(attempt)
                    logger.warning(
                        "Request error (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1,
                        self.config.max_retries + 1,
                        wait, e,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise UpstreamError(
                    f"request failed after {self.config.max_retries + 1}"
                    f" attempts: {e}",
                    code="upstream_request_error",
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
                _log_error_headers(resp)
                raise UpstreamHTTPError(
                    resp.status_code, resp.text,
                    _collect_forward_headers(resp),
                )

            if resp.status_code == 429:
                retry_after = _parse_retry_after(
                    resp.headers.get("retry-after"),
                )
                if (retry_after is not None
                        and retry_after < _MAX_429_RETRY_AFTER
                        and attempt < self.config.max_retries):
                    logger.warning(
                        "HTTP 429 with Retry-After: %.1fs "
                        "(attempt %d/%d), retrying",
                        retry_after,
                        attempt + 1,
                        self.config.max_retries + 1,
                    )
                    await asyncio.sleep(retry_after)
                    continue
                _log_error_headers(resp)
                raise UpstreamHTTPError(
                    resp.status_code, resp.text,
                    _collect_forward_headers(resp),
                )

            if resp.status_code >= 400:
                _log_error_headers(resp)
                raise UpstreamHTTPError(
                    resp.status_code, resp.text,
                    _collect_forward_headers(resp),
                )

            return resp.json()

        raise UpstreamError(
            f"exhausted retries: {last_error}",
            code="upstream_request_error",
        )

    async def send_stream(
        self, body: dict,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[SseEvent]:
        """发送流式请求，yield 已完成分帧的 SSE 事件

        仅在连接建立阶段（状态码返回前）和收到可重试状态码时重试。
        一旦开始 yield 事件，中断将直接抛出异常，不再重试，
        避免重试导致事件重复发送。
        """
        headers = self._build_headers(extra_headers)
        last_error: Exception | None = None
        streaming_started = False

        for attempt in range(self.config.max_retries + 1):
            resp: httpx.Response | None = None
            try:
                resp = await self._client.send(
                    self._client.build_request(
                        "POST", self._endpoint, json=body, headers=headers,
                    ),
                    stream=True,
                )

                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    body_text = (await resp.aread()).decode(errors="replace")
                    if attempt < self.config.max_retries:
                        wait = self._backoff(attempt)
                        logger.warning(
                            "Retryable stream HTTP %d (attempt %d/%d), "
                            "retrying in %.1fs",
                            resp.status_code,
                            attempt + 1,
                            self.config.max_retries + 1,
                            wait,
                        )
                        await resp.aclose()
                        await asyncio.sleep(wait)
                        continue
                    _log_error_headers(resp)
                    raise UpstreamHTTPError(
                        resp.status_code, body_text,
                        _collect_forward_headers(resp),
                    )

                if resp.status_code == 429:
                    body_text = (await resp.aread()).decode(
                        errors="replace",
                    )
                    retry_after = _parse_retry_after(
                        resp.headers.get("retry-after"),
                    )
                    if (retry_after is not None
                            and retry_after < _MAX_429_RETRY_AFTER
                            and attempt < self.config.max_retries):
                        logger.warning(
                            "Stream HTTP 429 with Retry-After: "
                            "%.1fs (attempt %d/%d), retrying",
                            retry_after,
                            attempt + 1,
                            self.config.max_retries + 1,
                        )
                        await resp.aclose()
                        await asyncio.sleep(retry_after)
                        continue
                    _log_error_headers(resp)
                    raise UpstreamHTTPError(
                        resp.status_code, body_text,
                        _collect_forward_headers(resp),
                    )

                if resp.status_code >= 400:
                    body_text = (await resp.aread()).decode(
                        errors="replace",
                    )
                    _log_error_headers(resp)
                    raise UpstreamHTTPError(
                        resp.status_code, body_text,
                        _collect_forward_headers(resp),
                    )

                async def _line_iter() -> AsyncIterator[str]:
                    async for line in resp.aiter_lines():
                        yield line

                async for event in parse_sse_events(_line_iter()):
                    streaming_started = True
                    yield event
                return
            except httpx.RequestError as e:
                if streaming_started:
                    raise UpstreamError(
                        f"stream interrupted after partial delivery: {e}",
                        code="upstream_stream_interrupted",
                    ) from e
                last_error = e
                if attempt < self.config.max_retries:
                    wait = self._backoff(attempt)
                    logger.warning(
                        "Stream request error (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1,
                        self.config.max_retries + 1,
                        wait,
                        e,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise UpstreamError(
                    f"stream request failed after "
                    f"{self.config.max_retries + 1} attempts: {e}",
                    code="upstream_request_error",
                ) from e
            finally:
                if resp is not None:
                    await resp.aclose()

        raise UpstreamError(
            f"stream exhausted retries: {last_error}",
            code="upstream_request_error",
        )

    async def close(self):
        await self._client.aclose()

    @staticmethod
    def _backoff(attempt: int) -> float:
        if attempt == 0:
            return 0
        return min(2 ** (attempt - 1), 8)
