"""直通转换器：协议相同且无需工具适配时使用"""

from __future__ import annotations

from collections.abc import AsyncIterator

from . import Converter
from ..sse import SseEvent, format_sse


class PassthroughConverter(Converter):

    def convert_request(self, body: dict, model: str) -> dict:
        result = {**body, "model": model}
        return result

    def convert_response(self, response: dict) -> dict:
        return response

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        async for event in upstream_events:
            yield format_sse(event.data, event.event)
