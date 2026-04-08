"""Converter 基类与注册表"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..sse import SseEvent


class Converter(ABC):
    """协议转换器基类"""

    def __init__(self, tool_adapter, thinking_extractor=None):
        self.tool_adapter = tool_adapter
        if thinking_extractor is None:
            from ..thinking import NullThinkingExtractor
            thinking_extractor = NullThinkingExtractor()
        self.thinking_extractor = thinking_extractor

    @abstractmethod
    def convert_request(self, body: dict, model: str) -> dict:
        """将客户端请求体转换为上游请求体"""

    @abstractmethod
    def convert_response(self, response: dict) -> dict:
        """将上游非流式响应转换为客户端协议格式"""

    @abstractmethod
    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        """将上游 SSE 流转换为客户端协议的 SSE 流"""
        yield  # pragma: no cover
