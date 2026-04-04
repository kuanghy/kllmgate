"""同协议工具适配转换器：OpenAI Responses → OpenAI Responses（tool_style 适配）

消息结构不做变换，仅通过 tool_adapter 处理工具定义注入和工具调用解析。
实现方式：委托 openai_responses_to_openai_chat 的请求解析，
再将 Chat 请求转回 Responses 格式。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from . import Converter
from ..sse import SseEvent, format_sse


class OpenaiResponsesToolAdaptConverter(Converter):

    def convert_request(self, body: dict, model: str) -> dict:
        result = {**body, "model": model}
        tools = result.pop("tools", None)

        if tools:
            prompt_add, tools_field = (
                self.tool_adapter.convert_tool_definitions(tools)
            )
            if prompt_add:
                instructions = result.get("instructions", "")
                if instructions:
                    result["instructions"] = (
                        instructions + "\n\n" + prompt_add
                    )
                else:
                    result["instructions"] = prompt_add
            elif tools_field:
                result["tools"] = tools_field

        return result

    def convert_response(self, response: dict) -> dict:
        return response

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        async for event in upstream_events:
            yield format_sse(event.data, event.event)
