"""同协议工具适配转换器：OpenAI Chat → OpenAI Chat（tool_style 适配）"""

from __future__ import annotations

from collections.abc import AsyncIterator

from . import Converter
from .openai_responses_to_openai_chat import (
    OpenaiResponsesToOpenaiChatConverter,
)
from ..sse import SseEvent, format_sse


class OpenaiChatToolAdaptConverter(Converter):
    """当 inbound 和 upstream 都是 openai.chat 但 tool_style 非 standard 时使用"""

    def convert_request(self, body: dict, model: str) -> dict:
        result = {**body, "model": model}
        tools = result.pop("tools", None)

        if tools:
            prompt_add, tools_field = (
                self.tool_adapter.convert_tool_definitions(tools)
            )
            if prompt_add:
                messages = list(result.get("messages", []))
                if (
                    messages
                    and messages[0].get("role") == "system"
                ):
                    messages[0] = {
                        **messages[0],
                        "content": (
                            messages[0]["content"]
                            + "\n\n" + prompt_add
                        ),
                    }
                else:
                    messages.insert(0, {
                        "role": "system",
                        "content": prompt_add,
                    })
                result["messages"] = messages
            elif tools_field:
                result["tools"] = tools_field

        return result

    def convert_response(self, response: dict) -> dict:
        choices = response.get("choices", [])
        if not choices:
            return response

        choice = choices[0]
        message = choice.get("message", {})
        content_text, tool_calls = (
            self.tool_adapter.extract_tool_calls(message)
        )

        if tool_calls:
            new_message = {**message, "content": content_text}
            new_message["tool_calls"] = [{
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            } for tc in tool_calls]
            new_choice = {**choice, "message": new_message}
            if choice.get("finish_reason") == "stop":
                new_choice["finish_reason"] = "tool_calls"
            return {**response, "choices": [new_choice]}

        return response

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        async for event in upstream_events:
            yield format_sse(event.data, event.event)
