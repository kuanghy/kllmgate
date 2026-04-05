"""同协议工具适配转换器：OpenAI Chat → OpenAI Chat（tool_style 适配）"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from . import Converter
from ..sse import SseEvent, format_data_only_sse


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
        resp_id = ""
        model_name = ""
        full_text = ""
        sent_pos = 0
        sent_role = False

        async for event in upstream_events:
            if event.data == "[DONE]":
                break
            try:
                chunk = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            resp_id = chunk.get("id", resp_id)
            model_name = chunk.get("model", model_name)
            choices = chunk.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            if delta.get("role") == "assistant" and not sent_role:
                sent_role = True
                yield format_data_only_sse({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                    }],
                })

            delta_text = delta.get("content", "")
            if delta_text:
                full_text += delta_text
                boundary = self.tool_adapter.detect_stream_tool_boundary(
                    full_text,
                )
                if boundary is not None:
                    safe_end = boundary
                else:
                    safe_end = max(
                        sent_pos, len(full_text) - len("<minimax:tool_call>"),
                    )
                unsent = full_text[sent_pos:safe_end]
                if unsent:
                    yield format_data_only_sse({
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": unsent},
                        }],
                    })
                    sent_pos = safe_end

            if finish_reason:
                clean_text, tool_calls = self.tool_adapter.extract_tool_calls(
                    {"content": full_text},
                )
                remaining = clean_text[sent_pos:]
                if remaining:
                    yield format_data_only_sse({
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": remaining},
                        }],
                    })
                final_delta: dict = {}
                mapped_finish_reason = finish_reason
                if tool_calls:
                    final_delta["tool_calls"] = [{
                        "index": index,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    } for index, tc in enumerate(tool_calls)]
                    mapped_finish_reason = "tool_calls"

                yield format_data_only_sse({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": final_delta,
                        "finish_reason": mapped_finish_reason,
                    }],
                })
                yield "data: [DONE]\n\n"
                return

        yield "data: [DONE]\n\n"
