"""同协议工具适配转换器：OpenAI Responses → OpenAI Responses（tool_style 适配）

消息结构不做变换，仅通过 tool_adapter 处理工具定义注入和工具调用解析。
实现方式：委托 openai_responses_to_openai_chat 的请求解析，
再将 Chat 请求转回 Responses 格式。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from . import Converter
from ..sse import SseEvent, format_named_sse


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
        output = []
        for item in response.get("output", []):
            if item.get("type") != "message":
                output.append(item)
                continue

            text_parts = []
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
            raw_text = "".join(text_parts)
            clean_text, tool_calls = self.tool_adapter.extract_tool_calls(
                {"content": raw_text},
            )

            normalized_message = {
                **item,
                "content": [{
                    "type": "output_text",
                    "text": clean_text,
                }],
            }
            output.append(normalized_message)

            for tc in tool_calls:
                output.append({
                    "type": "function_call",
                    "id": f"fc_{tc['id']}",
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "status": "completed",
                })

        return {**response, "output": output}

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        seq = 0
        full_text = ""
        sent_pos = 0
        xml_detected = False
        message_item: dict | None = None
        response_obj: dict | None = None

        def sse(event_name: str, data: dict) -> str:
            nonlocal seq
            data["sequence_number"] = seq
            seq += 1
            return format_named_sse(event_name, data)

        async for event in upstream_events:
            if not event.event:
                continue

            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            if event.event in {"response.created", "response.in_progress"}:
                if event.event == "response.created":
                    response_obj = data.get("response", {})
                yield sse(event.event, data)
                continue

            if event.event == "response.output_item.added":
                item = data.get("item", {})
                if item.get("type") == "message" and message_item is None:
                    message_item = item
                yield sse(event.event, data)
                continue

            if event.event == "response.content_part.added":
                yield sse(event.event, data)
                continue

            if event.event == "response.output_text.delta":
                delta = data.get("delta", "")
                full_text += delta
                boundary = self.tool_adapter.detect_stream_tool_boundary(
                    full_text,
                )
                if boundary is not None:
                    xml_detected = True
                    safe_end = boundary
                else:
                    safe_end = max(
                        sent_pos, len(full_text) - len("<minimax:tool_call>"),
                    )
                unsent = full_text[sent_pos:safe_end]
                if unsent:
                    new_data = {
                        **data,
                        "delta": unsent,
                    }
                    yield sse(event.event, new_data)
                    sent_pos = safe_end
                continue

            if event.event in {
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
            }:
                continue

            if event.event == "response.completed":
                clean_text, tool_calls = self.tool_adapter.extract_tool_calls(
                    {"content": full_text},
                )
                remaining = clean_text[sent_pos:]
                if remaining:
                    yield sse("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": (message_item or {}).get("id", ""),
                        "output_index": 0,
                        "content_index": 0,
                        "delta": remaining,
                    })

                msg_id = (message_item or {}).get("id", "")
                yield sse("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": clean_text,
                })
                yield sse("response.content_part.done", {
                    "type": "response.content_part.done",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": clean_text},
                })

                completed_msg = {
                    **(message_item or {}),
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": clean_text,
                    }],
                }
                yield sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": completed_msg,
                })

                output_items = [completed_msg]
                output_index = 1
                for tc in tool_calls:
                    fc_item = {
                        "type": "function_call",
                        "id": f"fc_{tc['id']}",
                        "call_id": tc["id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "status": "completed",
                    }
                    yield sse("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {**fc_item, "status": "in_progress", "arguments": ""},
                    })
                    yield sse("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": fc_item["id"],
                        "output_index": output_index,
                        "delta": tc["arguments"],
                    })
                    yield sse("response.function_call_arguments.done", {
                        "type": "response.function_call_arguments.done",
                        "item_id": fc_item["id"],
                        "output_index": output_index,
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    })
                    yield sse("response.output_item.done", {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": fc_item,
                    })
                    output_items.append(fc_item)
                    output_index += 1

                completed_response = data.get("response", {})
                completed_response["output"] = output_items
                yield sse("response.completed", {
                    "type": "response.completed",
                    "response": completed_response,
                })
                return

            yield sse(event.event, data)
