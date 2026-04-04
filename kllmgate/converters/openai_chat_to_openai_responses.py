"""OpenAI Chat Completions → OpenAI Responses API 转换器"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from . import Converter
from ._helpers import (
    make_resp_id,
    make_msg_id,
    now_ts,
    responses_usage_to_chat,
    STATUS_TO_FINISH_REASON,
)
from ..sse import SseEvent, format_data_only_sse


class OpenaiChatToOpenaiResponsesConverter(Converter):

    def convert_request(self, body: dict, model: str) -> dict:
        messages = body.get("messages", [])
        system_parts: list[str] = []
        input_items: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")

            if role == "system":
                system_parts.append(msg.get("content", ""))
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", ""),
                        })
                    continue
                input_items.append({
                    "role": "model",
                    "content": msg.get("content", ""),
                })
                continue

            if role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })
                continue

            input_items.append({
                "role": role,
                "content": msg.get("content", ""),
            })

        result: dict = {"model": model, "input": input_items}

        if system_parts:
            result["instructions"] = "\n\n".join(system_parts)

        if body.get("tools"):
            result["tools"] = body["tools"]
        if body.get("stream"):
            result["stream"] = True

        return result

    def convert_response(self, response: dict) -> dict:
        output = response.get("output", [])
        status = response.get("status", "completed")

        content_text = ""
        tool_calls = []

        for item in output:
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        content_text += part.get("text", "")
            elif item.get("type") == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                })

        if tool_calls:
            finish_reason = "tool_calls"
        elif status == "completed":
            finish_reason = "stop"
        else:
            finish_reason = STATUS_TO_FINISH_REASON.get(status, "stop")

        message: dict = {
            "role": "assistant",
            "content": content_text or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        usage = responses_usage_to_chat(response.get("usage", {}))

        return {
            "id": response.get("id", make_resp_id()),
            "object": "chat.completion",
            "created": response.get("created_at", now_ts()),
            "model": response.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        resp_id = ""
        model_name = ""

        async for event in upstream_events:
            if not event.event:
                continue

            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            if event.event == "response.created":
                resp_obj = data.get("response", {})
                resp_id = resp_obj.get("id", make_resp_id())
                model_name = resp_obj.get("model", "")
                yield format_data_only_sse({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                    }],
                })

            elif event.event == "response.output_text.delta":
                yield format_data_only_sse({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": data.get("delta", ""),
                        },
                    }],
                })

            elif event.event == "response.function_call_arguments.delta":
                yield format_data_only_sse({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "function": {
                                    "arguments": data.get("delta", ""),
                                },
                            }],
                        },
                    }],
                })

            elif event.event == "response.completed":
                resp_obj = data.get("response", {})
                usage = resp_obj.get("usage", {})
                chat_usage = responses_usage_to_chat(usage)
                yield format_data_only_sse({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                    "usage": chat_usage,
                })
                yield "data: [DONE]\n\n"
