"""OpenAI Chat Completions → Anthropic Messages 转换器"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from . import Converter
from ._helpers import (
    now_ts,
    normalize_text_content,
    anthropic_usage_to_chat,
    ANTHROPIC_STOP_TO_OPENAI,
)
from ..sse import SseEvent, format_data_only_sse


class OpenaiChatToAnthropicMessagesConverter(Converter):

    def convert_request(self, body: dict, model: str) -> dict:
        messages = body.get("messages", [])
        system_parts: list[str] = []
        anthropic_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_parts.append(normalize_text_content(content))
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    content_blocks = []
                    if content:
                        content_blocks.append(
                            {"type": "text", "text": content},
                        )
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str)
                        except (json.JSONDecodeError, ValueError):
                            args = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "input": args,
                        })
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })
                    continue
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content,
                })
                continue

            if role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }],
                })
                continue

            anthropic_messages.append({
                "role": "user", "content": content,
            })

        anthropic_messages = self._merge_consecutive(anthropic_messages)

        result: dict = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": body.get(
                "max_tokens",
                body.get("max_completion_tokens", 4096),
            ),
        }

        if system_parts:
            result["system"] = "\n\n".join(system_parts)

        if body.get("stream"):
            result["stream"] = True

        if body.get("tools"):
            _, tools = (
                self.tool_adapter.convert_tool_definitions(body["tools"])
            )
            if tools:
                result["tools"] = tools

        for key in ("temperature", "top_p"):
            if key in body:
                result[key] = body[key]

        return result

    @staticmethod
    def _merge_consecutive(messages: list[dict]) -> list[dict]:
        """合并 Anthropic 不允许的连续同角色消息"""
        if not messages:
            return messages
        merged = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                prev = merged[-1]["content"]
                curr = msg["content"]
                if isinstance(prev, str) and isinstance(curr, str):
                    merged[-1]["content"] = f"{prev}\n\n{curr}"
                else:
                    if isinstance(prev, str):
                        prev = [{"type": "text", "text": prev}]
                    if isinstance(curr, str):
                        curr = [{"type": "text", "text": curr}]
                    merged[-1]["content"] = prev + curr
            else:
                merged.append(msg)
        return merged

    def convert_response(self, response: dict) -> dict:
        content_blocks = response.get("content", [])
        stop_reason = response.get("stop_reason", "end_turn")
        usage = response.get("usage", {})

        content_text = ""
        tool_calls = []
        for block in content_blocks:
            if block.get("type") == "text":
                content_text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(block)

        message: dict = {
            "role": "assistant",
            "content": content_text or None,
        }
        if tool_calls:
            message["tool_calls"] = [{
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(
                        tc.get("input", {}), ensure_ascii=False,
                    ),
                },
            } for tc in tool_calls]

        finish_reason = ANTHROPIC_STOP_TO_OPENAI.get(
            stop_reason, "stop",
        )

        return {
            "id": response.get("id", ""),
            "object": "chat.completion",
            "created": now_ts(),
            "model": response.get("model", ""),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": anthropic_usage_to_chat(usage),
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

            if event.event == "message_start":
                msg = data.get("message", {})
                resp_id = msg.get("id", "")
                model_name = msg.get("model", "")
                yield format_data_only_sse({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                    }],
                })

            elif event.event == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield format_data_only_sse({
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": delta.get("text", ""),
                            },
                        }],
                    })
                elif delta.get("type") == "input_json_delta":
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
                                        "arguments": delta.get(
                                            "partial_json", "",
                                        ),
                                    },
                                }],
                            },
                        }],
                    })

            elif event.event == "message_delta":
                delta = data.get("delta", {})
                stop_reason = delta.get("stop_reason", "end_turn")
                finish_reason = ANTHROPIC_STOP_TO_OPENAI.get(
                    stop_reason, "stop",
                )
                usage = data.get("usage", {})
                yield format_data_only_sse({
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }],
                    "usage": anthropic_usage_to_chat(usage),
                })

            elif event.event == "message_stop":
                yield "data: [DONE]\n\n"
