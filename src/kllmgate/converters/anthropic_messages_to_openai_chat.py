"""Anthropic Messages → OpenAI Chat Completions 转换器"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from . import Converter
from ._helpers import (
    now_ts,
    chat_usage_to_anthropic,
    OPENAI_FINISH_TO_ANTHROPIC,
)
from ..sse import SseEvent, format_named_sse


class AnthropicMessagesToOpenaiChatConverter(Converter):

    def convert_request(self, body: dict, model: str) -> dict:
        messages: list[dict] = []

        if body.get("system"):
            messages.append({
                "role": "system",
                "content": body["system"],
            })

        for msg in body.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "assistant" and isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(
                                    block.get("input", {}),
                                    ensure_ascii=False,
                                ),
                            },
                        })
                chat_msg: dict = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) or None,
                }
                if tool_calls:
                    chat_msg["tool_calls"] = tool_calls
                messages.append(chat_msg)
                continue

            if role == "user" and isinstance(content, list):
                tool_results = []
                user_parts = []
                for block in content:
                    block_type = block.get("type", "")
                    if block_type == "tool_result":
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": block.get(
                                "tool_use_id", "",
                            ),
                            "content": block.get("content", ""),
                        })
                    elif block_type == "text":
                        user_parts.append({
                            "type": "text",
                            "text": block.get("text", ""),
                        })
                    elif block_type == "image":
                        source = block.get("source", {})
                        if source.get("type") == "url":
                            user_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": source.get("url", ""),
                                },
                            })
                        elif source.get("type") == "base64":
                            media = source.get("media_type", "image/png")
                            data = source.get("data", "")
                            user_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media};base64,{data}",
                                },
                            })
                        else:
                            user_parts.append(block)
                    else:
                        user_parts.append(block)
                messages.extend(tool_results)
                if user_parts:
                    messages.append({
                        "role": "user",
                        "content": user_parts,
                    })
                continue

            messages.append({"role": role, "content": content})

        result: dict = {"model": model, "messages": messages}

        if body.get("max_tokens"):
            result["max_tokens"] = body["max_tokens"]
        if body.get("stream"):
            result["stream"] = True
            result["stream_options"] = {"include_usage": True}

        if body.get("tools"):
            result["tools"] = [{
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            } for t in body["tools"]]

        for key in ("temperature", "top_p"):
            if key in body:
                result[key] = body[key]

        return result

    def convert_response(self, response: dict) -> dict:
        choices = response.get("choices", [])
        if not choices:
            return self._empty_anthropic(response)

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        content_text = message.get("content", "") or ""
        raw_tool_calls = message.get("tool_calls", [])

        content_blocks = []
        if content_text:
            content_blocks.append(
                {"type": "text", "text": content_text},
            )
        for tc in raw_tool_calls:
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

        stop_reason = OPENAI_FINISH_TO_ANTHROPIC.get(
            finish_reason, "end_turn",
        )
        usage = response.get("usage", {})

        return {
            "id": response.get("id", ""),
            "type": "message",
            "role": "assistant",
            "model": response.get("model", ""),
            "content": content_blocks or [
                {"type": "text", "text": ""},
            ],
            "stop_reason": stop_reason,
            "usage": chat_usage_to_anthropic(usage),
        }

    def _empty_anthropic(self, response: dict) -> dict:
        return {
            "id": response.get("id", ""),
            "type": "message",
            "role": "assistant",
            "model": response.get("model", ""),
            "content": [{"type": "text", "text": ""}],
            "stop_reason": "end_turn",
            "usage": chat_usage_to_anthropic(response.get("usage", {})),
        }

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        resp_id = ""
        model_name = ""
        content_index = 0

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

            delta = choices[0].get("delta", {})
            finish_reason = choices[0].get("finish_reason")

            if delta.get("role") == "assistant":
                yield format_named_sse("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": resp_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model_name,
                        "content": [],
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                })
                yield format_named_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": content_index,
                    "content_block": {"type": "text", "text": ""},
                })

            if "content" in delta and delta["content"]:
                yield format_named_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": content_index,
                    "delta": {
                        "type": "text_delta",
                        "text": delta["content"],
                    },
                })

            if "tool_calls" in delta:
                for tc_delta in delta["tool_calls"]:
                    func = tc_delta.get("function", {})
                    if func.get("name"):
                        content_index += 1
                        yield format_named_sse("content_block_start", {
                            "type": "content_block_start",
                            "index": content_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_delta.get("id", ""),
                                "name": func["name"],
                                "input": {},
                            },
                        })
                    if func.get("arguments"):
                        yield format_named_sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": content_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": func["arguments"],
                            },
                        })

            if finish_reason:
                yield format_named_sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": content_index,
                })
                stop_reason = OPENAI_FINISH_TO_ANTHROPIC.get(
                    finish_reason, "end_turn",
                )
                usage = chunk.get("usage", {})
                yield format_named_sse("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason},
                    "usage": chat_usage_to_anthropic(usage),
                })
                yield format_named_sse("message_stop", {
                    "type": "message_stop",
                })
