"""Anthropic Messages → OpenAI Chat Completions 转换器

响应方向（Chat → Anthropic）通过 tool_adapter / thinking_extractor 处理
文本中的工具调用和思考标签，转换为 Anthropic 的 tool_use / thinking 块。
"""

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


def _thinking_block_start(idx: int) -> str:
    return format_named_sse("content_block_start", {
        "type": "content_block_start",
        "index": idx,
        "content_block": {"type": "thinking", "thinking": ""},
    })


def _thinking_block_delta(idx: int, text: str) -> str:
    return format_named_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": idx,
        "delta": {"type": "thinking_delta", "thinking": text},
    })


def _text_block_start(idx: int) -> str:
    return format_named_sse("content_block_start", {
        "type": "content_block_start",
        "index": idx,
        "content_block": {"type": "text", "text": ""},
    })


def _text_block_delta(idx: int, text: str) -> str:
    return format_named_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": idx,
        "delta": {"type": "text_delta", "text": text},
    })


def _tool_block_start(idx: int, tc_id: str, name: str) -> str:
    return format_named_sse("content_block_start", {
        "type": "content_block_start",
        "index": idx,
        "content_block": {
            "type": "tool_use",
            "id": tc_id,
            "name": name,
            "input": {},
        },
    })


def _tool_block_delta(idx: int, partial_json: str) -> str:
    return format_named_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": idx,
        "delta": {
            "type": "input_json_delta",
            "partial_json": partial_json,
        },
    })


def _block_stop(idx: int) -> str:
    return format_named_sse("content_block_stop", {
        "type": "content_block_stop",
        "index": idx,
    })


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

    # ── 非流式响应 ──

    def convert_response(self, response: dict) -> dict:
        choices = response.get("choices", [])
        if not choices:
            return self._empty_anthropic(response)

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        raw_content = message.get("content", "") or ""
        existing_reasoning = message.get("reasoning_content")
        existing_tool_calls = message.get("tool_calls", [])

        if existing_reasoning is not None:
            reasoning = existing_reasoning
            text_for_tools = raw_content
        else:
            reasoning, text_for_tools = (
                self.thinking_extractor.extract(raw_content)
            )

        if existing_tool_calls:
            content_text = text_for_tools
            tool_calls_raw = existing_tool_calls
        else:
            content_text, extracted = (
                self.tool_adapter.extract_tool_calls(
                    {"content": text_for_tools},
                )
            )
            tool_calls_raw = [{
                "id": tc["id"],
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            } for tc in extracted]

        content_blocks: list[dict] = []
        if reasoning:
            content_blocks.append(
                {"type": "thinking", "thinking": reasoning},
            )
        if content_text:
            content_blocks.append(
                {"type": "text", "text": content_text},
            )
        for tc in tool_calls_raw:
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

        if tool_calls_raw:
            stop_reason = "tool_use"
        else:
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

    # ── 流式响应 ──

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        resp_id = ""
        model_name = ""
        content_index = 0
        block_open: str | None = None

        full_text = ""
        sent_pos = 0
        think_buf = self.thinking_extractor.stream_buffer_size
        phase = "content" if think_buf == 0 else "detecting"
        thinking_end_pos = 0

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
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                        },
                    },
                })
                if think_buf == 0:
                    yield _text_block_start(content_index)
                    block_open = "text"

            rc = delta.get("reasoning_content", "")
            if rc:
                if block_open != "thinking":
                    if block_open is not None:
                        yield _block_stop(content_index)
                        content_index += 1
                    yield _thinking_block_start(content_index)
                    block_open = "thinking"
                yield _thinking_block_delta(content_index, rc)

            delta_text = delta.get("content", "")
            if delta_text:
                full_text += delta_text

                if phase == "detecting":
                    open_tag = (
                        self.thinking_extractor.find_open_tag(full_text)
                    )
                    if open_tag is not None:
                        tag_start, content_start = open_tag
                        pre = full_text[sent_pos:tag_start]
                        if pre.strip():
                            if block_open != "text":
                                if block_open is not None:
                                    yield _block_stop(content_index)
                                    content_index += 1
                                yield _text_block_start(content_index)
                                block_open = "text"
                            yield _text_block_delta(content_index, pre)
                            yield _block_stop(content_index)
                            content_index += 1
                            block_open = None
                        elif block_open is not None:
                            yield _block_stop(content_index)
                            content_index += 1
                            block_open = None
                        yield _thinking_block_start(content_index)
                        block_open = "thinking"
                        sent_pos = content_start
                        phase = "thinking"
                    else:
                        safe = max(
                            sent_pos, len(full_text) - think_buf,
                        )
                        unsent = full_text[sent_pos:safe]
                        if unsent:
                            if block_open != "text":
                                if block_open is not None:
                                    yield _block_stop(content_index)
                                    content_index += 1
                                yield _text_block_start(content_index)
                                block_open = "text"
                            yield _text_block_delta(
                                content_index, unsent,
                            )
                            sent_pos = safe

                if phase == "thinking":
                    close_tag = (
                        self.thinking_extractor.find_close_tag(
                            full_text, sent_pos,
                        )
                    )
                    if close_tag is not None:
                        content_end, tag_end = close_tag
                        unsent = full_text[sent_pos:content_end]
                        if unsent:
                            if block_open != "thinking":
                                if block_open is not None:
                                    yield _block_stop(content_index)
                                    content_index += 1
                                yield _thinking_block_start(
                                    content_index,
                                )
                                block_open = "thinking"
                            yield _thinking_block_delta(
                                content_index, unsent,
                            )
                        if block_open == "thinking":
                            yield _block_stop(content_index)
                            content_index += 1
                            block_open = None
                        thinking_end_pos = tag_end
                        sent_pos = tag_end
                        phase = "content"
                    else:
                        safe = max(
                            sent_pos,
                            len(full_text) - think_buf,
                        )
                        unsent = full_text[sent_pos:safe]
                        if unsent:
                            if block_open != "thinking":
                                if block_open is not None:
                                    yield _block_stop(content_index)
                                    content_index += 1
                                yield _thinking_block_start(
                                    content_index,
                                )
                                block_open = "thinking"
                            yield _thinking_block_delta(
                                content_index, unsent,
                            )
                            sent_pos = safe

                if phase == "content":
                    content_area = full_text[thinking_end_pos:]
                    boundary = (
                        self.tool_adapter
                        .detect_stream_tool_boundary(content_area)
                    )
                    if boundary is not None:
                        safe = thinking_end_pos + boundary
                    elif self.tool_adapter.stream_buffer_size > 0:
                        safe = max(
                            sent_pos,
                            len(full_text)
                            - self.tool_adapter.stream_buffer_size,
                        )
                    else:
                        safe = len(full_text)
                    unsent = full_text[sent_pos:safe]
                    if unsent:
                        if block_open != "text":
                            if block_open is not None:
                                yield _block_stop(content_index)
                                content_index += 1
                            yield _text_block_start(content_index)
                            block_open = "text"
                        yield _text_block_delta(
                            content_index, unsent,
                        )
                        sent_pos = safe

            if "tool_calls" in delta:
                if block_open is not None:
                    yield _block_stop(content_index)
                    content_index += 1
                    block_open = None
                for tc_delta in delta["tool_calls"]:
                    func = tc_delta.get("function", {})
                    if func.get("name"):
                        yield _tool_block_start(
                            content_index,
                            tc_delta.get("id", ""),
                            func["name"],
                        )
                        block_open = "tool_use"
                    if func.get("arguments"):
                        yield _tool_block_delta(
                            content_index,
                            func["arguments"],
                        )

            if finish_reason:
                for ev in self._finish_stream(
                    finish_reason, chunk,
                    full_text, sent_pos, phase,
                    thinking_end_pos, block_open, content_index,
                ):
                    yield ev
                return

        if block_open is not None:
            yield _block_stop(content_index)
        yield format_named_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 0},
        })
        yield format_named_sse("message_stop", {
            "type": "message_stop",
        })

    def _finish_stream(
        self,
        finish_reason: str,
        chunk: dict,
        full_text: str,
        sent_pos: int,
        phase: str,
        thinking_end_pos: int,
        block_open: str | None,
        content_index: int,
    ):
        """生成流式响应结尾事件"""
        tool_calls: list[dict] = []

        if phase == "thinking":
            unsent = full_text[sent_pos:]
            if unsent:
                if block_open != "thinking":
                    if block_open is not None:
                        yield _block_stop(content_index)
                        content_index += 1
                    yield _thinking_block_start(content_index)
                    block_open = "thinking"
                yield _thinking_block_delta(content_index, unsent)
            if block_open is not None:
                yield _block_stop(content_index)
                content_index += 1
                block_open = None

        elif phase == "detecting":
            reasoning, remaining = (
                self.thinking_extractor.extract(full_text)
            )
            if reasoning:
                if block_open is not None:
                    yield _block_stop(content_index)
                    content_index += 1
                    block_open = None
                yield _thinking_block_start(content_index)
                yield _thinking_block_delta(content_index, reasoning)
                yield _block_stop(content_index)
                content_index += 1
            clean_text, extracted = (
                self.tool_adapter.extract_tool_calls(
                    {"content": remaining},
                )
            )
            tool_calls = extracted
            remaining_content = clean_text[sent_pos:]
            if remaining_content:
                if block_open != "text":
                    if block_open is not None:
                        yield _block_stop(content_index)
                        content_index += 1
                    yield _text_block_start(content_index)
                    block_open = "text"
                yield _text_block_delta(
                    content_index, remaining_content,
                )
            if block_open is not None:
                yield _block_stop(content_index)
                content_index += 1
                block_open = None

        else:
            content_portion = full_text[thinking_end_pos:]
            clean_text, extracted = (
                self.tool_adapter.extract_tool_calls(
                    {"content": content_portion},
                )
            )
            tool_calls = extracted
            content_sent = max(0, sent_pos - thinking_end_pos)
            remaining_content = clean_text[content_sent:]
            if remaining_content:
                if block_open != "text":
                    if block_open is not None:
                        yield _block_stop(content_index)
                        content_index += 1
                    yield _text_block_start(content_index)
                    block_open = "text"
                yield _text_block_delta(
                    content_index, remaining_content,
                )
            if block_open is not None:
                yield _block_stop(content_index)
                content_index += 1
                block_open = None

        for tc in tool_calls:
            args_str = tc["arguments"]
            yield _tool_block_start(
                content_index, tc["id"], tc["name"],
            )
            yield _tool_block_delta(content_index, args_str)
            yield _block_stop(content_index)
            content_index += 1

        if tool_calls:
            stop_reason = "tool_use"
        else:
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
