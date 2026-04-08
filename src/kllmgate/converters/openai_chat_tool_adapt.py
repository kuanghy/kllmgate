"""同协议工具适配转换器：OpenAI Chat → OpenAI Chat（tool_style + thinking_style 适配）"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from . import Converter
from ._helpers import normalize_text_content
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
                    existing = normalize_text_content(
                        messages[0]["content"],
                    )
                    messages[0] = {
                        **messages[0],
                        "content": existing + "\n\n" + prompt_add,
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

        if message.get("reasoning_content") is not None:
            reasoning = ""
            content_text, tool_calls = (
                self.tool_adapter.extract_tool_calls(message)
            )
        else:
            raw_content = message.get("content", "") or ""
            reasoning, remaining = (
                self.thinking_extractor.extract(raw_content)
            )
            content_text, tool_calls = (
                self.tool_adapter.extract_tool_calls(
                    {"content": remaining},
                )
            )

        if not tool_calls and not reasoning:
            return response

        new_message = {**message, "content": content_text}
        if reasoning:
            new_message["reasoning_content"] = reasoning
        new_choice = {**choice, "message": new_message}

        if tool_calls:
            new_message["tool_calls"] = [{
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            } for tc in tool_calls]
            if choice.get("finish_reason") == "stop":
                new_choice["finish_reason"] = "tool_calls"

        return {**response, "choices": [new_choice]}

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        resp_id = ""
        model_name = ""
        full_text = ""
        sent_pos = 0
        sent_role = False

        think_buf = self.thinking_extractor.stream_buffer_size
        phase = "content" if think_buf == 0 else "detecting"
        thinking_end_pos = 0

        def _chunk(delta: dict, finish_reason=None) -> str:
            ch: dict = {"index": 0, "delta": delta}
            if finish_reason is not None:
                ch["finish_reason"] = finish_reason
            return format_data_only_sse({
                "id": resp_id,
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [ch],
            })

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
                yield _chunk({"role": "assistant"})

            rc = delta.get("reasoning_content", "")
            if rc:
                yield _chunk({"reasoning_content": rc})

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
                            yield _chunk({"content": pre})
                        sent_pos = content_start
                        phase = "thinking"
                    else:
                        safe = max(
                            sent_pos, len(full_text) - think_buf,
                        )
                        unsent = full_text[sent_pos:safe]
                        if unsent:
                            yield _chunk({"content": unsent})
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
                            yield _chunk({"reasoning_content": unsent})
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
                            yield _chunk({"reasoning_content": unsent})
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
                        yield _chunk({"content": unsent})
                        sent_pos = safe

            if finish_reason:
                if phase == "thinking":
                    unsent = full_text[sent_pos:]
                    if unsent:
                        yield _chunk({"reasoning_content": unsent})
                    yield _chunk({}, finish_reason)
                    yield "data: [DONE]\n\n"
                    return

                if phase == "detecting":
                    reasoning, remaining = (
                        self.thinking_extractor.extract(full_text)
                    )
                    if reasoning:
                        yield _chunk(
                            {"reasoning_content": reasoning},
                        )
                    clean_text, tool_calls = (
                        self.tool_adapter.extract_tool_calls(
                            {"content": remaining},
                        )
                    )
                    remaining_content = clean_text[sent_pos:]
                else:
                    content_portion = full_text[thinking_end_pos:]
                    clean_text, tool_calls = (
                        self.tool_adapter.extract_tool_calls(
                            {"content": content_portion},
                        )
                    )
                    content_sent = max(
                        0, sent_pos - thinking_end_pos,
                    )
                    remaining_content = clean_text[content_sent:]

                if remaining_content:
                    yield _chunk({"content": remaining_content})

                final_delta: dict = {}
                mapped_finish = finish_reason
                if tool_calls:
                    final_delta["tool_calls"] = [{
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    } for idx, tc in enumerate(tool_calls)]
                    mapped_finish = "tool_calls"

                yield _chunk(final_delta, mapped_finish)
                yield "data: [DONE]\n\n"
                return

        yield "data: [DONE]\n\n"
