"""Anthropic Messages → OpenAI Responses API 转换器

请求转换复用 anthropic→chat + chat→responses 的组合逻辑。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from . import Converter
from .anthropic_messages_to_openai_chat import (
    AnthropicMessagesToOpenaiChatConverter,
)
from .openai_chat_to_openai_responses import (
    OpenaiChatToOpenaiResponsesConverter,
)
from ._helpers import (
    now_ts,
    make_resp_id,
    make_msg_id,
    OPENAI_FINISH_TO_ANTHROPIC,
    responses_usage_to_chat,
    chat_usage_to_anthropic,
)
from ..sse import SseEvent, format_named_sse


class AnthropicMessagesToOpenaiResponsesConverter(Converter):

    def convert_request(self, body: dict, model: str) -> dict:
        a2c = AnthropicMessagesToOpenaiChatConverter(self.tool_adapter)
        chat_body = a2c.convert_request(body, model)

        c2r = OpenaiChatToOpenaiResponsesConverter(self.tool_adapter)
        return c2r.convert_request(
            {**chat_body, "model": f"_/{model}"}, model,
        )

    def convert_response(self, response: dict) -> dict:
        c2r = OpenaiChatToOpenaiResponsesConverter(self.tool_adapter)
        chat_resp = c2r.convert_response(response)

        message = chat_resp.get("choices", [{}])[0].get("message", {})
        content_text = message.get("content", "") or ""
        finish_reason = (
            chat_resp.get("choices", [{}])[0]
            .get("finish_reason", "stop")
        )
        tool_calls_raw = message.get("tool_calls", [])

        content_blocks = []
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

        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})

        stop_reason = OPENAI_FINISH_TO_ANTHROPIC.get(
            finish_reason, "end_turn",
        )
        usage = chat_resp.get("usage", {})

        return {
            "id": response.get("id", make_resp_id()),
            "type": "message",
            "role": "assistant",
            "model": response.get("model", ""),
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": chat_usage_to_anthropic(usage),
        }

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        resp_id = ""
        model_name = ""
        content_index = 0
        full_text = ""
        usage: dict = {}

        async for event in upstream_events:
            if not event.event:
                continue
            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            if event.event == "response.created":
                resp_obj = data.get("response", {})
                resp_id = resp_obj.get("id", "")
                model_name = resp_obj.get("model", "")
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
                yield format_named_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": content_index,
                    "content_block": {"type": "text", "text": ""},
                })

            elif event.event == "response.output_text.delta":
                text = data.get("delta", "")
                full_text += text
                yield format_named_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": content_index,
                    "delta": {
                        "type": "text_delta",
                        "text": text,
                    },
                })

            elif event.event == "response.completed":
                resp_obj = data.get("response", {})
                if isinstance(resp_obj.get("usage"), dict):
                    usage.update(resp_obj["usage"])
                anthropic_usage = chat_usage_to_anthropic(
                    responses_usage_to_chat(usage),
                )
                yield format_named_sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": content_index,
                })
                yield format_named_sse("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "output_tokens": anthropic_usage.get(
                            "output_tokens", 0,
                        ),
                    },
                })
                yield format_named_sse("message_stop", {
                    "type": "message_stop",
                })
