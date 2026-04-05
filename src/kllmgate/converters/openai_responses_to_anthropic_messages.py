"""OpenAI Responses API → Anthropic Messages 转换器

请求转换复用 responses→chat + chat→anthropic 的组合逻辑。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from . import Converter
from .openai_responses_to_openai_chat import (
    OpenaiResponsesToOpenaiChatConverter,
)
from .openai_chat_to_anthropic_messages import (
    OpenaiChatToAnthropicMessagesConverter,
)
from ._helpers import (
    now_ts,
    make_resp_id,
    make_msg_id,
    anthropic_usage_to_chat,
    chat_usage_to_responses,
    ANTHROPIC_STOP_TO_OPENAI,
    FINISH_REASON_TO_STATUS,
)
from ..sse import SseEvent, format_named_sse


class OpenaiResponsesToAnthropicMessagesConverter(Converter):

    def convert_request(self, body: dict, model: str) -> dict:
        r2c = OpenaiResponsesToOpenaiChatConverter(self.tool_adapter)
        chat_body = r2c.convert_request(body, model)

        c2a = OpenaiChatToAnthropicMessagesConverter(self.tool_adapter)
        return c2a.convert_request(
            {**chat_body, "model": f"_/{model}"}, model,
        )

    def convert_response(self, response: dict) -> dict:
        c2a = OpenaiChatToAnthropicMessagesConverter(self.tool_adapter)
        chat_resp = c2a.convert_response(response)

        content = chat_resp.get("content", [])
        stop_reason = chat_resp.get("stop_reason", "end_turn")
        finish_reason = ANTHROPIC_STOP_TO_OPENAI.get(
            stop_reason, "stop",
        )
        status = FINISH_REASON_TO_STATUS.get(
            finish_reason, "completed",
        )

        output = []
        text_parts = []
        tool_calls = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(block)

        if text_parts:
            output.append({
                "type": "message",
                "id": make_msg_id(),
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "\n".join(text_parts),
                }],
            })

        for tc in tool_calls:
            output.append({
                "type": "function_call",
                "id": f"fc_{tc.get('id', '')}",
                "call_id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "arguments": json.dumps(
                    tc.get("input", {}), ensure_ascii=False,
                ),
                "status": "completed",
            })

        if not output:
            output.append({
                "type": "message",
                "id": make_msg_id(),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": ""}],
            })

        usage = chat_resp.get("usage", {})
        return {
            "id": response.get("id", make_resp_id()),
            "object": "response",
            "created_at": now_ts(),
            "status": status,
            "model": response.get("model", ""),
            "output": output,
            "usage": chat_usage_to_responses(
                anthropic_usage_to_chat(usage),
            ),
        }

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        resp_id = make_resp_id()
        msg_id = make_msg_id()
        model_name = ""
        seq = 0

        def sse(event: str, data: dict) -> str:
            nonlocal seq
            data["sequence_number"] = seq
            seq += 1
            return format_named_sse(event, data)

        response_obj = {
            "id": resp_id,
            "object": "response",
            "created_at": now_ts(),
            "status": "in_progress",
            "model": "",
            "output": [],
        }

        yield sse("response.created", {
            "type": "response.created",
            "response": response_obj,
        })
        yield sse("response.in_progress", {
            "type": "response.in_progress",
            "response": response_obj,
        })
        yield sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": msg_id,
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        })
        yield sse("response.content_part.added", {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        })

        full_text = ""
        usage: dict = {}

        async for event in upstream_events:
            if not event.event:
                continue
            try:
                data = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            if event.event == "message_start":
                msg = data.get("message", {})
                model_name = msg.get("model", "")
                response_obj["model"] = model_name
                if isinstance(msg.get("usage"), dict):
                    usage.update(msg["usage"])

            elif event.event == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    full_text += text
                    yield sse("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text,
                    })

            elif event.event == "message_delta":
                if isinstance(data.get("usage"), dict):
                    usage.update(data["usage"])

            elif event.event == "message_stop":
                yield sse("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": full_text,
                })
                yield sse("response.content_part.done", {
                    "type": "response.content_part.done",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": full_text},
                })
                completed_msg = {
                    "type": "message",
                    "id": msg_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text", "text": full_text,
                    }],
                }
                yield sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": completed_msg,
                })
                resp_usage = chat_usage_to_responses(
                    anthropic_usage_to_chat(usage),
                )
                response_obj.update({
                    "status": "completed",
                    "output": [completed_msg],
                    "usage": resp_usage,
                })
                yield sse("response.completed", {
                    "type": "response.completed",
                    "response": response_obj,
                })
