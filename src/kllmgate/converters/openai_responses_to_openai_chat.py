"""OpenAI Responses API → OpenAI Chat Completions 转换器"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from . import Converter
from ._helpers import (
    convert_content_list,
    make_resp_id,
    make_msg_id,
    now_ts,
    chat_usage_to_responses,
    FINISH_REASON_TO_STATUS,
)
from ..sse import SseEvent, format_named_sse

logger = logging.getLogger(__name__)


class OpenaiResponsesToOpenaiChatConverter(Converter):

    @staticmethod
    def _normalize_tools(tools: list[dict]) -> list[dict]:
        """将 Responses API 工具定义转为 Chat Completions 格式

        Responses: {"type": "function", "name": "...", "parameters": {...}}
        Chat:      {"type": "function", "function": {"name": "...", "parameters": {...}}}

        过滤掉非 function 类型（如 web_search）
        """
        result = []
        for t in tools:
            if t.get("type") != "function":
                continue
            if "function" in t:
                result.append(t)
            else:
                func = {
                    k: v for k, v in t.items()
                    if k not in ("type", "strict")
                }
                entry: dict = {"type": "function", "function": func}
                if "strict" in t:
                    entry["function"]["strict"] = t["strict"]
                result.append(entry)
        return result

    def convert_request(self, body: dict, model: str) -> dict:
        system_contents: list[str] = []
        if body.get("instructions"):
            system_contents.append(body["instructions"])

        tools = body.get("tools")
        if tools:
            tools = self._normalize_tools(tools)
            prompt_add, tools_field = (
                self.tool_adapter.convert_tool_definitions(tools)
            )
            if prompt_add:
                system_contents.append(prompt_add)
                tools = None
            else:
                tools = tools_field or None

        raw_messages: list[dict] = []
        pending_calls: list[dict] = []
        pending_results: list[dict] = []
        call_id_to_name: dict[str, str] = {}

        raw_input = body.get("input", [])
        if isinstance(raw_input, str):
            raw_input = [raw_input]

        for item in raw_input:
            if isinstance(item, str):
                self._flush(
                    raw_messages, pending_calls, pending_results,
                )
                raw_messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")

            if item_type == "function_call":
                call_id = item.get("call_id", item.get("id", ""))
                func_name = item.get("name", "")
                call_id_to_name[call_id] = func_name
                pending_calls.append({
                    "name": func_name,
                    "arguments": item.get("arguments", ""),
                    "call_id": call_id,
                })
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id", "")
                pending_results.append({
                    "call_id": call_id,
                    "name": call_id_to_name.get(call_id, "unknown"),
                    "output": item.get("output", ""),
                })
                continue

            self._flush(raw_messages, pending_calls, pending_results)

            role = item.get("role", "user")
            if role == "model":
                role = "assistant"
            elif role == "developer":
                role = "system"

            content = item.get("content", "")
            if isinstance(content, list):
                content = convert_content_list(content)

            if role == "system":
                system_contents.append(content)
            else:
                raw_messages.append({"role": role, "content": content})

        self._flush(raw_messages, pending_calls, pending_results)

        messages: list[dict] = []
        if system_contents:
            messages.append({
                "role": "system",
                "content": "\n\n".join(system_contents),
            })

        for msg in raw_messages:
            if (
                messages
                and messages[-1]["role"] == msg["role"]
                and msg["role"] != "assistant"
            ):
                messages[-1]["content"] += "\n\n" + msg["content"]
            else:
                messages.append(msg)

        result: dict = {"model": model, "messages": messages}

        if tools:
            result["tools"] = tools
        if body.get("stream"):
            result["stream"] = True
            result["stream_options"] = {"include_usage": True}

        return result

    def _flush(
        self,
        raw_messages: list[dict],
        pending_calls: list[dict],
        pending_results: list[dict],
    ):
        if pending_calls:
            msg = self.tool_adapter.make_tool_calls_message(pending_calls)
            raw_messages.append(msg)
            pending_calls.clear()

        for result in pending_results:
            msg = self.tool_adapter.make_tool_result_message(
                result["call_id"], result["name"], result["output"],
            )
            raw_messages.append(msg)
        pending_results.clear()

    def convert_response(self, response: dict) -> dict:
        choices = response.get("choices", [])
        if not choices:
            return self._empty_response(response)

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        content_text, tool_calls = (
            self.tool_adapter.extract_tool_calls(message)
        )

        output = []
        if content_text:
            output.append({
                "type": "message",
                "id": make_msg_id(),
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": content_text},
                ],
            })

        for tc in tool_calls:
            output.append({
                "type": "function_call",
                "id": f"fc_{tc['id']}",
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": tc["arguments"],
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

        usage = chat_usage_to_responses(response.get("usage", {}))
        status = FINISH_REASON_TO_STATUS.get(finish_reason, "completed")

        return {
            "id": response.get("id", make_resp_id()),
            "object": "response",
            "created_at": response.get("created", now_ts()),
            "status": status,
            "model": response.get("model", ""),
            "output": output,
            "usage": usage,
        }

    def _empty_response(self, response: dict) -> dict:
        return {
            "id": response.get("id", make_resp_id()),
            "object": "response",
            "created_at": response.get("created", now_ts()),
            "status": "completed",
            "model": response.get("model", ""),
            "output": [{
                "type": "message",
                "id": make_msg_id(),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": ""}],
            }],
            "usage": chat_usage_to_responses(response.get("usage", {})),
        }

    async def convert_stream(
        self,
        upstream_events: AsyncIterator[SseEvent],
    ) -> AsyncIterator[str]:
        resp_id = make_resp_id()
        msg_id = make_msg_id()
        model_name = ""
        created_ts = now_ts()
        full_text = ""
        sent_pos = 0
        tc_detected = False
        streamed_tool_calls: dict[int, dict] = {}
        seq = 0
        usage = {}

        def sse(event: str, data: dict) -> str:
            nonlocal seq
            data["sequence_number"] = seq
            seq += 1
            return format_named_sse(event, data)

        response_obj = {
            "id": resp_id,
            "object": "response",
            "created_at": created_ts,
            "status": "in_progress",
            "model": model_name,
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

        msg_item = {
            "type": "message",
            "id": msg_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        yield sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": msg_item,
        })
        yield sse("response.content_part.added", {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        })

        async for event in upstream_events:
            if event.data == "[DONE]":
                break
            try:
                chunk = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            if not model_name:
                model_name = chunk.get("model", "")
                response_obj["model"] = model_name

            if chunk_usage := chunk.get("usage"):
                usage = chunk_usage

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            for tc_delta in delta.get("tool_calls", []):
                tc_index = tc_delta.get("index", 0)
                existing = streamed_tool_calls.setdefault(tc_index, {
                    "id": "",
                    "name": "",
                    "arguments": "",
                })
                if tc_delta.get("id"):
                    existing["id"] = tc_delta["id"]
                function = tc_delta.get("function", {})
                if function.get("name"):
                    existing["name"] = function["name"]
                if function.get("arguments"):
                    existing["arguments"] += function["arguments"]

            delta_text = delta.get("content", "")
            if not delta_text:
                continue

            full_text += delta_text
            if tc_detected:
                continue

            boundary = self.tool_adapter.detect_stream_tool_boundary(
                full_text,
            )
            if boundary is not None:
                tc_detected = True
                safe_end = boundary
            else:
                tag_len = len("<minimax:tool_call>")
                safe_end = max(
                    sent_pos, len(full_text) - tag_len,
                )

            unsent = full_text[sent_pos:safe_end]
            if unsent:
                yield sse("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": unsent,
                })
                sent_pos = safe_end

        clean_text, tool_calls = (
            self.tool_adapter.extract_tool_calls({"content": full_text})
        )
        if streamed_tool_calls:
            tool_calls.extend(
                {
                    "id": item["id"] or f"call_{index}",
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
                for index, item in sorted(streamed_tool_calls.items())
            )

        remaining = clean_text[sent_pos:]
        if remaining:
            yield sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "delta": remaining,
            })

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
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": clean_text}],
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

        resp_usage = chat_usage_to_responses(usage)
        response_obj.update({
            "status": "completed",
            "output": output_items,
            "usage": resp_usage,
        })
        yield sse("response.completed", {
            "type": "response.completed",
            "response": response_obj,
        })
