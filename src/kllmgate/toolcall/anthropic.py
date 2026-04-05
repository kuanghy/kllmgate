"""Anthropic 格式工具适配器"""

from __future__ import annotations

import json

from . import ToolAdapter


class AnthropicToolAdapter(ToolAdapter):

    def convert_tool_definitions(
        self, tools: list[dict],
    ) -> tuple[str | None, list[dict] | None]:
        converted = []
        for t in tools:
            func = t.get("function", t)
            converted.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            })
        return None, converted

    def make_tool_calls_message(self, calls: list[dict]) -> dict:
        content_blocks = []
        for call in calls:
            args = call["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            content_blocks.append({
                "type": "tool_use",
                "id": call["call_id"],
                "name": call["name"],
                "input": args,
            })
        return {"role": "assistant", "content": content_blocks}

    def make_tool_result_message(
        self, call_id: str, name: str, output: str,
    ) -> dict:
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": output,
            }],
        }

    def extract_tool_calls(
        self, message: dict,
    ) -> tuple[str, list[dict]]:
        raw_content = message.get("content", "")
        if isinstance(raw_content, str):
            return raw_content, []

        text_parts = []
        calls = []
        for block in raw_content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                input_data = block.get("input", {})
                calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(
                        input_data, ensure_ascii=False,
                    ),
                })

        return "\n".join(text_parts), calls

    def detect_stream_tool_boundary(self, text: str) -> int | None:
        return None
