"""Qwen 文本标签格式工具适配器

Qwen 系列模型的工具调用以 <tool_call> XML 标签包裹 JSON 的方式输出。
当模型通过本地部署框架提供服务且未将此格式转为结构化 tool_calls 时使用。

模型输出格式示例：
    <tool_call>
    {"name": "get_weather", "arguments": {"location": "Hangzhou"}}
    </tool_call>
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from . import ToolAdapter

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL,
)

_TOOL_CALL_TAG = "<tool_call>"


def _parse_tool_call_json(data: dict) -> tuple[str, str]:
    """从 JSON 对象中提取函数名和参数

    支持两种格式：
    - 标准：{"name": "func", "arguments": {"k": "v"}}
    - 扁平：{"function": "func", "k": "v", ...}
    """
    func_name = data.get("name") or data.get("function", "")
    if "arguments" in data:
        args = data["arguments"]
        if isinstance(args, str):
            return func_name, args
        return func_name, json.dumps(args, ensure_ascii=False)
    args = {
        k: v for k, v in data.items()
        if k not in ("name", "function")
    }
    return func_name, json.dumps(args, ensure_ascii=False)


class QwenToolAdapter(ToolAdapter):
    """解析 Qwen <tool_call> 标签格式的工具调用

    请求侧以标准 OpenAI 格式透传 tools（部署框架通过 chat_template 注入），
    响应侧从文本内容中解析 <tool_call> 标签并转换为结构化 tool_calls。
    """

    @property
    def stream_buffer_size(self) -> int:
        return len(_TOOL_CALL_TAG)

    def convert_tool_definitions(
        self, tools: list[dict],
    ) -> tuple[str | None, list[dict] | None]:
        return None, tools

    def make_tool_calls_message(self, calls: list[dict]) -> dict:
        tool_calls = []
        for call in calls:
            tool_calls.append({
                "id": call["call_id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            })
        return {"role": "assistant", "tool_calls": tool_calls}

    def make_tool_result_message(
        self, call_id: str, name: str, output: str,
    ) -> dict:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": output,
        }

    def extract_tool_calls(
        self, message: dict,
    ) -> tuple[str, list[dict]]:
        raw_content = message.get("content", "")
        if not raw_content or _TOOL_CALL_TAG not in raw_content:
            return raw_content or "", []

        calls = []
        for match in _TOOL_CALL_RE.finditer(raw_content):
            json_str = match.group(1).strip()
            try:
                data = json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Failed to parse tool_call JSON: %.200s", json_str,
                )
                continue
            func_name, args_str = _parse_tool_call_json(data)
            if not func_name:
                logger.warning(
                    "tool_call JSON missing function name: %.200s",
                    json_str,
                )
                continue
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "name": func_name,
                "arguments": args_str,
            })

        if not calls:
            logger.warning(
                "Detected <tool_call> tag but extracted "
                "0 calls: %.300s", raw_content,
            )
            return raw_content, []

        clean_text = _TOOL_CALL_RE.sub("", raw_content).strip()
        return clean_text, calls

    def detect_stream_tool_boundary(self, text: str) -> int | None:
        pos = text.find(_TOOL_CALL_TAG)
        return pos if pos >= 0 else None
