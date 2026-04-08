"""DeepSeek 原生 token 格式工具适配器

DeepSeek 模型通过本地部署框架（vLLM、Ollama 等）提供服务时，工具调用
可能以原生特殊 token 格式输出在文本内容中，而非结构化的 tool_calls 字段。
此适配器解析这些原生 token 并转换为标准 OpenAI tool_calls 格式。

模型原始格式示例（标准带代码围栏）：
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"location": "Hangzhou"}
    ```
    <｜tool▁call▁end｜><｜tool▁calls▁end｜>

变体（双下划线 + 无代码围栏，部分蒸馏模型）：
    < | tool__calls__begin | >< | tool__call__begin | >function
    < | tool_sep | >get_weather< | tool_sep | >
    {"location": "Hangzhou"}
    < | tool__call__end | >< | tool__calls__end | >
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from . import ToolAdapter

logger = logging.getLogger(__name__)


def _tok_re(name: str) -> str:
    """构建匹配 DeepSeek 特殊 token 的灵活正则

    处理多种渲染变体：原始 token（<｜tool▁calls▁begin｜>）、
    ASCII 变体（<|tool_calls_begin|>）、带空格变体（< | tool_calls_begin | >）、
    双下划线变体（< | tool__calls__begin | >）
    """
    flex = name.replace("_", r"[_\u2581\s]+")
    return rf"<\s*[|\uff5c]\s*{flex}\s*[|\uff5c]\s*>"


_CALLS_BEGIN = _tok_re("tool_calls_begin")
_CALLS_END = _tok_re("tool_calls_end")
_CALL_BEGIN = _tok_re("tool_call_begin")
_CALL_END = _tok_re("tool_call_end")
_TOOL_SEP = _tok_re("tool_sep")

_TOOL_CALLS_BLOCK_RE = re.compile(
    rf"{_CALLS_BEGIN}(.*?){_CALLS_END}", re.DOTALL,
)

_SINGLE_CALL_RE = re.compile(
    rf"{_CALL_BEGIN}\s*\w*\s*{_TOOL_SEP}\s*([^\s<]+)\s*"
    rf"(?:```(?:json)?\s*\n?|{_TOOL_SEP}\s*)"
    rf"(.*?)"
    rf"(?:\n?\s*```)?\s*"
    rf"{_CALL_END}",
    re.DOTALL,
)

_BOUNDARY_RE = re.compile(rf"{_CALLS_BEGIN}")

_QUICK_CHECK_RE = re.compile(r"tool[_\u2581\s]+calls[_\u2581\s]+begin")


class DeepseekToolAdapter(ToolAdapter):
    """解析 DeepSeek 原生 token 格式的工具调用

    请求侧以标准 OpenAI 格式透传 tools（部署框架通过 chat_template 注入），
    响应侧从文本内容中解析原生 token 并转换为结构化 tool_calls。
    """

    @property
    def stream_buffer_size(self) -> int:
        return 30

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
        if not raw_content or not _QUICK_CHECK_RE.search(raw_content):
            return raw_content or "", []

        calls = []
        for block_match in _TOOL_CALLS_BLOCK_RE.finditer(raw_content):
            block = block_match.group(1)
            for m in _SINGLE_CALL_RE.finditer(block):
                func_name = m.group(1).strip()
                args_str = m.group(2).strip()
                try:
                    json.loads(args_str)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "Failed to parse tool arguments for %r: %.200s",
                        func_name, args_str,
                    )
                    args_str = json.dumps(
                        {"__raw": args_str}, ensure_ascii=False,
                    )
                calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "name": func_name,
                    "arguments": args_str,
                })

        if not calls:
            logger.warning(
                "Detected tool_calls_begin token but extracted "
                "0 calls: %.300s", raw_content,
            )
            return raw_content, []

        clean_text = _TOOL_CALLS_BLOCK_RE.sub("", raw_content).strip()
        return clean_text, calls

    def detect_stream_tool_boundary(self, text: str) -> int | None:
        m = _BOUNDARY_RE.search(text)
        return m.start() if m else None
