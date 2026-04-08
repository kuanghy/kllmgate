"""DeepSeek + Qwen 组合工具适配器

同时支持 DeepSeek 原生 token 格式和 Qwen <tool_call> 格式的解析。
适用于混合使用两种格式的模型（如 DeepSeek-R1 蒸馏到 Qwen 架构的模型），
优先尝试 DeepSeek 格式，回退到 Qwen 格式。
"""

from __future__ import annotations

from . import ToolAdapter
from .deepseek import DeepseekToolAdapter
from .qwen import QwenToolAdapter


class DeepseekQwenToolAdapter(ToolAdapter):
    """组合 DeepSeek + Qwen 双格式解析"""

    def __init__(self):
        self._deepseek = DeepseekToolAdapter()
        self._qwen = QwenToolAdapter()

    @property
    def stream_buffer_size(self) -> int:
        return max(
            self._deepseek.stream_buffer_size,
            self._qwen.stream_buffer_size,
        )

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
        content, calls = self._deepseek.extract_tool_calls(message)
        if calls:
            return content, calls
        return self._qwen.extract_tool_calls(message)

    def detect_stream_tool_boundary(
        self, text: str,
    ) -> int | None:
        ds = self._deepseek.detect_stream_tool_boundary(text)
        qw = self._qwen.detect_stream_tool_boundary(text)
        if ds is not None and qw is not None:
            return min(ds, qw)
        return ds if ds is not None else qw
