"""ToolAdapter 基类"""

from abc import ABC, abstractmethod


class ToolAdapter(ABC):
    """工具调用适配器基类"""

    @abstractmethod
    def convert_tool_definitions(
        self, tools: list[dict],
    ) -> tuple[str | None, list[dict] | None]:
        """转换工具定义为上游格式

        Returns:
            (system_prompt_addition, tools_field)
        """

    @abstractmethod
    def make_tool_calls_message(
        self, calls: list[dict],
    ) -> dict:
        """将历史工具调用格式化为上游可接受的 assistant 消息"""

    @abstractmethod
    def make_tool_result_message(
        self, call_id: str, name: str, output: str,
    ) -> dict:
        """将工具执行结果格式化为上游可接受的消息"""

    @abstractmethod
    def extract_tool_calls(
        self, message: dict,
    ) -> tuple[str, list[dict]]:
        """从上游响应消息中提取工具调用

        Returns:
            (cleaned_content, normalized_calls)
        """

    @property
    def stream_buffer_size(self) -> int:
        """流式工具调用检测需要的回看缓冲字节数，0 表示无需缓冲"""
        return 0

    @abstractmethod
    def detect_stream_tool_boundary(
        self, text: str,
    ) -> int | None:
        """检测流式文本中工具调用边界的起始位置"""
