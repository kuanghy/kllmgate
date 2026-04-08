"""思考过程提取器

从模型响应文本中提取结构化的思考/推理内容，
转换为 OpenAI 兼容的 reasoning_content 字段。

支持的标签格式：
- <think>...</think>             常见于 DeepSeek-R1、QwQ 等推理模型
- <lvl_N_xxx>...</lvl_N_xxx>     DeepSeek-R1 蒸馏模型（如 R1-0528-Qwen3-8B）
  xxx 为语义动作后缀：entry、plan、thought、output、refine 等
- <|thinking|>...<|/thinking|>   DeepSeek 原生特殊 token 格式
  渲染变体包括全角竖线（<｜thinking｜>）和带空格（< | thinking | >）
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class ThinkingExtractor(ABC):
    """思考过程提取器基类"""

    @abstractmethod
    def extract(self, text: str) -> tuple[str, str]:
        """从完整文本中提取思考内容

        Returns:
            (reasoning_content, remaining_content)
        """

    @property
    @abstractmethod
    def stream_buffer_size(self) -> int:
        """流式标签检测所需的回看缓冲大小"""

    @abstractmethod
    def find_open_tag(self, text: str) -> tuple[int, int] | None:
        """查找思考开始标签

        Returns:
            (tag_start, content_start) or None
        """

    @abstractmethod
    def find_close_tag(
        self, text: str, start: int = 0,
    ) -> tuple[int, int] | None:
        """查找思考结束标签

        Returns:
            (content_end, tag_end) or None
        """


class NullThinkingExtractor(ThinkingExtractor):
    """无操作提取器，用于不需要思考提取的场景"""

    def extract(self, text: str) -> tuple[str, str]:
        return "", text

    @property
    def stream_buffer_size(self) -> int:
        return 0

    def find_open_tag(self, text: str) -> tuple[int, int] | None:
        return None

    def find_close_tag(
        self, text: str, start: int = 0,
    ) -> tuple[int, int] | None:
        return None


class _RegexThinkingExtractor(ThinkingExtractor):
    """基于正则的通用思考提取器"""

    def __init__(
        self,
        block_re: re.Pattern[str],
        open_re: re.Pattern[str],
        close_re: re.Pattern[str],
        buffer_size: int,
        content_group: int = 1,
    ):
        self._block_re = block_re
        self._open_re = open_re
        self._close_re = close_re
        self._buffer_size = buffer_size
        self._content_group = content_group

    def extract(self, text: str) -> tuple[str, str]:
        parts: list[str] = []
        has_match = False
        for m in self._block_re.finditer(text):
            has_match = True
            content = m.group(self._content_group).strip()
            if content:
                parts.append(self._clean_inner(content))
        if not has_match:
            return "", text
        remaining = self._block_re.sub("", text).strip()
        return "\n\n".join(parts), remaining

    def _clean_inner(self, text: str) -> str:
        """清理提取内容中的内部标签，子类可覆盖"""
        return text

    @property
    def stream_buffer_size(self) -> int:
        return self._buffer_size

    def find_open_tag(self, text: str) -> tuple[int, int] | None:
        m = self._open_re.search(text)
        if m:
            return m.start(), m.end()
        return None

    def find_close_tag(
        self, text: str, start: int = 0,
    ) -> tuple[int, int] | None:
        m = self._close_re.search(text, start)
        if m:
            return m.start(), m.end()
        return None


_THINK_BLOCK_RE = re.compile(
    r"<think>\s*(.*?)\s*</think>", re.DOTALL,
)
_THINK_OPEN_RE = re.compile(r"<think>")
_THINK_CLOSE_RE = re.compile(r"</think>")


class ThinkTagExtractor(_RegexThinkingExtractor):
    """提取 <think>...</think> 标签"""

    def __init__(self):
        super().__init__(
            block_re=_THINK_BLOCK_RE,
            open_re=_THINK_OPEN_RE,
            close_re=_THINK_CLOSE_RE,
            buffer_size=len("</think>"),
        )


_BAR = r"[|\uff5c]"

_THINKING_TOKEN_BLOCK_RE = re.compile(
    rf"<\s*{_BAR}\s*thinking\s*{_BAR}\s*>"
    r"\s*(.*?)\s*"
    rf"<\s*{_BAR}\s*/?\s*thinking\s*{_BAR}\s*>",
    re.DOTALL,
)
_THINKING_TOKEN_OPEN_RE = re.compile(
    rf"<\s*{_BAR}\s*thinking\s*{_BAR}\s*>",
)
_THINKING_TOKEN_CLOSE_RE = re.compile(
    rf"<\s*{_BAR}\s*/?\s*thinking\s*{_BAR}\s*>",
)


class ThinkingTokenExtractor(_RegexThinkingExtractor):
    """提取 <|thinking|>...<|/thinking|> 特殊 token（DeepSeek 原生格式）

    兼容多种渲染变体：
    - 原始 Unicode：<｜thinking｜>...<｜/thinking｜>
    - ASCII：<|thinking|>...<|/thinking|>
    - 带空格：< | thinking | >...< | /thinking | >

    部分推理框架渲染闭合标签时可能省略 /，闭合模式同时匹配两种情况。
    """

    def __init__(self):
        super().__init__(
            block_re=_THINKING_TOKEN_BLOCK_RE,
            open_re=_THINKING_TOKEN_OPEN_RE,
            close_re=_THINKING_TOKEN_CLOSE_RE,
            buffer_size=20,
        )


_LVL0_BLOCK_RE = re.compile(
    r"<lvl_0_(\w+)>\s*(.*?)\s*</lvl_0_\1>", re.DOTALL,
)
_LVL_OPEN_RE = re.compile(r"<lvl_0_entry>")
_LVL_CLOSE_RE = re.compile(r"</lvl_0_entry>")
_LVL_TAG_STRIP_RE = re.compile(r"</?lvl_\d+_\w+>")


class LvlEntryThinkingExtractor(_RegexThinkingExtractor):
    """提取 <lvl_N_xxx> 标签（DeepSeek-R1 蒸馏模型）

    xxx 为语义动作后缀（entry、plan、thought、output、refine 等）。
    使用反向引用匹配配对的 <lvl_0_xxx>...</lvl_0_xxx> 块，
    内部嵌套的所有 <lvl_N_xxx> 标签会被清理为纯文本。

    流式模式下仅检测 lvl_0 层级标签作为思考边界，
    内层标签（lvl_1、lvl_2 等）不会触发提前退出。
    """

    def __init__(self):
        super().__init__(
            block_re=_LVL0_BLOCK_RE,
            open_re=_LVL_OPEN_RE,
            close_re=_LVL_CLOSE_RE,
            buffer_size=len("</lvl_0_thought>"),
            content_group=2,
        )

    def _clean_inner(self, text: str) -> str:
        return _LVL_TAG_STRIP_RE.sub("", text).strip()
