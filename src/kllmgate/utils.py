"""通用工具函数"""

from __future__ import annotations


def text_shorten(text: str, width: int = 60, placeholder: str = "...") -> str:
    """截断文本，超过 width 时追加 placeholder"""
    return text[:width] + (text[width:] and placeholder)
