"""SSE 解析与格式化工具函数"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class SseEvent:
    event: str | None
    data: str
    raw_lines: list[str]


def format_sse(data: str, event: str | None = None) -> str:
    """格式化单条 SSE 消息

    多行 data 会拆分为多条 data: 行。
    """
    lines = []
    if event is not None:
        lines.append(f"event: {event}")
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def format_data_only_sse(data: dict) -> str:
    """格式化仅含 data 行的 SSE 消息（OpenAI Chat 格式）"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_named_sse(event: str, data: dict) -> str:
    """格式化含 event + data 行的 SSE 消息"""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


def parse_sse_line(line: str) -> tuple[str | None, str | None]:
    """解析单行 SSE 内容

    Returns:
        (field_name, field_value) 或 (None, None)
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None, None

    if ":" in stripped:
        field, _, value = stripped.partition(":")
        value = value.lstrip(" ")
        return field, value

    return stripped, ""


async def parse_sse_events(
    lines: AsyncIterator[str],
) -> AsyncIterator[SseEvent]:
    """将原始 SSE 行流组装为事件帧"""
    current_event: str | None = None
    data_parts: list[str] = []
    raw_lines: list[str] = []

    async for line in lines:
        field, value = parse_sse_line(line)

        if field is None and value is None:
            if not line.strip():
                if data_parts:
                    yield SseEvent(
                        event=current_event,
                        data="\n".join(data_parts),
                        raw_lines=raw_lines,
                    )
                    current_event = None
                    data_parts = []
                    raw_lines = []
            else:
                raw_lines.append(line)
            continue

        raw_lines.append(line)

        if field == "event":
            current_event = value
        elif field == "data":
            data_parts.append(value)

    if data_parts:
        yield SseEvent(
            event=current_event,
            data="\n".join(data_parts),
            raw_lines=raw_lines,
        )
