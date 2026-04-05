"""Converter 内部共享的辅助函数"""

from __future__ import annotations

import json
import time
import uuid


def make_resp_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def make_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def now_ts() -> int:
    return int(time.time())


def normalize_text_content(content) -> str:
    """将 content 规范化为纯文本字符串

    支持 str（直接返回）和 list（提取 text 块并拼接）两种输入。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in ("text", "input_text", "output_text"):
                    parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def merge_chat_content(prev_content, curr_content):
    """合并两个 Chat completions 的 content

    处理 str+str, list+str, str+list, list+list 各种情况
    """
    if isinstance(prev_content, str) and isinstance(curr_content, str):
        return f"{prev_content}\n\n{curr_content}"

    prev_list = (
        prev_content if isinstance(prev_content, list)
        else [{"type": "text", "text": prev_content}]
    )
    curr_list = (
        curr_content if isinstance(curr_content, list)
        else [{"type": "text", "text": curr_content}]
    )
    return prev_list + curr_list


def _responses_part_to_chat(part: dict) -> dict:
    """将 Responses API 内容块映射为 Chat Completions content part"""
    part_type = part.get("type", "")
    if part_type in ("text", "input_text", "output_text"):
        return {"type": "text", "text": part.get("text", "")}
    if part_type == "input_image":
        return {
            "type": "image_url",
            "image_url": part.get("image_url", {}),
        }
    if part_type == "input_audio":
        return {
            "type": "input_audio",
            "input_audio": part.get("input_audio", {}),
        }
    if part_type == "input_file":
        return {
            "type": "file",
            "file": part.get("file", part.get("input_file", {})),
        }
    return part


def convert_content_to_chat_parts(content: list) -> list[dict]:
    """将 Responses API content list 映射为 Chat content parts 格式"""
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            parts.append(_responses_part_to_chat(item))
    return parts


FINISH_REASON_TO_STATUS = {
    "stop": "completed",
    "tool_calls": "completed",
    "length": "incomplete",
    "content_filter": "incomplete",
}

STATUS_TO_FINISH_REASON = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "stop",
}

ANTHROPIC_STOP_TO_OPENAI = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
}

OPENAI_FINISH_TO_ANTHROPIC = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def chat_usage_to_responses(usage: dict) -> dict:
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": usage.get(
            "input_tokens", usage.get("prompt_tokens", 0),
        ),
        "output_tokens": usage.get(
            "output_tokens", usage.get("completion_tokens", 0),
        ),
        "total_tokens": usage.get("total_tokens", 0),
    }


def responses_usage_to_chat(usage: dict) -> dict:
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": usage.get(
            "prompt_tokens", usage.get("input_tokens", 0),
        ),
        "completion_tokens": usage.get(
            "completion_tokens", usage.get("output_tokens", 0),
        ),
        "total_tokens": usage.get("total_tokens", 0),
    }


def chat_usage_to_anthropic(usage: dict) -> dict:
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


def anthropic_usage_to_chat(usage: dict) -> dict:
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    it = usage.get("input_tokens", 0)
    ot = usage.get("output_tokens", 0)
    return {
        "prompt_tokens": it,
        "completion_tokens": ot,
        "total_tokens": it + ot,
    }
