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


def convert_content_part(part: dict) -> str:
    """将单个 content part 转为纯文本"""
    part_type = part.get("type", "")

    if part_type in ("text", "input_text", "output_text"):
        return part.get("text", "")
    if part_type == "image_url":
        url = part.get("image_url", {})
        if isinstance(url, str):
            return f"[image: {url}]"
        return (
            f'[image: {url.get("url", "unknown")} '
            f'detail={url.get("detail", "auto")}]'
        )
    if part_type == "image":
        image_url = part.get("image_url") or part.get("url", "")
        file_id = part.get("file_id", "")
        if image_url:
            return f"[image: {image_url}]"
        if file_id:
            return f"[image file: {file_id}]"
        return "[image]"
    if part_type in ("input_audio", "audio"):
        transcript = part.get("transcript", "")
        audio_id = part.get("id", part.get("file_id", ""))
        if transcript:
            return f"[audio transcript: {transcript}]"
        if audio_id:
            return f"[audio: {audio_id}]"
        return "[audio]"
    if part_type == "file":
        file_id = part.get("file_id", part.get("id", ""))
        filename = part.get("filename", "")
        return f"[file: {filename or file_id or 'unknown'}]"
    if part_type == "refusal":
        return part.get("refusal", "[refusal]")

    return str(part)


def convert_content_list(content: list) -> str:
    """将 content 列表转为纯文本"""
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(convert_content_part(item))
    return "\n".join(parts)


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
