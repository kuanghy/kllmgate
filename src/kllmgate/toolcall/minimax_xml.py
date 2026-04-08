"""MiniMax XML 风格工具适配器"""

from __future__ import annotations

import json
import logging
import re
import uuid

from . import ToolAdapter

logger = logging.getLogger(__name__)

_MINIMAX_TC_RE = re.compile(
    r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL,
)
_INVOKE_RE = re.compile(r"<invoke name=(.*?)</invoke>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter name=(.*?)</parameter>", re.DOTALL)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    return s


def _coerce_param_value(value: str):
    """尝试将参数值解析为合适的 Python 类型"""
    value = value.strip().strip("\n")
    if not value:
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


_TOOL_CALL_TAG = "<minimax:tool_call>"


class MinimaxXmlToolAdapter(ToolAdapter):

    @property
    def stream_buffer_size(self) -> int:
        return len(_TOOL_CALL_TAG)

    def convert_tool_definitions(
        self, tools: list[dict],
    ) -> tuple[str | None, list[dict] | None]:
        tool_tags = []
        for t in tools:
            func = t.get("function", t)
            tool_def = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }
            tool_tags.append(
                f"<tool>{json.dumps(tool_def, ensure_ascii=False)}</tool>"
            )
        prompt = (
            "# Tools\n"
            "You may call one or more tools to assist "
            "with the user query.\n"
            "Here are the tools available in JSONSchema format:\n\n"
            "<tools>\n"
            f"{''.join(tool_tags)}\n"
            "</tools>\n\n"
            "When making tool calls, use XML format to invoke tools "
            "and pass parameters:\n\n"
            "<minimax:tool_call>\n"
            '<invoke name="tool-name">\n'
            '<parameter name="param-name">param-value</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        return prompt, None

    def make_tool_calls_message(self, calls: list[dict]) -> dict:
        xml_lines = ["<minimax:tool_call>"]
        for tc in calls:
            xml_lines.append(f'<invoke name="{tc["name"]}">')
            try:
                args = (
                    json.loads(tc["arguments"])
                    if tc["arguments"] else {}
                )
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Failed to parse tool arguments for %r: %.200s",
                    tc["name"], tc["arguments"],
                )
                xml_lines.append(
                    f'<parameter name="__raw_arguments">'
                    f'{tc["arguments"]}</parameter>'
                )
                xml_lines.append("</invoke>")
                continue
            for k, v in args.items():
                if not isinstance(v, str):
                    v = json.dumps(v, ensure_ascii=False)
                xml_lines.append(f'<parameter name="{k}">{v}</parameter>')
            xml_lines.append("</invoke>")
        xml_lines.append("</minimax:tool_call>")
        return {"role": "assistant", "content": "\n".join(xml_lines)}

    def make_tool_result_message(
        self, call_id: str, name: str, output: str,
    ) -> dict:
        return {
            "role": "user",
            "content": f"[Tool Result] {name}:\n{output}",
        }

    def extract_tool_calls(
        self, message: dict,
    ) -> tuple[str, list[dict]]:
        raw_content = message.get("content", "")
        if "<minimax:tool_call>" not in raw_content:
            return raw_content, []

        calls = []
        blocks = _MINIMAX_TC_RE.findall(raw_content)
        for block in blocks:
            invokes = _INVOKE_RE.findall(block)
            if not invokes:
                logger.warning(
                    "tool_call block has no <invoke>: %.200s", block,
                )
                continue
            for invoke_body in invokes:
                name_match = re.search(r"^([^>]+)", invoke_body)
                if not name_match:
                    logger.warning(
                        "Failed to parse invoke name: %.200s",
                        invoke_body,
                    )
                    continue
                func_name = _strip_quotes(name_match.group(1))

                params = {}
                for param_body in _PARAM_RE.findall(invoke_body):
                    kv = re.search(r"^([^>]+)>(.*)", param_body, re.DOTALL)
                    if kv:
                        k = _strip_quotes(kv.group(1))
                        v = _coerce_param_value(kv.group(2))
                        params[k] = v
                    else:
                        logger.warning(
                            "Failed to parse parameter in %r: %.200s",
                            func_name, param_body,
                        )

                calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "name": func_name,
                    "arguments": json.dumps(
                        params, ensure_ascii=False,
                    ),
                })

        if blocks and not calls:
            logger.warning(
                "Detected <minimax:tool_call> tag but extracted "
                "0 calls: %.300s", raw_content,
            )

        clean_text = _MINIMAX_TC_RE.sub("", raw_content).strip()
        return clean_text, calls

    def detect_stream_tool_boundary(self, text: str) -> int | None:
        pos = text.find("<minimax:tool_call>")
        return pos if pos >= 0 else None
