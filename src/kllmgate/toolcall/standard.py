"""OpenAI 原生格式工具适配器"""

from . import ToolAdapter


class StandardToolAdapter(ToolAdapter):
    """标准 OpenAI 工具格式，字段原样传递"""

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
        content = message.get("content", "")
        raw_calls = message.get("tool_calls", [])
        if not raw_calls:
            return content, []
        calls = []
        for tc in raw_calls:
            func = tc.get("function", {})
            calls.append({
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": func.get("arguments", ""),
            })
        return content, calls

    def detect_stream_tool_boundary(self, text: str) -> int | None:
        return None
