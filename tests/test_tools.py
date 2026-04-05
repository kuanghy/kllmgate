"""工具适配器的单元测试"""

import json

import pytest

from kllmgate.toolcall.standard import StandardToolAdapter
from kllmgate.toolcall.minimax_xml import MinimaxXmlToolAdapter
from kllmgate.toolcall.anthropic import AnthropicToolAdapter


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    },
]

SAMPLE_CALLS = [
    {"name": "get_weather", "arguments": '{"location": "Beijing"}', "call_id": "call_1"},
]


class TestStandardToolAdapter:

    def setup_method(self):
        self.adapter = StandardToolAdapter()

    def test_convert_tool_definitions_passthrough(self):
        prompt, tools = self.adapter.convert_tool_definitions(SAMPLE_TOOLS)
        assert prompt is None
        assert tools == SAMPLE_TOOLS

    def test_make_tool_calls_message(self):
        msg = self.adapter.make_tool_calls_message(SAMPLE_CALLS)
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"location": "Beijing"}'

    def test_make_tool_result_message(self):
        msg = self.adapter.make_tool_result_message(
            "call_1", "get_weather", '{"temp": 25}',
        )
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"
        assert msg["content"] == '{"temp": 25}'

    def test_extract_tool_calls_with_tool_calls(self):
        message = {
            "content": "Let me check the weather.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Beijing"}',
                    },
                },
            ],
        }
        content, calls = self.adapter.extract_tool_calls(message)
        assert content == "Let me check the weather."
        assert len(calls) == 1
        assert calls[0]["id"] == "call_1"
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"] == '{"location": "Beijing"}'

    def test_extract_tool_calls_no_tools(self):
        message = {"content": "Hello!"}
        content, calls = self.adapter.extract_tool_calls(message)
        assert content == "Hello!"
        assert calls == []

    def test_detect_stream_tool_boundary(self):
        assert self.adapter.detect_stream_tool_boundary("any text") is None
        assert self.adapter.detect_stream_tool_boundary("") is None

    def test_make_tool_calls_message_multiple(self):
        calls = [
            {"name": "func_a", "arguments": "{}", "call_id": "c1"},
            {"name": "func_b", "arguments": '{"x": 1}', "call_id": "c2"},
        ]
        msg = self.adapter.make_tool_calls_message(calls)
        assert len(msg["tool_calls"]) == 2


class TestMinimaxXmlToolAdapter:

    def setup_method(self):
        self.adapter = MinimaxXmlToolAdapter()

    def test_convert_tool_definitions_returns_prompt(self):
        prompt, tools = self.adapter.convert_tool_definitions(SAMPLE_TOOLS)
        assert tools is None
        assert prompt is not None
        assert "get_weather" in prompt
        assert "<tool>" in prompt
        assert "minimax:tool_call" in prompt

    def test_make_tool_calls_message(self):
        msg = self.adapter.make_tool_calls_message(SAMPLE_CALLS)
        assert msg["role"] == "assistant"
        assert "<minimax:tool_call>" in msg["content"]
        assert "get_weather" in msg["content"]
        assert "Beijing" in msg["content"]

    def test_make_tool_result_message(self):
        msg = self.adapter.make_tool_result_message(
            "call_1", "get_weather", '{"temp": 25}',
        )
        assert msg["role"] == "user"
        assert "[Tool Result]" in msg["content"]
        assert "get_weather" in msg["content"]

    def test_extract_tool_calls_basic(self):
        text = (
            'Here is some text\n'
            '<minimax:tool_call>\n'
            '<invoke name="get_weather">\n'
            '<parameter name="location">Beijing</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        message = {"content": text}
        content, calls = self.adapter.extract_tool_calls(message)
        assert "Here is some text" in content
        assert "<minimax:tool_call>" not in content
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        args = json.loads(calls[0]["arguments"])
        assert args["location"] == "Beijing"

    def test_extract_tool_calls_no_tools(self):
        message = {"content": "Just a normal message"}
        content, calls = self.adapter.extract_tool_calls(message)
        assert content == "Just a normal message"
        assert calls == []

    def test_extract_tool_calls_multiple_params(self):
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="search">\n'
            '<parameter name="query">python</parameter>\n'
            '<parameter name="limit">10</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        message = {"content": text}
        content, calls = self.adapter.extract_tool_calls(message)
        assert len(calls) == 1
        args = json.loads(calls[0]["arguments"])
        assert args["query"] == "python"
        assert args["limit"] == 10

    def test_extract_tool_calls_multiple_invocations(self):
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="func_a">\n'
            '<parameter name="x">1</parameter>\n'
            '</invoke>\n'
            '<invoke name="func_b">\n'
            '<parameter name="y">2</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        message = {"content": text}
        _, calls = self.adapter.extract_tool_calls(message)
        assert len(calls) == 2
        assert calls[0]["name"] == "func_a"
        assert calls[1]["name"] == "func_b"

    def test_detect_stream_tool_boundary(self):
        assert self.adapter.detect_stream_tool_boundary(
            "Hello <minimax:tool_call>"
        ) == 6
        assert self.adapter.detect_stream_tool_boundary(
            "No tools here"
        ) is None

    def test_extract_tool_calls_json_array_param(self):
        text = (
            '<minimax:tool_call>\n'
            '<invoke name="multi">\n'
            '<parameter name="items">[1, 2, 3]</parameter>\n'
            '</invoke>\n'
            '</minimax:tool_call>'
        )
        message = {"content": text}
        _, calls = self.adapter.extract_tool_calls(message)
        args = json.loads(calls[0]["arguments"])
        assert args["items"] == [1, 2, 3]

    def test_make_tool_calls_message_invalid_json_fallback(self):
        calls = [{"name": "func", "arguments": "not json", "call_id": "c1"}]
        msg = self.adapter.make_tool_calls_message(calls)
        assert "__raw_arguments" in msg["content"]


class TestAnthropicToolAdapter:

    def setup_method(self):
        self.adapter = AnthropicToolAdapter()

    def test_convert_tool_definitions(self):
        prompt, tools = self.adapter.convert_tool_definitions(SAMPLE_TOOLS)
        assert prompt is None
        assert len(tools) == 1
        tool = tools[0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather for a location"
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "type" not in tool or tool.get("type") != "function"

    def test_convert_tool_definitions_without_function_wrapper(self):
        """测试工具定义不带 function 包装的情况"""
        tools = [{
            "name": "search",
            "description": "Search something",
            "parameters": {"type": "object", "properties": {}},
        }]
        _, converted = self.adapter.convert_tool_definitions(tools)
        assert converted[0]["name"] == "search"
        assert "input_schema" in converted[0]

    def test_make_tool_calls_message(self):
        msg = self.adapter.make_tool_calls_message(SAMPLE_CALLS)
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], list)
        block = msg["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_1"
        assert block["name"] == "get_weather"
        assert block["input"] == {"location": "Beijing"}

    def test_make_tool_result_message(self):
        msg = self.adapter.make_tool_result_message(
            "call_1", "get_weather", '{"temp": 25}',
        )
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        block = msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_1"
        assert block["content"] == '{"temp": 25}'

    def test_extract_tool_calls_with_tool_use(self):
        message = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "get_weather",
                    "input": {"location": "Beijing"},
                },
            ],
        }
        content, calls = self.adapter.extract_tool_calls(message)
        assert content == "Let me check."
        assert len(calls) == 1
        assert calls[0]["id"] == "tu_1"
        assert calls[0]["name"] == "get_weather"
        assert json.loads(calls[0]["arguments"]) == {"location": "Beijing"}

    def test_extract_tool_calls_no_tools(self):
        message = {"content": [{"type": "text", "text": "Hello!"}]}
        content, calls = self.adapter.extract_tool_calls(message)
        assert content == "Hello!"
        assert calls == []

    def test_extract_tool_calls_string_content(self):
        message = {"content": "Just text"}
        content, calls = self.adapter.extract_tool_calls(message)
        assert content == "Just text"
        assert calls == []

    def test_extract_tool_calls_multiple_text_blocks(self):
        message = {
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
        }
        content, calls = self.adapter.extract_tool_calls(message)
        assert "Part 1" in content
        assert "Part 2" in content

    def test_detect_stream_tool_boundary(self):
        assert self.adapter.detect_stream_tool_boundary("any text") is None
