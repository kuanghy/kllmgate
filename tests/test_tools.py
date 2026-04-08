"""工具适配器的单元测试"""

import json

import pytest

from kllmgate.toolcall.standard import StandardToolAdapter
from kllmgate.toolcall.minimax_xml import MinimaxXmlToolAdapter
from kllmgate.toolcall.deepseek import DeepseekToolAdapter
from kllmgate.toolcall.qwen import QwenToolAdapter
from kllmgate.toolcall.deepseek_qwen import DeepseekQwenToolAdapter
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


class TestDeepseekToolAdapter:

    def setup_method(self):
        self.adapter = DeepseekToolAdapter()

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

    def test_make_tool_result_message(self):
        msg = self.adapter.make_tool_result_message(
            "call_1", "get_weather", '{"temp": 25}',
        )
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"

    def test_extract_tool_calls_canonical_tokens(self):
        """原始模型 token 格式（全角竖线 + ▁ 分隔）"""
        text = (
            "Let me check.\n"
            "<｜tool\u2581calls\u2581begin｜>"
            "<｜tool\u2581call\u2581begin｜>function"
            "<｜tool\u2581sep｜>get_weather\n"
            "```json\n"
            '{"location": "Beijing"}\n'
            "```\n"
            "<｜tool\u2581call\u2581end｜>"
            "<｜tool\u2581calls\u2581end｜>"
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert "Let me check." in content
        assert "tool" not in content.lower() or "calls" not in content.lower()
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        args = json.loads(calls[0]["arguments"])
        assert args["location"] == "Beijing"

    def test_extract_tool_calls_ascii_variant(self):
        """ASCII 竖线 + 下划线变体"""
        text = (
            "<|tool_calls_begin|>"
            "<|tool_call_begin|>function"
            "<|tool_sep|>exec_command\n"
            "```json\n"
            '{"cmd": "ls -la"}\n'
            "```\n"
            "<|tool_call_end|>"
            "<|tool_calls_end|>"
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "exec_command"
        assert json.loads(calls[0]["arguments"])["cmd"] == "ls -la"
        assert content == ""

    def test_extract_tool_calls_spaced_variant(self):
        """带空格的渲染变体"""
        text = (
            "< | tool_calls_begin | >"
            "< | tool_call_begin | >function"
            "< | tool_sep | >get_weather\n"
            "```json\n"
            '{"location": "Hangzhou"}\n'
            "```\n"
            "< | tool_call_end | >"
            "< | tool_calls_end | >"
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    def test_extract_tool_calls_multiple(self):
        text = (
            "<|tool_calls_begin|>"
            "<|tool_call_begin|>function<|tool_sep|>func_a\n"
            "```json\n{\"x\": 1}\n```\n"
            "<|tool_call_end|>\n"
            "<|tool_call_begin|>function<|tool_sep|>func_b\n"
            "```json\n{\"y\": 2}\n```\n"
            "<|tool_call_end|>"
            "<|tool_calls_end|>"
        )
        _, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 2
        assert calls[0]["name"] == "func_a"
        assert calls[1]["name"] == "func_b"

    def test_extract_tool_calls_no_tools(self):
        content, calls = self.adapter.extract_tool_calls(
            {"content": "Just a normal message"},
        )
        assert content == "Just a normal message"
        assert calls == []

    def test_extract_tool_calls_empty_content(self):
        content, calls = self.adapter.extract_tool_calls({"content": ""})
        assert content == ""
        assert calls == []

    def test_extract_tool_calls_invalid_json_fallback(self):
        text = (
            "<|tool_calls_begin|>"
            "<|tool_call_begin|>function<|tool_sep|>broken\n"
            "```json\n{not valid json}\n```\n"
            "<|tool_call_end|>"
            "<|tool_calls_end|>"
        )
        _, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "broken"
        assert "__raw" in calls[0]["arguments"]

    def test_detect_stream_tool_boundary(self):
        assert self.adapter.detect_stream_tool_boundary(
            "Hello <|tool_calls_begin|>",
        ) == 6
        assert self.adapter.detect_stream_tool_boundary(
            "No tools here",
        ) is None

    def test_detect_stream_tool_boundary_canonical(self):
        text = "Preamble<\uff5ctool\u2581calls\u2581begin\uff5c>"
        pos = self.adapter.detect_stream_tool_boundary(text)
        assert pos == len("Preamble")

    def test_stream_buffer_size_covers_spaced_variant(self):
        spaced = "< | tool__calls__begin | >"
        assert self.adapter.stream_buffer_size >= len(spaced)

    def test_extract_double_underscore_no_codefence(self):
        """双下划线 + 无代码围栏 + 第二个 tool_sep 分隔参数"""
        text = (
            "< | tool__calls__begin | >"
            "< | tool__call__begin | >function"
            "< | tool_sep | >exec_command"
            "< | tool_sep | >"
            '{"justification":"test","login":true,"shell":"/bin/zsh"}'
            "< | tool__call__end | >"
            "< | tool__calls__end | >"
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "exec_command"
        args = json.loads(calls[0]["arguments"])
        assert args["login"] is True
        assert args["shell"] == "/bin/zsh"
        assert content == ""

    def test_extract_double_underscore_with_preamble(self):
        """双下划线变体带前导文本"""
        text = (
            "I will review the workspace.\n"
            "< | tool__calls__begin | >"
            "< | tool__call__begin | >function"
            "< | tool_sep | >list_files"
            "< | tool_sep | >"
            '{"path": "/src"}'
            "< | tool__call__end | >"
            "< | tool__calls__end | >"
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert "I will review" in content
        assert len(calls) == 1
        assert calls[0]["name"] == "list_files"

    def test_detect_stream_boundary_double_underscore(self):
        """双下划线变体的流式边界检测"""
        text = "Preamble< | tool__calls__begin | >"
        pos = self.adapter.detect_stream_tool_boundary(text)
        assert pos == len("Preamble")

    def test_extract_preserves_raw_on_parse_failure(self):
        """内部格式不匹配时应保留原文而非静默丢弃"""
        text = (
            "Preamble\n"
            "<|tool_calls_begin|>"
            "unexpected format without code fence"
            "<|tool_calls_end|>"
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert calls == []
        assert "unexpected format" in content
        assert "Preamble" in content

    def test_extract_without_json_lang_tag(self):
        """code block 无 json 语言标签"""
        text = (
            "<|tool_calls_begin|>"
            "<|tool_call_begin|>function<|tool_sep|>test_fn\n"
            "```\n{\"a\": 1}\n```\n"
            "<|tool_call_end|>"
            "<|tool_calls_end|>"
        )
        _, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "test_fn"


class TestQwenToolAdapter:

    def setup_method(self):
        self.adapter = QwenToolAdapter()

    def test_convert_tool_definitions_passthrough(self):
        prompt, tools = self.adapter.convert_tool_definitions(SAMPLE_TOOLS)
        assert prompt is None
        assert tools == SAMPLE_TOOLS

    def test_make_tool_calls_message(self):
        msg = self.adapter.make_tool_calls_message(SAMPLE_CALLS)
        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"

    def test_make_tool_result_message(self):
        msg = self.adapter.make_tool_result_message(
            "call_1", "get_weather", '{"temp": 25}',
        )
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"

    def test_extract_standard_format(self):
        """标准 Qwen 格式：name + arguments"""
        text = (
            "Let me check.\n"
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"location": "Beijing"}}\n'
            "</tool_call>"
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert "Let me check." in content
        assert "<tool_call>" not in content
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        args = json.loads(calls[0]["arguments"])
        assert args["location"] == "Beijing"

    def test_extract_flat_format(self):
        """扁平格式：function 键 + 其余字段作为参数"""
        text = (
            "<tool_call>\n"
            '{"function": "exec_command", "cmd": "ls -la", '
            '"sandbox_permissions": "use_default"}\n'
            "</tool_call>"
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "exec_command"
        args = json.loads(calls[0]["arguments"])
        assert args["cmd"] == "ls -la"
        assert args["sandbox_permissions"] == "use_default"

    def test_extract_multiple(self):
        text = (
            "<tool_call>\n"
            '{"name": "func_a", "arguments": {"x": 1}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "func_b", "arguments": {"y": 2}}\n'
            "</tool_call>"
        )
        _, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 2
        assert calls[0]["name"] == "func_a"
        assert calls[1]["name"] == "func_b"

    def test_extract_no_tools(self):
        content, calls = self.adapter.extract_tool_calls(
            {"content": "Just a normal message"},
        )
        assert content == "Just a normal message"
        assert calls == []

    def test_extract_empty_content(self):
        content, calls = self.adapter.extract_tool_calls({"content": ""})
        assert content == ""
        assert calls == []

    def test_extract_preserves_raw_on_parse_failure(self):
        """JSON 解析全部失败时保留原文"""
        text = "Preamble\n<tool_call>\nnot valid json\n</tool_call>"
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert calls == []
        assert "Preamble" in content
        assert "not valid json" in content

    def test_extract_missing_function_name(self):
        """JSON 缺少函数名时跳过该调用"""
        text = (
            '<tool_call>\n{"arguments": {"x": 1}}\n</tool_call>\n'
            '<tool_call>\n{"name": "ok", "arguments": {}}\n</tool_call>'
        )
        _, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "ok"

    def test_detect_stream_tool_boundary(self):
        assert self.adapter.detect_stream_tool_boundary(
            "Hello <tool_call>",
        ) == 6
        assert self.adapter.detect_stream_tool_boundary(
            "No tools here",
        ) is None

    def test_stream_buffer_size(self):
        assert self.adapter.stream_buffer_size >= len("<tool_call>")

    def test_extract_string_arguments(self):
        """arguments 为字符串时直接透传"""
        text = (
            '<tool_call>\n'
            '{"name": "func", "arguments": "{\\"a\\": 1}"}\n'
            '</tool_call>'
        )
        _, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["arguments"] == '{"a": 1}'


class TestDeepseekQwenToolAdapter:

    def setup_method(self):
        self.adapter = DeepseekQwenToolAdapter()

    def test_extract_deepseek_format(self):
        """能识别 DeepSeek 原生 token 格式"""
        text = (
            "<|tool_calls_begin|>"
            "<|tool_call_begin|>function<|tool_sep|>get_weather\n"
            "```json\n"
            '{"location": "Beijing"}\n'
            "```\n"
            "<|tool_call_end|>"
            "<|tool_calls_end|>"
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    def test_extract_qwen_format(self):
        """能识别 Qwen <tool_call> 格式"""
        text = (
            '<tool_call>\n'
            '{"name": "exec_command", "arguments": {"cmd": "ls"}}\n'
            '</tool_call>'
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "exec_command"

    def test_deepseek_takes_priority(self):
        """同时存在两种格式时优先使用 DeepSeek"""
        text = (
            "<|tool_calls_begin|>"
            "<|tool_call_begin|>function<|tool_sep|>ds_func\n"
            "```json\n{}\n```\n"
            "<|tool_call_end|>"
            "<|tool_calls_end|>\n"
            '<tool_call>\n{"name": "qw_func", "arguments": {}}\n</tool_call>'
        )
        _, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "ds_func"

    def test_fallback_to_qwen_when_no_deepseek(self):
        """没有 DeepSeek 格式时回退到 Qwen"""
        text = (
            "Some thinking...\n"
            '<tool_call>\n'
            '{"function": "exec_command", "cmd": "ls -la"}\n'
            '</tool_call>'
        )
        content, calls = self.adapter.extract_tool_calls({"content": text})
        assert len(calls) == 1
        assert calls[0]["name"] == "exec_command"
        assert "Some thinking" in content

    def test_no_tools(self):
        content, calls = self.adapter.extract_tool_calls(
            {"content": "Normal text"},
        )
        assert content == "Normal text"
        assert calls == []

    def test_detect_stream_boundary_deepseek(self):
        text = "Hello <|tool_calls_begin|>"
        assert self.adapter.detect_stream_tool_boundary(text) == 6

    def test_detect_stream_boundary_qwen(self):
        text = "Hello <tool_call>"
        assert self.adapter.detect_stream_tool_boundary(text) == 6

    def test_detect_stream_boundary_picks_earliest(self):
        text = "A<tool_call>B<|tool_calls_begin|>"
        pos = self.adapter.detect_stream_tool_boundary(text)
        assert pos == 1

    def test_stream_buffer_size(self):
        assert self.adapter.stream_buffer_size >= 24

    def test_convert_tool_definitions_passthrough(self):
        prompt, tools = self.adapter.convert_tool_definitions(SAMPLE_TOOLS)
        assert prompt is None
        assert tools == SAMPLE_TOOLS


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
