"""转换器的单元测试"""

import json
import time

import pytest

from kllmgate.sse import SseEvent
from kllmgate.toolcall.standard import StandardToolAdapter
from kllmgate.toolcall.minimax_xml import MinimaxXmlToolAdapter
from kllmgate.toolcall.anthropic import AnthropicToolAdapter
from kllmgate.converters.passthrough import PassthroughConverter
from kllmgate.converters.openai_responses_to_openai_chat import (
    OpenaiResponsesToOpenaiChatConverter,
)
from kllmgate.converters.openai_chat_to_openai_responses import (
    OpenaiChatToOpenaiResponsesConverter,
)
from kllmgate.converters.openai_chat_to_anthropic_messages import (
    OpenaiChatToAnthropicMessagesConverter,
)
from kllmgate.converters.anthropic_messages_to_openai_chat import (
    AnthropicMessagesToOpenaiChatConverter,
)
from kllmgate.converters.openai_responses_to_anthropic_messages import (
    OpenaiResponsesToAnthropicMessagesConverter,
)
from kllmgate.converters.anthropic_messages_to_openai_responses import (
    AnthropicMessagesToOpenaiResponsesConverter,
)
from kllmgate.converters.openai_chat_tool_adapt import (
    OpenaiChatToolAdaptConverter,
)
from kllmgate.converters.openai_responses_tool_adapt import (
    OpenaiResponsesToolAdaptConverter,
)


async def _collect_stream(converter, events):
    """Helper: 收集流式转换的所有 SSE 文本"""
    async def _gen():
        for ev in events:
            yield ev

    chunks = []
    async for chunk in converter.convert_stream(_gen()):
        chunks.append(chunk)
    return chunks


# ──────────────────── PassthroughConverter ────────────────────


class TestPassthroughConverter:

    def setup_method(self):
        self.converter = PassthroughConverter(StandardToolAdapter())

    def test_convert_request_replaces_model(self):
        body = {"model": "provider/gpt-4.1", "messages": []}
        result = self.converter.convert_request(body, "gpt-4.1")
        assert result["model"] == "gpt-4.1"
        assert result["messages"] == []

    def test_convert_request_preserves_other_fields(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "stream": True,
        }
        result = self.converter.convert_request(body, "m")
        assert result["temperature"] == 0.7
        assert result["stream"] is True

    def test_convert_response_passthrough(self):
        resp = {"id": "1", "choices": [{"message": {"content": "hi"}}]}
        assert self.converter.convert_response(resp) == resp

    @pytest.mark.asyncio
    async def test_convert_stream_passthrough(self):
        events = [
            SseEvent(None, '{"choices": [{"delta": {"content": "a"}}]}', []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(self.converter, events)
        assert len(chunks) == 2
        assert "data:" in chunks[0]


# ──────── OpenaiResponsesToOpenaiChat ────────


class TestOpenaiResponsesToOpenaiChatRequest:

    def setup_method(self):
        self.converter = OpenaiResponsesToOpenaiChatConverter(
            StandardToolAdapter(),
        )

    def test_basic_text_input(self):
        body = {
            "model": "provider/gpt-4.1",
            "input": [
                {"role": "user", "content": "Hello"},
            ],
        }
        result = self.converter.convert_request(body, "gpt-4.1")
        assert result["model"] == "gpt-4.1"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"

    def test_instructions_become_system(self):
        body = {
            "model": "p/m",
            "instructions": "You are helpful.",
            "input": [{"role": "user", "content": "Hi"}],
        }
        result = self.converter.convert_request(body, "m")
        assert result["messages"][0]["role"] == "system"
        assert "You are helpful." in result["messages"][0]["content"]

    def test_model_role_becomes_assistant(self):
        body = {
            "model": "p/m",
            "input": [
                {"role": "user", "content": "Hi"},
                {"role": "model", "content": "Hello"},
            ],
        }
        result = self.converter.convert_request(body, "m")
        assert result["messages"][1]["role"] == "assistant"

    def test_developer_role_becomes_system(self):
        body = {
            "model": "p/m",
            "input": [
                {"role": "developer", "content": "Be concise"},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = self.converter.convert_request(body, "m")
        assert result["messages"][0]["role"] == "system"
        assert "Be concise" in result["messages"][0]["content"]

    def test_stream_adds_stream_options(self):
        body = {
            "model": "p/m",
            "input": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        result = self.converter.convert_request(body, "m")
        assert result["stream"] is True
        assert result["stream_options"]["include_usage"] is True

    def test_function_call_input(self):
        body = {
            "model": "p/m",
            "input": [
                {"role": "user", "content": "weather?"},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "get_weather",
                    "arguments": '{"city": "Beijing"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": "Sunny 25°C",
                },
            ],
        }
        result = self.converter.convert_request(body, "m")
        has_assistant = any(
            m["role"] == "assistant" for m in result["messages"]
        )
        has_tool = any(
            m["role"] == "tool" for m in result["messages"]
        )
        assert has_assistant
        assert has_tool

    def test_string_input_item(self):
        body = {
            "model": "p/m",
            "input": "Just a string",
        }
        result = self.converter.convert_request(body, "m")
        assert any(m["content"] == "Just a string" for m in result["messages"])

    def test_tools_delegated_to_adapter(self):
        converter = OpenaiResponsesToOpenaiChatConverter(
            MinimaxXmlToolAdapter(),
        )
        body = {
            "model": "p/m",
            "input": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }
        result = converter.convert_request(body, "m")
        assert result["messages"][0]["role"] == "system"
        assert "search" in result["messages"][0]["content"]
        assert "tools" not in result


class TestOpenaiResponsesToOpenaiChatResponse:

    def setup_method(self):
        self.converter = OpenaiResponsesToOpenaiChatConverter(
            StandardToolAdapter(),
        )

    def test_basic_text_response(self):
        chat_resp = {
            "id": "chatcmpl-1",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = self.converter.convert_response(chat_resp)
        assert result["id"] == "chatcmpl-1"
        assert result["object"] == "response"
        assert result["status"] == "completed"
        output_msg = result["output"][0]
        assert output_msg["type"] == "message"
        assert output_msg["content"][0]["type"] == "output_text"
        assert output_msg["content"][0]["text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_calls_response(self):
        chat_resp = {
            "id": "chatcmpl-2",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Beijing"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = self.converter.convert_response(chat_resp)
        assert result["status"] == "completed"
        fc_items = [o for o in result["output"] if o["type"] == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "get_weather"

    def test_finish_reason_length_maps_to_incomplete(self):
        chat_resp = {
            "id": "chatcmpl-3",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "message": {"role": "assistant", "content": "partial..."},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        result = self.converter.convert_response(chat_resp)
        assert result["status"] == "incomplete"


class TestOpenaiResponsesToOpenaiChatStream:

    def setup_method(self):
        self.converter = OpenaiResponsesToOpenaiChatConverter(
            StandardToolAdapter(),
        )

    @pytest.mark.asyncio
    async def test_basic_text_stream(self):
        events = [
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "choices": [{"delta": {"content": "Hello"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "choices": [{"delta": {"content": " world"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert "response.created" in full
        assert "response.output_text.delta" in full
        assert "response.completed" in full

    @pytest.mark.asyncio
    async def test_standard_tool_calls_stream(self):
        events = [
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "model": "gpt-4.1",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "model": "gpt-4.1",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather"},
                        }],
                    },
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "model": "gpt-4.1",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '{"city":"Beijing"}'},
                        }],
                    },
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "model": "gpt-4.1",
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert "response.function_call_arguments.delta" in full
        assert "get_weather" in full


# ──────── OpenaiChatToOpenaiResponses ────────


class TestOpenaiChatToOpenaiResponsesRequest:

    def setup_method(self):
        self.converter = OpenaiChatToOpenaiResponsesConverter(
            StandardToolAdapter(),
        )

    def test_basic_messages(self):
        body = {
            "model": "p/m",
            "messages": [
                {"role": "user", "content": "Hi"},
            ],
        }
        result = self.converter.convert_request(body, "m")
        assert result["model"] == "m"
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"

    def test_system_messages_become_instructions(self):
        body = {
            "model": "p/m",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = self.converter.convert_request(body, "m")
        assert result["instructions"] == "Be helpful."

    def test_assistant_role_becomes_model(self):
        body = {
            "model": "p/m",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        }
        result = self.converter.convert_request(body, "m")
        model_msgs = [
            i for i in result["input"] if i.get("role") == "model"
        ]
        assert len(model_msgs) == 1

    def test_tool_messages_become_function_call_output(self):
        body = {
            "model": "p/m",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "BJ"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "Sunny",
                },
            ],
        }
        result = self.converter.convert_request(body, "m")
        fc_items = [
            i for i in result["input"]
            if i.get("type") == "function_call"
        ]
        fco_items = [
            i for i in result["input"]
            if i.get("type") == "function_call_output"
        ]
        assert len(fc_items) == 1
        assert len(fco_items) == 1


class TestOpenaiChatToOpenaiResponsesResponse:

    def setup_method(self):
        self.converter = OpenaiChatToOpenaiResponsesConverter(
            StandardToolAdapter(),
        )

    def test_basic_text_response(self):
        resp = {
            "id": "resp_1",
            "created_at": 1700000000,
            "model": "gpt-4.1",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        result = self.converter.convert_response(resp)
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_function_call_in_response(self):
        resp = {
            "id": "resp_2",
            "created_at": 1700000000,
            "model": "gpt-4.1",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Beijing"}',
            }],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }
        result = self.converter.convert_response(resp)
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"

    def test_incomplete_status(self):
        resp = {
            "id": "resp_3",
            "created_at": 1700000000,
            "model": "m",
            "status": "incomplete",
            "output": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        result = self.converter.convert_response(resp)
        assert result["choices"][0]["finish_reason"] == "length"


# ──────── OpenaiChatToAnthropicMessages ────────


class TestOpenaiChatToAnthropicMessagesRequest:

    def setup_method(self):
        self.converter = OpenaiChatToAnthropicMessagesConverter(
            AnthropicToolAdapter(),
        )

    def test_basic_messages(self):
        body = {
            "model": "p/claude",
            "messages": [
                {"role": "user", "content": "Hi"},
            ],
        }
        result = self.converter.convert_request(body, "claude")
        assert result["model"] == "claude"
        assert len(result["messages"]) == 1

    def test_system_extracted_to_top_level(self):
        body = {
            "model": "p/m",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = self.converter.convert_request(body, "m")
        assert result["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in result["messages"])

    def test_default_max_tokens(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = self.converter.convert_request(body, "m")
        assert result["max_tokens"] == 4096

    def test_explicit_max_tokens(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1000,
        }
        result = self.converter.convert_request(body, "m")
        assert result["max_tokens"] == 1000

    def test_max_completion_tokens_fallback(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 2000,
        }
        result = self.converter.convert_request(body, "m")
        assert result["max_tokens"] == 2000

    def test_tools_converted(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object"},
                },
            }],
        }
        result = self.converter.convert_request(body, "m")
        assert "tools" in result
        assert result["tools"][0]["name"] == "search"
        assert "input_schema" in result["tools"][0]

    def test_temperature_and_top_p_passthrough(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
        }
        result = self.converter.convert_request(body, "m")
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9


class TestOpenaiChatToAnthropicMessagesResponse:

    def setup_method(self):
        self.converter = OpenaiChatToAnthropicMessagesConverter(
            AnthropicToolAdapter(),
        )

    def test_basic_text_response(self):
        anthropic_resp = {
            "id": "msg_1",
            "model": "claude",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = self.converter.convert_response(anthropic_resp)
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_tool_use_response(self):
        anthropic_resp = {
            "id": "msg_2",
            "model": "claude",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "get_weather",
                    "input": {"city": "Beijing"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = self.converter.convert_response(anthropic_resp)
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"

    def test_stop_reason_mapping(self):
        for stop_reason, expected in [
            ("end_turn", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
        ]:
            resp = {
                "id": "msg",
                "model": "claude",
                "content": [{"type": "text", "text": ""}],
                "stop_reason": stop_reason,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
            result = self.converter.convert_response(resp)
            assert result["choices"][0]["finish_reason"] == expected


# ──────── AnthropicMessagesToOpenaiChat ────────


class TestAnthropicMessagesToOpenaiChatRequest:

    def setup_method(self):
        self.converter = AnthropicMessagesToOpenaiChatConverter(
            StandardToolAdapter(),
        )

    def test_basic_messages(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        }
        result = self.converter.convert_request(body, "m")
        assert result["model"] == "m"
        assert result["messages"][0]["role"] == "user"

    def test_system_to_messages(self):
        body = {
            "model": "p/m",
            "system": "Be helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        }
        result = self.converter.convert_request(body, "m")
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Be helpful."
        assert result["messages"][1]["role"] == "user"

    def test_tool_use_to_tool_calls(self):
        body = {
            "model": "p/m",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "get_weather",
                            "input": {"city": "BJ"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "content": "Sunny",
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result = self.converter.convert_request(body, "m")
        assistant_msg = result["messages"][1]
        assert "tool_calls" in assistant_msg
        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "tool"

    def test_anthropic_tools_converted(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object"},
            }],
            "max_tokens": 1024,
        }
        result = self.converter.convert_request(body, "m")
        assert "tools" in result
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "search"


class TestAnthropicMessagesToOpenaiChatResponse:

    def setup_method(self):
        self.converter = AnthropicMessagesToOpenaiChatConverter(
            StandardToolAdapter(),
        )

    def test_basic_text_response(self):
        chat_resp = {
            "id": "chatcmpl-1",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self.converter.convert_response(chat_resp)
        assert result["id"] == "chatcmpl-1"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"

    def test_tool_calls_to_tool_use(self):
        chat_resp = {
            "id": "chatcmpl-2",
            "created": 1700000000,
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "BJ"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = self.converter.convert_response(chat_resp)
        assert result["stop_reason"] == "tool_use"
        tu_blocks = [
            b for b in result["content"]
            if b["type"] == "tool_use"
        ]
        assert len(tu_blocks) == 1

    def test_finish_reason_mapping(self):
        for finish_reason, expected in [
            ("stop", "end_turn"),
            ("tool_calls", "tool_use"),
            ("length", "max_tokens"),
        ]:
            resp = {
                "id": "c",
                "created": 0,
                "model": "m",
                "choices": [{
                    "message": {"content": ""},
                    "finish_reason": finish_reason,
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            result = self.converter.convert_response(resp)
            assert result["stop_reason"] == expected


# ──────── OpenaiResponsesToAnthropicMessages ────────


class TestOpenaiResponsesToAnthropicMessagesRequest:

    def setup_method(self):
        self.converter = OpenaiResponsesToAnthropicMessagesConverter(
            AnthropicToolAdapter(),
        )

    def test_basic_input(self):
        body = {
            "model": "p/claude",
            "input": [{"role": "user", "content": "Hi"}],
        }
        result = self.converter.convert_request(body, "claude")
        assert result["model"] == "claude"
        assert len(result["messages"]) == 1
        assert result["max_tokens"] == 4096

    def test_instructions_become_system(self):
        body = {
            "model": "p/m",
            "instructions": "Be helpful.",
            "input": [{"role": "user", "content": "Hi"}],
        }
        result = self.converter.convert_request(body, "m")
        assert result["system"] == "Be helpful."


# ──────── AnthropicMessagesToOpenaiResponses ────────


class TestAnthropicMessagesToOpenaiResponsesRequest:

    def setup_method(self):
        self.converter = AnthropicMessagesToOpenaiResponsesConverter(
            StandardToolAdapter(),
        )

    def test_basic_input(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        }
        result = self.converter.convert_request(body, "m")
        assert result["model"] == "m"
        assert len(result["input"]) >= 1

    def test_system_becomes_instructions(self):
        body = {
            "model": "p/m",
            "system": "Be helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        }
        result = self.converter.convert_request(body, "m")
        assert result["instructions"] == "Be helpful."


class TestAnthropicMessagesToOpenaiResponsesResponse:

    def setup_method(self):
        self.converter = AnthropicMessagesToOpenaiResponsesConverter(
            StandardToolAdapter(),
        )

    def test_basic_text_response(self):
        resp = {
            "id": "resp_1",
            "created_at": 1700000000,
            "model": "m",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        result = self.converter.convert_response(resp)
        assert result["type"] == "message"
        text_blocks = [
            b for b in result["content"] if b["type"] == "text"
        ]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Hello!"


# ──────── Tool Adapt Converters ────────


class TestOpenaiChatToolAdaptConverter:

    def setup_method(self):
        self.converter = OpenaiChatToolAdaptConverter(
            MinimaxXmlToolAdapter(),
        )

    def test_tools_injected_to_system(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search things",
                    "parameters": {"type": "object"},
                },
            }],
        }
        result = self.converter.convert_request(body, "m")
        assert result["model"] == "m"
        assert "tools" not in result
        sys_msg = result["messages"][0]
        assert sys_msg["role"] == "system"
        assert "search" in sys_msg["content"]

    def test_no_tools_passthrough(self):
        body = {
            "model": "p/m",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = self.converter.convert_request(body, "m")
        assert result["model"] == "m"
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_stream_normalizes_minimax_tool_call(self):
        events = [
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [{
                    "delta": {
                        "content": (
                            '<minimax:tool_call><invoke name="search">'
                            '<parameter name="q">python</parameter>'
                        ),
                    },
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [{
                    "delta": {
                        "content": '</invoke></minimax:tool_call>',
                    },
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "chatcmpl-1",
                "model": "m",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert '"tool_calls"' in full
        assert "<minimax:tool_call>" not in full


class TestOpenaiResponsesToolAdaptConverter:

    def setup_method(self):
        self.converter = OpenaiResponsesToolAdaptConverter(
            MinimaxXmlToolAdapter(),
        )

    def test_tools_injected_to_instructions(self):
        body = {
            "model": "p/m",
            "input": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object"},
                },
            }],
        }
        result = self.converter.convert_request(body, "m")
        assert "tools" not in result

    def test_response_normalizes_minimax_tool_call(self):
        response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "m",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": (
                        'before <minimax:tool_call><invoke name="search">'
                        '<parameter name="q">python</parameter>'
                        '</invoke></minimax:tool_call>'
                    ),
                }],
            }],
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        }
        result = self.converter.convert_response(response)
        function_calls = [
            item for item in result["output"]
            if item["type"] == "function_call"
        ]
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_stream_normalizes_minimax_tool_call(self):
        events = [
            SseEvent("response.created", json.dumps({
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "in_progress",
                    "model": "m",
                    "output": [],
                },
                "sequence_number": 0,
            }), []),
            SseEvent("response.in_progress", json.dumps({
                "type": "response.in_progress",
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "in_progress",
                    "model": "m",
                    "output": [],
                },
                "sequence_number": 1,
            }), []),
            SseEvent("response.output_item.added", json.dumps({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
                "sequence_number": 2,
            }), []),
            SseEvent("response.content_part.added", json.dumps({
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
                "sequence_number": 3,
            }), []),
            SseEvent("response.output_text.delta", json.dumps({
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": (
                    '<minimax:tool_call><invoke name="search">'
                    '<parameter name="q">python</parameter>'
                ),
                "sequence_number": 4,
            }), []),
            SseEvent("response.output_text.delta", json.dumps({
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": '</invoke></minimax:tool_call>',
                "sequence_number": 5,
            }), []),
            SseEvent("response.completed", json.dumps({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "m",
                    "output": [],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 2,
                        "total_tokens": 3,
                    },
                },
                "sequence_number": 6,
            }), []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert "response.function_call_arguments.delta" in full
        assert "search" in full
        assert "<minimax:tool_call>" not in full


# ──────── Review 修复：流式 usage、缓冲、多模态、system list ────────


class TestStreamUsageResponsesToAnthropic:
    """跨协议流式 usage 修复：Responses->Anthropic 上游"""

    def setup_method(self):
        self.converter = OpenaiResponsesToAnthropicMessagesConverter(
            AnthropicToolAdapter(),
        )

    @pytest.mark.asyncio
    async def test_stream_usage_from_anthropic_events(self):
        events = [
            SseEvent("message_start", json.dumps({
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude",
                    "content": [],
                    "usage": {"input_tokens": 42, "output_tokens": 0},
                },
            }), []),
            SseEvent("content_block_delta", json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            }), []),
            SseEvent("message_delta", json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 7},
            }), []),
            SseEvent("message_stop", json.dumps({
                "type": "message_stop",
            }), []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert "response.completed" in full
        completed_data = None
        for chunk in chunks:
            if "response.completed" in chunk:
                for line in chunk.split("\n"):
                    if line.startswith("data: "):
                        completed_data = json.loads(line[6:])
        assert completed_data is not None
        usage = completed_data["response"]["usage"]
        assert usage["input_tokens"] == 42
        assert usage["output_tokens"] == 7
        assert usage["total_tokens"] == 49


class TestStreamUsageAnthropicToResponses:
    """跨协议流式 usage 修复：Anthropic->Responses 上游"""

    def setup_method(self):
        self.converter = AnthropicMessagesToOpenaiResponsesConverter(
            StandardToolAdapter(),
        )

    @pytest.mark.asyncio
    async def test_stream_usage_from_responses_events(self):
        events = [
            SseEvent("response.created", json.dumps({
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "model": "gpt-4.1",
                    "status": "in_progress",
                    "output": [],
                },
            }), []),
            SseEvent("response.output_text.delta", json.dumps({
                "type": "response.output_text.delta",
                "delta": "Hello",
            }), []),
            SseEvent("response.completed", json.dumps({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "model": "gpt-4.1",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            }), []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert "message_delta" in full
        for chunk in chunks:
            if "message_delta" in chunk and "output_tokens" in chunk:
                for line in chunk.split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "message_delta":
                            assert data["usage"]["output_tokens"] == 5


class TestStreamBufferSizeProperty:
    """stream_buffer_size 属性：标准适配器为 0，MiniMax 为 19"""

    def test_standard_adapter_no_buffer(self):
        assert StandardToolAdapter().stream_buffer_size == 0

    def test_anthropic_adapter_no_buffer(self):
        assert AnthropicToolAdapter().stream_buffer_size == 0

    def test_minimax_adapter_has_buffer(self):
        adapter = MinimaxXmlToolAdapter()
        assert adapter.stream_buffer_size == len("<minimax:tool_call>")

    @pytest.mark.asyncio
    async def test_standard_adapter_no_delay(self):
        """标准适配器流式时文本不应被延迟"""
        converter = OpenaiChatToolAdaptConverter(StandardToolAdapter())
        events = [
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": "Hello world"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(converter, events)
        content_chunks = [
            c for c in chunks
            if '"content"' in c and "Hello" in c
        ]
        assert len(content_chunks) == 1
        data = json.loads(
            content_chunks[0].removeprefix("data: ").strip(),
        )
        assert data["choices"][0]["delta"]["content"] == "Hello world"


class TestMultimodalContentMapping:
    """多模态内容映射修复"""

    def test_responses_to_chat_content_list_preserved(self):
        converter = OpenaiResponsesToOpenaiChatConverter(
            StandardToolAdapter(),
        )
        body = {
            "model": "p/m",
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image"},
                    {
                        "type": "input_image",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            }],
        }
        result = converter.convert_request(body, "m")
        user_msg = [
            m for m in result["messages"]
            if m["role"] == "user"
        ][0]
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) == 2
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"

    def test_responses_to_chat_consecutive_user_lists_merged(self):
        converter = OpenaiResponsesToOpenaiChatConverter(
            StandardToolAdapter(),
        )
        body = {
            "model": "p/m",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Part 1"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Part 2"}],
                },
            ],
        }
        result = converter.convert_request(body, "m")
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 1
        content = user_msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["text"] == "Part 1"
        assert content[1]["text"] == "Part 2"

    def test_anthropic_to_chat_image_mapped(self):
        converter = AnthropicMessagesToOpenaiChatConverter(
            StandardToolAdapter(),
        )
        body = {
            "model": "p/m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/photo.jpg",
                        },
                    },
                ],
            }],
            "max_tokens": 1024,
        }
        result = converter.convert_request(body, "m")
        user_msgs = [
            m for m in result["messages"]
            if m["role"] == "user"
        ]
        assert len(user_msgs) == 1
        content = user_msgs[0]["content"]
        assert isinstance(content, list)
        assert any(p.get("type") == "image_url" for p in content)

    def test_anthropic_to_chat_base64_image_mapped(self):
        converter = AnthropicMessagesToOpenaiChatConverter(
            StandardToolAdapter(),
        )
        body = {
            "model": "p/m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "abc123",
                        },
                    },
                ],
            }],
            "max_tokens": 1024,
        }
        result = converter.convert_request(body, "m")
        user_msg = [
            m for m in result["messages"]
            if m["role"] == "user"
        ][0]
        img_part = [
            p for p in user_msg["content"]
            if p.get("type") == "image_url"
        ][0]
        assert "data:image/jpeg;base64,abc123" in (
            img_part["image_url"]["url"]
        )

    def test_anthropic_to_chat_unknown_block_preserved(self):
        converter = AnthropicMessagesToOpenaiChatConverter(
            StandardToolAdapter(),
        )
        body = {
            "model": "p/m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "custom_block", "data": "xyz"},
                ],
            }],
            "max_tokens": 1024,
        }
        result = converter.convert_request(body, "m")
        user_msg = [
            m for m in result["messages"]
            if m["role"] == "user"
        ][0]
        assert any(
            p.get("type") == "custom_block" for p in user_msg["content"]
        )


class TestSystemContentListNormalization:
    """system content 为 list 时不再崩溃"""

    def test_chat_to_anthropic_system_list(self):
        converter = OpenaiChatToAnthropicMessagesConverter(
            AnthropicToolAdapter(),
        )
        body = {
            "model": "p/m",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Be helpful."},
                        {"type": "text", "text": "Be concise."},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
        }
        result = converter.convert_request(body, "m")
        assert "Be helpful." in result["system"]
        assert "Be concise." in result["system"]

    def test_chat_to_responses_system_list(self):
        converter = OpenaiChatToOpenaiResponsesConverter(
            StandardToolAdapter(),
        )
        body = {
            "model": "p/m",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "System prompt."},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
        }
        result = converter.convert_request(body, "m")
        assert "System prompt." in result.get("instructions", "")

    def test_chat_tool_adapt_system_list(self):
        converter = OpenaiChatToolAdaptConverter(MinimaxXmlToolAdapter())
        body = {
            "model": "p/m",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Base system."},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object"},
                },
            }],
        }
        result = converter.convert_request(body, "m")
        sys_msg = result["messages"][0]
        assert sys_msg["role"] == "system"
        assert "Base system." in sys_msg["content"]
        assert "search" in sys_msg["content"]
