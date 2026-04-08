"""思考过程提取器的单元测试"""

import json

import pytest

from kllmgate.thinking import (
    NullThinkingExtractor,
    ThinkTagExtractor,
    ThinkingTokenExtractor,
    LvlEntryThinkingExtractor,
)
from kllmgate.sse import SseEvent
from kllmgate.toolcall.standard import StandardToolAdapter
from kllmgate.toolcall.deepseek_qwen import DeepseekQwenToolAdapter
from kllmgate.converters.openai_chat_tool_adapt import (
    OpenaiChatToolAdaptConverter,
)
from kllmgate.converters.anthropic_messages_to_openai_chat import (
    AnthropicMessagesToOpenaiChatConverter,
)
from kllmgate.converters.anthropic_messages_to_openai_responses import (
    AnthropicMessagesToOpenaiResponsesConverter,
)


# ──────── NullThinkingExtractor ────────


class TestNullThinkingExtractor:

    def setup_method(self):
        self.ext = NullThinkingExtractor()

    def test_extract_returns_empty_reasoning(self):
        reasoning, remaining = self.ext.extract("Hello world")
        assert reasoning == ""
        assert remaining == "Hello world"

    def test_extract_with_think_tags_unchanged(self):
        text = "<think>some thinking</think>answer"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == text

    def test_stream_buffer_size_zero(self):
        assert self.ext.stream_buffer_size == 0

    def test_find_open_tag_always_none(self):
        assert self.ext.find_open_tag("<think>test") is None

    def test_find_close_tag_always_none(self):
        assert self.ext.find_close_tag("</think>") is None


# ──────── ThinkTagExtractor ────────


class TestThinkTagExtractor:

    def setup_method(self):
        self.ext = ThinkTagExtractor()

    def test_extract_basic(self):
        text = "<think>reasoning here</think>answer text"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "reasoning here"
        assert remaining == "answer text"

    def test_extract_with_whitespace(self):
        text = "<think>\n  step 1\n  step 2\n</think>\nfinal answer"
        reasoning, remaining = self.ext.extract(text)
        assert "step 1" in reasoning
        assert "step 2" in reasoning
        assert remaining == "final answer"

    def test_extract_no_tags(self):
        text = "plain text without thinking"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == text

    def test_extract_empty_think_block(self):
        text = "<think></think>answer"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == "answer"

    def test_extract_multiple_blocks(self):
        text = (
            "<think>first thought</think>"
            "middle text"
            "<think>second thought</think>"
            "end"
        )
        reasoning, remaining = self.ext.extract(text)
        assert "first thought" in reasoning
        assert "second thought" in reasoning
        assert "middle text" in remaining
        assert "end" in remaining

    def test_extract_with_preamble(self):
        text = "preamble text\n<think>reasoning</think>answer"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "reasoning"
        assert "preamble text" in remaining
        assert "answer" in remaining

    def test_extract_unclosed_tag(self):
        """未关闭的标签不应被提取"""
        text = "<think>reasoning without close tag"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == text

    def test_stream_buffer_size(self):
        assert self.ext.stream_buffer_size == len("</think>")

    def test_find_open_tag(self):
        text = "prefix<think>content"
        result = self.ext.find_open_tag(text)
        assert result is not None
        tag_start, content_start = result
        assert tag_start == 6
        assert content_start == 13

    def test_find_open_tag_not_found(self):
        assert self.ext.find_open_tag("no tags here") is None

    def test_find_close_tag(self):
        text = "reasoning</think>after"
        result = self.ext.find_close_tag(text)
        assert result is not None
        content_end, tag_end = result
        assert content_end == 9
        assert tag_end == 17

    def test_find_close_tag_with_start(self):
        text = "</think>first</think>second"
        result = self.ext.find_close_tag(text, start=8)
        assert result is not None
        assert result[0] == 13

    def test_find_close_tag_not_found(self):
        assert self.ext.find_close_tag("no close tag") is None


# ──────── ThinkingTokenExtractor ────────


class TestThinkingTokenExtractor:

    def setup_method(self):
        self.ext = ThinkingTokenExtractor()

    def test_extract_ascii(self):
        text = "<|thinking|>reasoning here<|/thinking|>answer"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "reasoning here"
        assert remaining == "answer"

    def test_extract_unicode(self):
        text = "<\uff5cthinking\uff5c>deep thought<\uff5c/thinking\uff5c>result"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "deep thought"
        assert remaining == "result"

    def test_extract_spaced_variant(self):
        text = (
            "< | thinking | >\n"
            "分析 config.example.toml 的更新\n"
            "< | /thinking | >\n"
            "这是最终回答"
        )
        reasoning, remaining = self.ext.extract(text)
        assert "分析 config.example.toml" in reasoning
        assert "最终回答" in remaining

    def test_extract_spaced_close_without_slash(self):
        """部分推理框架闭合标签省略 /"""
        text = (
            "< | thinking | >\n"
            "thinking content\n"
            "< | thinking | >\n"
            "answer"
        )
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "thinking content"
        assert "answer" in remaining

    def test_extract_with_inner_structured_tags(self):
        """思考内容包含结构化标记（如 section 等），应原样保留"""
        text = (
            "<|thinking|>\n"
            "分析代码变更\n"
            "< | section | 重点修改区域 | section >\n"
            "src/main.py 有关键修改\n"
            "< | section_end | >\n"
            "<|/thinking|>\n"
            "审查完成"
        )
        reasoning, remaining = self.ext.extract(text)
        assert "分析代码变更" in reasoning
        assert "重点修改区域" in reasoning
        assert "src/main.py" in reasoning
        assert "审查完成" in remaining

    def test_extract_multiple_blocks(self):
        text = (
            "<|thinking|>first thought<|/thinking|>"
            "middle text"
            "<|thinking|>second thought<|/thinking|>"
            "final"
        )
        reasoning, remaining = self.ext.extract(text)
        assert "first thought" in reasoning
        assert "second thought" in reasoning
        assert "middle text" in remaining
        assert "final" in remaining

    def test_extract_empty_block(self):
        text = "<|thinking|><|/thinking|>answer"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == "answer"

    def test_extract_no_tags(self):
        text = "just plain text"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == text

    def test_extract_unclosed(self):
        text = "<|thinking|>reasoning without close tag"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == text

    def test_stream_buffer_size(self):
        assert self.ext.stream_buffer_size >= len("< | /thinking | >")

    def test_find_open_tag_ascii(self):
        text = "prefix<|thinking|>content"
        result = self.ext.find_open_tag(text)
        assert result is not None
        assert result[0] == 6
        assert result[1] == 18

    def test_find_open_tag_spaced(self):
        text = "prefix< | thinking | >content"
        result = self.ext.find_open_tag(text)
        assert result is not None
        assert result[0] == 6

    def test_find_open_tag_not_found(self):
        assert self.ext.find_open_tag("no tags here") is None

    def test_find_close_tag_with_slash(self):
        text = "reasoning<|/thinking|>after"
        result = self.ext.find_close_tag(text)
        assert result is not None
        content_end, tag_end = result
        assert content_end == 9
        assert tag_end == 22

    def test_find_close_tag_without_slash(self):
        """闭合标签省略 / 时也应匹配"""
        text = "reasoning< | thinking | >after"
        result = self.ext.find_close_tag(text)
        assert result is not None
        assert result[0] == 9

    def test_find_close_tag_spaced_with_slash(self):
        text = "reasoning< | /thinking | >after"
        result = self.ext.find_close_tag(text)
        assert result is not None
        assert result[0] == 9

    def test_find_close_tag_not_found(self):
        assert self.ext.find_close_tag("no close tag") is None


# ──────── LvlEntryThinkingExtractor ────────


class TestLvlEntryThinkingExtractor:

    def setup_method(self):
        self.ext = LvlEntryThinkingExtractor()

    def test_extract_basic(self):
        text = "<lvl_0_entry>reasoning here</lvl_0_entry>answer"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "reasoning here"
        assert remaining == "answer"

    def test_extract_strips_inner_tags(self):
        text = (
            "<lvl_0_entry>"
            "outer thinking\n"
            "<lvl_1_entry>inner thinking</lvl_1_entry>\n"
            "more outer"
            "</lvl_0_entry>"
            "answer"
        )
        reasoning, remaining = self.ext.extract(text)
        assert "outer thinking" in reasoning
        assert "inner thinking" in reasoning
        assert "more outer" in reasoning
        assert "<lvl_1_entry>" not in reasoning
        assert "</lvl_1_entry>" not in reasoning
        assert remaining == "answer"

    def test_extract_with_whitespace(self):
        text = "<lvl_0_entry>\n  thinking\n</lvl_0_entry>\nresult"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "thinking"
        assert remaining == "result"

    def test_extract_no_tags(self):
        text = "just plain text"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == text

    def test_extract_unclosed_tag(self):
        text = "<lvl_0_entry>reasoning without close"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == text

    def test_stream_buffer_size(self):
        assert self.ext.stream_buffer_size >= len("</lvl_0_entry>")
        assert self.ext.stream_buffer_size >= len("</lvl_0_thought>")

    def test_find_open_tag_entry(self):
        result = self.ext.find_open_tag("text<lvl_0_entry>more")
        assert result is not None
        assert result == (4, 17)

    def test_find_open_tag_non_entry_ignored(self):
        """流式模式只检测 entry，plan/thought 等不触发"""
        assert self.ext.find_open_tag("text<lvl_0_plan>more") is None
        assert self.ext.find_open_tag("text<lvl_0_thought>more") is None

    def test_find_open_tag_inner_level_ignored(self):
        """内层标签不触发流式检测"""
        assert self.ext.find_open_tag("text<lvl_1_entry>more") is None

    def test_find_close_tag_entry(self):
        result = self.ext.find_close_tag("text</lvl_0_entry>more")
        assert result is not None
        assert result == (4, 18)

    def test_find_close_tag_non_entry_ignored(self):
        """流式模式只检测 entry 关闭，内部 plan/thought 不触发提前退出"""
        assert self.ext.find_close_tag("text</lvl_0_plan>more") is None
        assert self.ext.find_close_tag("text</lvl_0_thought>more") is None

    def test_find_close_tag_inner_level_ignored(self):
        assert self.ext.find_close_tag("text</lvl_1_entry>more") is None

    def test_extract_multiline_reasoning(self):
        text = (
            "<lvl_0_entry>\n"
            "Let me think step by step:\n"
            "1. First consideration\n"
            "2. Second consideration\n"
            "</lvl_0_entry>\n"
            "Based on my analysis..."
        )
        reasoning, remaining = self.ext.extract(text)
        assert "step by step" in reasoning
        assert "First consideration" in reasoning
        assert "Second consideration" in reasoning
        assert remaining == "Based on my analysis..."

    def test_extract_lvl_0_plan_top_level(self):
        """顶层 <lvl_0_plan> 块可被提取"""
        text = "<lvl_0_plan>制定行动计划</lvl_0_plan>最终回答"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "制定行动计划"
        assert remaining == "最终回答"

    def test_extract_lvl_0_thought_top_level(self):
        text = "<lvl_0_thought>深入思考过程</lvl_0_thought>结论"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "深入思考过程"
        assert remaining == "结论"

    def test_extract_mixed_suffixes_at_top_level(self):
        """多种后缀并列在顶层时全部提取"""
        text = (
            "<lvl_0_plan>计划</lvl_0_plan>\n"
            "<lvl_0_entry>推理</lvl_0_entry>\n"
            "最终回答"
        )
        reasoning, remaining = self.ext.extract(text)
        assert "计划" in reasoning
        assert "推理" in reasoning
        assert remaining == "最终回答"

    def test_extract_nested_plan_inside_entry(self):
        """嵌套在 entry 内的 plan/thought 标签被清理"""
        text = (
            "<lvl_0_entry>\n"
            "<lvl_0_plan>行动计划</lvl_0_plan>\n"
            "<lvl_0_thought>思考过程</lvl_0_thought>\n"
            "<lvl_1_entry>\n"
            "<lvl_1_plan>子任务计划</lvl_1_plan>\n"
            "</lvl_1_entry>\n"
            "<lvl_0_output>阶段结论</lvl_0_output>\n"
            "</lvl_0_entry>\n"
            "最终回答"
        )
        reasoning, remaining = self.ext.extract(text)
        assert "行动计划" in reasoning
        assert "思考过程" in reasoning
        assert "子任务计划" in reasoning
        assert "阶段结论" in reasoning
        assert "<lvl_" not in reasoning
        assert remaining == "最终回答"

    def test_extract_lvl_0_refine(self):
        text = "<lvl_0_refine>修正之前的分析</lvl_0_refine>修正后的答案"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == "修正之前的分析"
        assert remaining == "修正后的答案"

    def test_extract_mismatched_suffix_not_matched(self):
        """开闭标签后缀不一致时不匹配（反向引用保证配对）"""
        text = "<lvl_0_plan>content</lvl_0_entry>answer"
        reasoning, remaining = self.ext.extract(text)
        assert reasoning == ""
        assert remaining == text


# ──────── Converter 集成测试：非流式 ────────


async def _collect_stream(converter, events):
    async def _gen():
        for ev in events:
            yield ev
    chunks = []
    async for chunk in converter.convert_stream(_gen()):
        chunks.append(chunk)
    return chunks


class TestChatToolAdaptWithThinking:
    """OpenaiChatToolAdaptConverter + ThinkTagExtractor 集成测试"""

    def setup_method(self):
        self.converter = OpenaiChatToolAdaptConverter(
            StandardToolAdapter(),
            ThinkTagExtractor(),
        )

    def test_response_extracts_reasoning(self):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "<think>reasoning here</think>"
                        "final answer"
                    ),
                },
                "finish_reason": "stop",
            }],
        }
        result = self.converter.convert_response(response)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "reasoning here"
        assert msg["content"] == "final answer"

    def test_response_no_thinking_passthrough(self):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "plain answer",
                },
                "finish_reason": "stop",
            }],
        }
        result = self.converter.convert_response(response)
        assert result is response

    def test_response_existing_reasoning_content_passthrough(self):
        """上游已有 reasoning_content 时直接透传"""
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "upstream reasoning",
                },
                "finish_reason": "stop",
            }],
        }
        result = self.converter.convert_response(response)
        assert result is response

    def test_response_thinking_with_tool_calls(self):
        """思考 + 工具调用同时存在（使用 DeepseekQwen 适配器）"""
        converter = OpenaiChatToolAdaptConverter(
            DeepseekQwenToolAdapter(),
            ThinkTagExtractor(),
        )
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "<think>let me check</think>"
                        'Result:\n<tool_call>\n'
                        '{"name": "search", "arguments": {"q": "test"}}\n'
                        '</tool_call>'
                    ),
                },
                "finish_reason": "stop",
            }],
        }
        result = converter.convert_response(response)
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "let me check"
        assert "Result:" in msg["content"]
        assert "<tool_call>" not in msg["content"]
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert result["choices"][0]["finish_reason"] == "tool_calls"


# ──────── Converter 集成测试：流式 ────────


class TestChatToolAdaptStreamWithThinking:

    def setup_method(self):
        self.converter = OpenaiChatToolAdaptConverter(
            StandardToolAdapter(),
            ThinkTagExtractor(),
        )

    @pytest.mark.asyncio
    async def test_stream_thinking_then_content(self):
        events = [
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": "<think>reasoning"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": " here</think>"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": "answer"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert '"reasoning_content"' in full
        assert '"content"' in full
        assert "<think>" not in full
        assert "</think>" not in full

        reasoning_parts = []
        content_parts = []
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk.removeprefix("data: ").strip())
            delta = data.get("choices", [{}])[0].get("delta", {})
            if "reasoning_content" in delta:
                reasoning_parts.append(delta["reasoning_content"])
            if "content" in delta and delta["content"]:
                content_parts.append(delta["content"])
        assert "".join(reasoning_parts).strip() != ""
        assert "answer" in "".join(content_parts)

    @pytest.mark.asyncio
    async def test_stream_no_thinking(self):
        """无思考标签时直接作为 content 输出"""
        events = [
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": "just an answer"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert '"reasoning_content"' not in full

        content_parts = []
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk.removeprefix("data: ").strip())
            delta = data.get("choices", [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                content_parts.append(delta["content"])
        assert "just an answer" in "".join(content_parts)

    @pytest.mark.asyncio
    async def test_stream_passthrough_upstream_reasoning_content(self):
        """上游已有 reasoning_content 字段时直接透传"""
        events = [
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{
                    "delta": {"role": "assistant"},
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{
                    "delta": {"reasoning_content": "upstream thinking"},
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": "answer"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert "upstream thinking" in full

    @pytest.mark.asyncio
    async def test_stream_thinking_with_tool_calls(self):
        """流式：思考 + 工具调用"""
        converter = OpenaiChatToolAdaptConverter(
            DeepseekQwenToolAdapter(),
            ThinkTagExtractor(),
        )
        events = [
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": "<think>planning</think>"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{
                    "delta": {
                        "content": (
                            '<tool_call>\n'
                            '{"name": "search", "arguments": {"q": "test"}}\n'
                            '</tool_call>'
                        ),
                    },
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(converter, events)
        full = "".join(chunks)
        assert '"reasoning_content"' in full
        assert '"tool_calls"' in full
        assert "<think>" not in full
        assert "<tool_call>" not in full

    @pytest.mark.asyncio
    async def test_stream_unclosed_thinking(self):
        """流式：思考标签未关闭时全部作为 reasoning_content"""
        events = [
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{
                    "delta": {"content": "<think>reasoning that never ends"},
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(self.converter, events)

        reasoning_parts = []
        content_parts = []
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk.removeprefix("data: ").strip())
            delta = data.get("choices", [{}])[0].get("delta", {})
            if "reasoning_content" in delta:
                reasoning_parts.append(delta["reasoning_content"])
            if "content" in delta and delta["content"]:
                content_parts.append(delta["content"])
        assert "reasoning that never ends" in "".join(reasoning_parts)


# ──────── Pipeline 集成测试 ────────


class TestGetThinkingExtractor:

    def test_disabled(self):
        from kllmgate.pipeline import get_thinking_extractor
        from kllmgate.models import ProviderConfig
        provider = ProviderConfig(
            name="test", base_url="http://localhost",
            protocol="openai", thinking_style="disabled",
        )
        ext = get_thinking_extractor(provider)
        assert isinstance(ext, NullThinkingExtractor)

    def test_think(self):
        from kllmgate.pipeline import get_thinking_extractor
        from kllmgate.models import ProviderConfig
        provider = ProviderConfig(
            name="test", base_url="http://localhost",
            protocol="openai", thinking_style="think",
        )
        ext = get_thinking_extractor(provider)
        assert isinstance(ext, ThinkTagExtractor)

    def test_lvl_entry(self):
        from kllmgate.pipeline import get_thinking_extractor
        from kllmgate.models import ProviderConfig
        provider = ProviderConfig(
            name="test", base_url="http://localhost",
            protocol="openai", thinking_style="lvl_entry",
        )
        ext = get_thinking_extractor(provider)
        assert isinstance(ext, LvlEntryThinkingExtractor)

    def test_unknown_raises(self):
        from kllmgate.pipeline import get_thinking_extractor
        from kllmgate.models import ProviderConfig
        from kllmgate.errors import ConfigError
        provider = ProviderConfig(
            name="test", base_url="http://localhost",
            protocol="openai", thinking_style="unknown_style",
        )
        with pytest.raises(ConfigError):
            get_thinking_extractor(provider)


# ──────── Anthropic 转换器集成测试：非流式 ────────


class TestAnthropicChatWithThinking:
    """AnthropicMessagesToOpenaiChatConverter + thinking/tool 集成"""

    def setup_method(self):
        self.converter = AnthropicMessagesToOpenaiChatConverter(
            DeepseekQwenToolAdapter(),
            ThinkTagExtractor(),
        )

    def test_response_thinking_and_tool_calls(self):
        response = {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "<think>planning steps</think>"
                        'Here is the result.\n'
                        '<tool_call>\n'
                        '{"name": "search", "arguments": {"q": "test"}}\n'
                        '</tool_call>'
                    ),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        result = self.converter.convert_response(response)
        assert result["type"] == "message"
        blocks = result["content"]
        types = [b["type"] for b in blocks]
        assert "thinking" in types
        assert "text" in types
        assert "tool_use" in types
        thinking_block = next(b for b in blocks if b["type"] == "thinking")
        assert thinking_block["thinking"] == "planning steps"
        tool_block = next(b for b in blocks if b["type"] == "tool_use")
        assert tool_block["name"] == "search"
        assert result["stop_reason"] == "tool_use"

    def test_response_thinking_only(self):
        response = {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<think>deep thought</think>final answer",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = self.converter.convert_response(response)
        blocks = result["content"]
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "deep thought"
        assert blocks[1]["type"] == "text"
        assert blocks[1]["text"] == "final answer"
        assert result["stop_reason"] == "end_turn"

    def test_response_tool_calls_from_text(self):
        """文本中的 tool call 被正确提取为 Anthropic tool_use"""
        converter = AnthropicMessagesToOpenaiChatConverter(
            DeepseekQwenToolAdapter(),
        )
        response = {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        'Let me search.\n'
                        '<tool_call>\n'
                        '{"name": "web_search", '
                        '"arguments": {"query": "python"}}\n'
                        '</tool_call>'
                    ),
                },
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        result = converter.convert_response(response)
        types = [b["type"] for b in result["content"]]
        assert "tool_use" in types
        assert result["stop_reason"] == "tool_use"

    def test_response_existing_structured_passthrough(self):
        """上游已有结构化 tool_calls 和 reasoning_content 时直接转换"""
        response = {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "upstream reasoning",
                    "tool_calls": [{
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Beijing"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        result = self.converter.convert_response(response)
        types = [b["type"] for b in result["content"]]
        assert "thinking" in types
        assert "tool_use" in types
        assert result["stop_reason"] == "tool_use"

    def test_response_plain_text_passthrough(self):
        converter = AnthropicMessagesToOpenaiChatConverter(
            StandardToolAdapter(),
        )
        response = {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "simple answer",
                },
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        result = converter.convert_response(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "simple answer"


# ──────── Anthropic 转换器集成测试：流式 ────────


class TestAnthropicChatStreamWithThinking:

    def setup_method(self):
        self.converter = AnthropicMessagesToOpenaiChatConverter(
            DeepseekQwenToolAdapter(),
            ThinkTagExtractor(),
        )

    @pytest.mark.asyncio
    async def test_stream_thinking_text_tool(self):
        events = [
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": "<think>plan</think>"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{
                    "delta": {
                        "content": (
                            'Result:\n<tool_call>\n'
                            '{"name": "search", '
                            '"arguments": {"q": "x"}}\n'
                            '</tool_call>'
                        ),
                    },
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(self.converter, events)
        full = "".join(chunks)
        assert "thinking" in full
        assert "text_delta" in full
        assert "tool_use" in full
        assert "tool_use" in full  # stop_reason
        assert "<think>" not in full
        assert "<tool_call>" not in full

    @pytest.mark.asyncio
    async def test_stream_no_thinking(self):
        converter = AnthropicMessagesToOpenaiChatConverter(
            StandardToolAdapter(),
        )
        events = [
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": "hello world"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(converter, events)
        full = "".join(chunks)
        assert "message_start" in full
        assert "text_delta" in full
        assert "hello world" in full
        assert "message_stop" in full

    @pytest.mark.asyncio
    async def test_stream_passthrough_structured_tool_calls(self):
        """上游已有结构化 tool_calls delta 时直接转换"""
        converter = AnthropicMessagesToOpenaiChatConverter(
            StandardToolAdapter(),
        )
        events = [
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"role": "assistant"}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"content": "Let me check."}}],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "tc1",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "BJ"}',
                            },
                        }],
                    },
                }],
            }), []),
            SseEvent(None, json.dumps({
                "id": "c1", "model": "m",
                "choices": [{
                    "delta": {},
                    "finish_reason": "tool_calls",
                }],
            }), []),
            SseEvent(None, "[DONE]", []),
        ]
        chunks = await _collect_stream(converter, events)
        full = "".join(chunks)
        assert "tool_use" in full
        assert "get_weather" in full


# ──────── AnthropicToResponses 非流式 ────────


class TestAnthropicResponsesWithThinking:

    def test_response_thinking_and_tool_calls(self):
        converter = AnthropicMessagesToOpenaiResponsesConverter(
            DeepseekQwenToolAdapter(),
            ThinkTagExtractor(),
        )
        upstream_resp = {
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
                        "<think>thinking</think>"
                        'Answer.\n<tool_call>\n'
                        '{"name": "func", "arguments": {"a": 1}}\n'
                        '</tool_call>'
                    ),
                }],
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
        }
        result = converter.convert_response(upstream_resp)
        assert result["type"] == "message"
        types = [b["type"] for b in result["content"]]
        assert "thinking" in types
        assert "text" in types
        assert "tool_use" in types
        assert result["stop_reason"] == "tool_use"
