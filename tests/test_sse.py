"""SSE 工具函数的单元测试"""

import json

import pytest

from kllmgate.sse import (
    SseEvent,
    format_sse,
    format_data_only_sse,
    format_named_sse,
    parse_sse_line,
    parse_sse_events,
)


class TestSseEvent:

    def test_dataclass_fields(self):
        ev = SseEvent(event="message_start", data='{"type": "message"}', raw_lines=[])
        assert ev.event == "message_start"
        assert ev.data == '{"type": "message"}'
        assert ev.raw_lines == []

    def test_data_only_event(self):
        ev = SseEvent(event=None, data='{"id": "1"}', raw_lines=["data: {\"id\": \"1\"}"])
        assert ev.event is None


class TestFormatSse:

    def test_data_only(self):
        result = format_sse('{"key": "value"}')
        assert result == 'data: {"key": "value"}\n\n'

    def test_with_event(self):
        result = format_sse('{"type": "done"}', event="response.completed")
        assert result == 'event: response.completed\ndata: {"type": "done"}\n\n'

    def test_multiline_data(self):
        result = format_sse("line1\nline2")
        assert result == "data: line1\ndata: line2\n\n"

    def test_done_marker(self):
        result = format_sse("[DONE]")
        assert result == "data: [DONE]\n\n"


class TestFormatDataOnlySse:

    def test_basic(self):
        data = {"choices": [{"delta": {"content": "hi"}}]}
        result = format_data_only_sse(data)
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        parsed = json.loads(result[6:].strip())
        assert parsed == data

    def test_empty_dict(self):
        result = format_data_only_sse({})
        assert result == "data: {}\n\n"


class TestFormatNamedSse:

    def test_basic(self):
        data = {"type": "response.created", "response": {"id": "resp_1"}}
        result = format_named_sse("response.created", data)
        lines = result.strip().split("\n")
        assert lines[0] == "event: response.created"
        assert lines[1].startswith("data: ")
        parsed = json.loads(lines[1][6:])
        assert parsed["type"] == "response.created"

    def test_ends_with_double_newline(self):
        result = format_named_sse("test", {"a": 1})
        assert result.endswith("\n\n")


class TestParseSseLine:

    def test_data_line(self):
        field, value = parse_sse_line('data: {"id": "1"}')
        assert field == "data"
        assert value == '{"id": "1"}'

    def test_event_line(self):
        field, value = parse_sse_line("event: message_start")
        assert field == "event"
        assert value == "message_start"

    def test_empty_line(self):
        field, value = parse_sse_line("")
        assert field is None
        assert value is None

    def test_comment_line(self):
        field, value = parse_sse_line(": this is a comment")
        assert field is None
        assert value is None

    def test_data_no_space_after_colon(self):
        field, value = parse_sse_line("data:hello")
        assert field == "data"
        assert value == "hello"

    def test_whitespace_only_line(self):
        field, value = parse_sse_line("   ")
        assert field is None
        assert value is None

    def test_data_done(self):
        field, value = parse_sse_line("data: [DONE]")
        assert field == "data"
        assert value == "[DONE]"


class TestParseSseEvents:

    @pytest.mark.asyncio
    async def test_single_data_event(self):
        async def lines():
            yield 'data: {"id": "1"}'
            yield ""

        events = []
        async for ev in parse_sse_events(lines()):
            events.append(ev)
        assert len(events) == 1
        assert events[0].data == '{"id": "1"}'
        assert events[0].event is None

    @pytest.mark.asyncio
    async def test_named_event(self):
        async def lines():
            yield "event: message_start"
            yield 'data: {"type": "message_start"}'
            yield ""

        events = []
        async for ev in parse_sse_events(lines()):
            events.append(ev)
        assert len(events) == 1
        assert events[0].event == "message_start"
        assert '"message_start"' in events[0].data

    @pytest.mark.asyncio
    async def test_multiple_events(self):
        async def lines():
            yield 'data: {"chunk": 1}'
            yield ""
            yield 'data: {"chunk": 2}'
            yield ""
            yield "data: [DONE]"
            yield ""

        events = []
        async for ev in parse_sse_events(lines()):
            events.append(ev)
        assert len(events) == 3
        assert events[2].data == "[DONE]"

    @pytest.mark.asyncio
    async def test_multiline_data(self):
        async def lines():
            yield "data: line1"
            yield "data: line2"
            yield ""

        events = []
        async for ev in parse_sse_events(lines()):
            events.append(ev)
        assert len(events) == 1
        assert events[0].data == "line1\nline2"

    @pytest.mark.asyncio
    async def test_comment_lines_preserved_in_raw(self):
        async def lines():
            yield ": keepalive"
            yield 'data: {"ok": true}'
            yield ""

        events = []
        async for ev in parse_sse_events(lines()):
            events.append(ev)
        assert len(events) == 1
        assert ": keepalive" in events[0].raw_lines

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        async def lines():
            return
            yield

        events = []
        async for ev in parse_sse_events(lines()):
            events.append(ev)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_trailing_event_without_blank_line(self):
        """流结束时即使没有空行，也应产出最后一个事件"""
        async def lines():
            yield 'data: {"final": true}'

        events = []
        async for ev in parse_sse_events(lines()):
            events.append(ev)
        assert len(events) == 1
        assert events[0].data == '{"final": true}'
