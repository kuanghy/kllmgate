import os
import uuid
import time
import json
import re
import asyncio
import logging

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
TARGET_BASE = "https://api.scnet.cn/api/llm/v1"
AUTH_TOKEN = f"Bearer {os.getenv('SCNET_API_KEY')}"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _make_resp_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def _make_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


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
    """尝试将参数值解析为合适的 Python 类型（数组、对象、数字、布尔），
    解析失败则保留为字符串
    """
    value = value.strip().strip("\n")
    if not value:
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _parse_minimax_tool_calls(text: str) -> tuple[str, list[dict]]:
    """从文本中提取 MiniMax XML 格式的工具调用，返回 (清理后文本, function_call列表)"""
    if "<minimax:tool_call>" not in text:
        return text, []

    calls = []
    for block in _MINIMAX_TC_RE.findall(text):
        for invoke_body in _INVOKE_RE.findall(block):
            name_match = re.search(r"^([^>]+)", invoke_body)
            if not name_match:
                continue
            func_name = _strip_quotes(name_match.group(1))

            params = {}
            for param_body in _PARAM_RE.findall(invoke_body):
                kv = re.search(r"^([^>]+)>(.*)", param_body, re.DOTALL)
                if kv:
                    k = _strip_quotes(kv.group(1))
                    v = _coerce_param_value(kv.group(2))
                    params[k] = v

            fc_id = f"fc_{uuid.uuid4().hex[:24]}"
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            calls.append({
                "type": "function_call",
                "id": fc_id,
                "call_id": call_id,
                "name": func_name,
                "arguments": json.dumps(params, ensure_ascii=False),
                "status": "completed",
            })

    clean_text = _MINIMAX_TC_RE.sub("", text).strip()
    return clean_text, calls


_TC_FRAGMENT_RE = re.compile(
    r"</minimax:tool_call>|</invoke>|<parameter\s+name="
    r"|<invoke\s+name=|<minimax:tool_call>"
)


def _detect_malformed_tool_call(
    raw_text: str, parsed_calls: list[dict],
) -> str | None:
    """检测模型输出中格式错误的工具调用

    Returns:
        错误描述字符串（检测到问题时），或 None（正常时）
    """
    fragments = _TC_FRAGMENT_RE.findall(raw_text)
    if not fragments:
        return None

    if not parsed_calls and len(fragments) >= 2:
        return (
            "Malformed tool call: found XML fragments but no valid "
            "calls parsed. Likely caused by context degradation."
        )

    open_tc = raw_text.count("<minimax:tool_call>")
    close_tc = raw_text.count("</minimax:tool_call>")
    if open_tc != close_tc:
        return (
            f"Unbalanced tool call tags: "
            f"{open_tc} open vs {close_tc} close"
        )

    for call in parsed_calls:
        name = call.get("name", "")
        if not name:
            return "Tool call has empty function name"
        if len(name) > 200 or re.search(r"[\s<>]", name):
            return f"Tool call has invalid function name: {name!r}"
        args_str = call.get("arguments", "{}")
        if args_str:
            try:
                parsed = json.loads(args_str)
                if not isinstance(parsed, dict):
                    return (
                        f"Tool call arguments is "
                        f"{type(parsed).__name__}, expected object"
                    )
            except (json.JSONDecodeError, ValueError) as e:
                return f"Tool call arguments is invalid JSON: {e}"

    return None


def _strip_tc_fragments(text: str) -> str:
    """移除文本中残留的工具调用 XML 片段"""
    text = re.sub(r"</?minimax:tool_call[^>]*>", "", text)
    text = re.sub(r"<invoke\s+name=[^>]*>", "", text)
    text = re.sub(r"</invoke>", "", text)
    text = re.sub(
        r"<parameter\s+name=[^>]*>[^<]*</parameter>", "", text,
    )
    text = re.sub(r"</?parameter[^>]*>", "", text)
    return text.strip()


def _format_tools_prompt(tools: list[dict]) -> str:
    """将 Responses API tools 定义转为 MiniMax 的 system prompt 格式"""
    tool_tags = []
    for t in tools:
        func = t.get("function", t)
        tool_def = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        }
        tool_tags.append(f"<tool>{json.dumps(tool_def, ensure_ascii=False)}</tool>")
    return (
        "# Tools\n"
        "You may call one or more tools to assist with the user query.\n"
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


def _convert_content_part(part: dict) -> str:
    """将单个 content part 转为纯文本表示，支持多种内容类型"""
    part_type = part.get("type", "")

    if part_type in ("text", "input_text", "output_text"):
        return part.get("text", "")

    if part_type == "image_url":
        url = part.get("image_url", {})
        if isinstance(url, str):
            return f"[image: {url}]"
        return f'[image: {url.get("url", "unknown")} detail={url.get("detail", "auto")}]'

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
        label = filename or file_id or "unknown"
        return f"[file: {label}]"

    if part_type == "refusal":
        return part.get("refusal", "[refusal]")

    logger.warning("Unrecognized content part type %r, converting to string", part_type)
    return str(part)


def _convert_content_list(content: list) -> str:
    """将 content 列表转为纯文本，保留所有类型的语义信息"""
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(_convert_content_part(item))
    return "\n".join(parts)


def _parse_input(body: dict) -> list[dict]:
    """将 Responses API 的 input 转换为 Chat Completions messages"""
    system_contents = []
    if body.get("instructions"):
        system_contents.append(body["instructions"])

    if tools := body.get("tools"):
        system_contents.append(_format_tools_prompt(tools))

    raw_messages = []
    pending_tool_calls: list[dict] = []
    pending_tool_results: list[dict] = []
    call_id_to_name: dict[str, str] = {}

    for item in body.get("input", []):
        if isinstance(item, str):
            _flush_tool_messages(raw_messages, pending_tool_calls, pending_tool_results)
            raw_messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "")

        if item_type == "function_call":
            call_id = item.get("call_id", item.get("id", ""))
            func_name = item.get("name", "")
            call_id_to_name[call_id] = func_name
            pending_tool_calls.append({
                "name": func_name,
                "arguments": item.get("arguments", ""),
            })
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id", "")
            pending_tool_results.append({
                "name": call_id_to_name.get(call_id, "unknown"),
                "output": item.get("output", ""),
            })
            continue

        _flush_tool_messages(raw_messages, pending_tool_calls, pending_tool_results)

        role = item.get("role", "user")
        if role == "model":
            role = "assistant"
        elif role == "developer":
            role = "system"

        content = item.get("content", "")
        if isinstance(content, list):
            content = _convert_content_list(content)

        if role == "system":
            system_contents.append(content)
        else:
            raw_messages.append({"role": role, "content": content})

    _flush_tool_messages(raw_messages, pending_tool_calls, pending_tool_results)

    messages = []
    if system_contents:
        messages.append({
            "role": "system",
            "content": "\n\n".join(system_contents),
        })

    for msg in raw_messages:
        if not messages or messages[-1]["role"] != msg["role"]:
            messages.append(msg)
        else:
            messages[-1]["content"] += "\n\n" + msg["content"]

    return messages


def _flush_tool_messages(
    raw_messages: list[dict],
    pending_tool_calls: list[dict],
    pending_tool_results: list[dict],
):
    """将累积的 function_call / function_call_output 还原为 MiniMax 原生消息格式"""
    if pending_tool_calls:
        xml_lines = ["<minimax:tool_call>"]
        for tc in pending_tool_calls:
            xml_lines.append(f'<invoke name="{tc["name"]}">')
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Failed to parse tool arguments for %r, "
                    "using raw string as fallback: %.200s",
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
        raw_messages.append({
            "role": "assistant",
            "content": "\n".join(xml_lines),
        })
        pending_tool_calls.clear()

    for result in pending_tool_results:
        raw_messages.append({
            "role": "user",
            "content": (
                f'[Tool Result] {result["name"]}:\n{result["output"]}'
            ),
        })
    pending_tool_results.clear()


@app.post("/responses")
async def responses_to_chat(request: Request):
    body = await request.json()
    messages = _parse_input(body)
    chat_payload = {
        "model": body.get("model", "MiniMax-M2.5"),
        "messages": messages,
        "stream": body.get("stream", False),
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": AUTH_TOKEN,
    }

    if chat_payload.get("stream"):
        chat_payload["stream_options"] = {"include_usage": True}
        return StreamingResponse(
            _stream_responses_api(chat_payload, headers),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await _non_stream_request(chat_payload, headers)


async def _non_stream_request(chat_payload: dict, headers: dict):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{TARGET_BASE}/chat/completions",
                json=chat_payload,
                headers=headers,
                timeout=120.0,
            )
            resp.raise_for_status()
            try:
                chat_resp = resp.json()
            except Exception:
                raw_text = resp.text
                logger.error("Failed to parse upstream JSON: %s", raw_text)
                chat_resp = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "created": int(time.time()),
                    "model": chat_payload["model"],
                    "choices": [
                        {"message": {"role": "assistant", "content": raw_text}}
                    ],
                }
        except httpx.HTTPStatusError as e:
            logger.error(
                "Upstream HTTP %s: %s",
                e.response.status_code, e.response.text,
            )
            return JSONResponse(
                status_code=e.response.status_code,
                content={"error": {"message": e.response.text}},
            )
        except Exception as e:
            logger.exception("Request error")
            return JSONResponse(
                status_code=502,
                content=_error_response_body(chat_payload["model"], str(e)),
            )

    if "choices" not in chat_resp or not chat_resp["choices"]:
        return chat_resp

    choice = chat_resp["choices"][0]
    message = choice.get("message", {})
    raw_content = message.get("content", "")
    clean_text, minimax_calls = _parse_minimax_tool_calls(raw_content)

    malformed_reason = _detect_malformed_tool_call(raw_content, minimax_calls)
    if malformed_reason:
        logger.warning(
            "Malformed tool call detected: %s | raw=%.500s",
            malformed_reason, raw_content,
        )
        clean_text = _strip_tc_fragments(clean_text)
        usage = chat_resp.get("usage", {})
        _log_usage(usage)
        resp_usage = _convert_usage(usage)
        output = []
        if clean_text:
            output.append({
                "type": "message",
                "id": _make_msg_id(),
                "status": "incomplete",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": clean_text}
                ],
            })
        return {
            "id": chat_resp.get("id", _make_resp_id()),
            "object": "response",
            "created_at": chat_resp.get("created", int(time.time())),
            "status": "incomplete",
            "model": chat_resp.get("model"),
            "output": output,
            "usage": resp_usage,
            "error": {
                "type": "server_error",
                "code": "malformed_tool_call",
                "message": malformed_reason,
            },
        }

    output = []
    if clean_text:
        output.append({
            "type": "message",
            "id": _make_msg_id(),
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": clean_text}
            ],
        })
    output.extend(minimax_calls)
    for tc in message.get("tool_calls", []):
        output.append({
            "type": "function_call",
            "id": tc.get("id", _make_msg_id()),
            "call_id": tc.get("id", ""),
            "name": tc.get("function", {}).get("name", ""),
            "arguments": tc.get("function", {}).get("arguments", ""),
            "status": "completed",
        })
    if not output:
        output.append({
            "type": "message",
            "id": _make_msg_id(),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": ""}],
        })

    usage = chat_resp.get("usage", {})
    _log_usage(usage)

    return {
        "id": chat_resp.get("id", _make_resp_id()),
        "object": "response",
        "created_at": chat_resp.get("created", int(time.time())),
        "status": "completed",
        "model": chat_resp.get("model"),
        "output": output,
        "usage": _convert_usage(usage),
    }


async def _stream_responses_api(chat_payload: dict, headers: dict):
    """将上游 Chat Completions SSE 转换为 Responses API 流式事件"""
    resp_id = _make_resp_id()
    msg_id = _make_msg_id()
    model_name = chat_payload["model"]
    created_ts = int(time.time())

    full_text = ""
    sent_pos = 0
    tc_detected = False
    seq = 0
    output_index = 0

    error_msg = ""
    upstream_completed = False
    usage = {}

    def sse(event: str, data: dict) -> str:
        nonlocal seq
        data["sequence_number"] = seq
        seq += 1
        return _sse(event, data)

    response_obj = {
        "id": resp_id,
        "object": "response",
        "created_at": created_ts,
        "status": "in_progress",
        "model": model_name,
        "output": [],
    }

    yield sse("response.created", {
        "type": "response.created",
        "response": response_obj,
    })

    yield sse("response.in_progress", {
        "type": "response.in_progress",
        "response": response_obj,
    })

    msg_item = {
        "type": "message",
        "id": msg_id,
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    yield sse("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": msg_item,
    })

    yield sse("response.content_part.added", {
        "type": "response.content_part.added",
        "output_index": output_index,
        "content_index": 0,
        "part": {"type": "output_text", "text": ""},
    })

    max_retries = 5
    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    try:
        for attempt in range(max_retries):
            try:
                async with client.stream(
                    "POST",
                    f"{TARGET_BASE}/chat/completions",
                    json=chat_payload,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        error_msg = (
                            f"HTTP {resp.status_code}:"
                            f" {body.decode()}"
                        )
                        logger.error("Upstream returned %s", error_msg)
                        break

                    async for line in resp.aiter_lines():
                        if (not line.strip()
                                or not line.startswith("data: ")):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            upstream_completed = True
                            break
                        try:
                            chat_chunk = json.loads(data_str)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "Upstream chunk parse"
                                " error: %.200s -> %s",
                                data_str, e,
                            )
                            continue

                        if chunk_usage := chat_chunk.get("usage"):
                            usage = chunk_usage
                        choices = chat_chunk.get("choices", [])
                        if not choices:
                            continue
                        delta_text = (
                            choices[0].get("delta", {})
                            .get("content", "")
                        )
                        if not delta_text:
                            continue

                        full_text += delta_text
                        if tc_detected:
                            continue

                        tc_tag = "<minimax:tool_call>"
                        tc_pos = full_text.find(tc_tag)
                        if tc_pos >= 0:
                            tc_detected = True
                            safe_end = tc_pos
                        elif "</minimax:tool_call>" in full_text:
                            tc_detected = True
                            safe_end = sent_pos
                        else:
                            safe_end = max(
                                sent_pos,
                                len(full_text) - len(tc_tag),
                            )
                        unsent = full_text[sent_pos:safe_end]
                        if unsent:
                            yield sse(
                                "response.output_text.delta",
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": unsent,
                                },
                            )
                            sent_pos = safe_end
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                if full_text:
                    error_msg = f"Connection lost after partial data: {e}"
                    logger.warning(
                        "Stream interrupted after receiving %d chars: %s",
                        len(full_text), e,
                    )
                    break
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt - 2) if attempt >= 2 else 0
                    logger.warning(
                        "Connect error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, max_retries, wait, e,
                    )
                    if wait:
                        await asyncio.sleep(wait)
                else:
                    error_msg = f"Connect failed after {max_retries} attempts: {e}"
                    logger.error(error_msg)
    except httpx.ReadTimeout as e:
        error_msg = str(e)
        logger.warning("Upstream read timeout: %s", e)
    except Exception as e:
        error_msg = str(e)
        logger.exception("Upstream streaming error")
    finally:
        await client.aclose()

    stream_ok = upstream_completed and not error_msg
    partial_failure = bool(error_msg) and bool(full_text)
    total_failure = bool(error_msg) and not full_text

    clean_text, minimax_calls = _parse_minimax_tool_calls(full_text)

    malformed_reason = _detect_malformed_tool_call(full_text, minimax_calls)
    if malformed_reason:
        logger.warning(
            "Malformed tool call in stream: %s | len=%d",
            malformed_reason, len(full_text),
        )
        minimax_calls = []
        clean_text = _strip_tc_fragments(clean_text)

    if total_failure:
        response_obj.update({
            "status": "failed",
            "error": {
                "type": "server_error",
                "code": "upstream_error",
                "message": error_msg,
            },
        })
        yield sse("response.completed", {
            "type": "response.completed",
            "response": response_obj,
        })
        return

    remaining = "" if malformed_reason else clean_text[sent_pos:]
    if remaining:
        yield sse("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "delta": remaining,
        })

    if malformed_reason or not stream_ok:
        msg_status = "incomplete"
    else:
        msg_status = "completed"
    completed_msg = {
        "type": "message",
        "id": msg_id,
        "status": msg_status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": clean_text}],
    }

    yield sse("response.output_text.done", {
        "type": "response.output_text.done",
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "text": clean_text,
    })

    yield sse("response.content_part.done", {
        "type": "response.content_part.done",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": clean_text},
    })

    yield sse("response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": completed_msg,
    })

    output_items = [completed_msg]
    output_index = 1

    for fc_item in minimax_calls:
        pending_item = {**fc_item, "status": "in_progress", "arguments": ""}
        yield sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": pending_item,
        })

        yield sse("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta",
            "item_id": fc_item["id"],
            "output_index": output_index,
            "delta": fc_item["arguments"],
        })

        yield sse("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": fc_item["id"],
            "output_index": output_index,
            "name": fc_item["name"],
            "arguments": fc_item["arguments"],
        })

        yield sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": fc_item,
        })
        output_items.append(fc_item)
        output_index += 1

    _log_usage(usage)
    resp_usage = _convert_usage(usage)

    if malformed_reason:
        response_obj.update({
            "status": "incomplete",
            "output": output_items,
            "usage": resp_usage,
            "error": {
                "type": "server_error",
                "code": "malformed_tool_call",
                "message": malformed_reason,
            },
        })
    elif partial_failure:
        response_obj.update({
            "status": "incomplete",
            "output": output_items,
            "usage": resp_usage,
            "error": {
                "type": "server_error",
                "code": "stream_interrupted",
                "message": error_msg,
            },
        })
    else:
        response_obj.update({
            "status": "completed",
            "output": output_items,
            "usage": resp_usage,
        })
    yield sse("response.completed", {
        "type": "response.completed",
        "response": response_obj,
    })


def _convert_usage(usage: dict) -> dict:
    """将 Chat Completions 的 usage 字段转换为 Responses API 格式"""
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


def _log_usage(usage: dict) -> None:
    if usage:
        logger.info(
            "Token usage: prompt=%s completion=%s total=%s",
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
            usage.get("total_tokens", "?"),
        )


def _error_response_body(model: str, error_msg: str) -> dict:
    return {
        "id": _make_resp_id(),
        "object": "response",
        "created_at": int(time.time()),
        "status": "failed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": _make_msg_id(),
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": f"Proxy Error: {error_msg}"}
                ],
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500)
