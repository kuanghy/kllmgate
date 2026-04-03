# kllmgate

[![GitHub](https://img.shields.io/badge/GitHub-kllmgate-blue)](https://github.com/kuanghy/kllmgate)

通用 LLM API 协议转换网关，在不同模型厂商的 API 协议之间进行透明转换，让客户端无需适配即可调用不同厂商的模型。

## 功能概述

- **Responses API → Chat Completions**：将 OpenAI Responses API 请求转换为 Chat Completions 格式，再将响应转换回 Responses API 格式
- **MiniMax 工具调用兼容**：自动处理 MiniMax 模型的 XML 风格工具调用（`<minimax:tool_call>`），将其转换为 OpenAI 标准的 `function_call` 格式
- **流式 / 非流式支持**：完整支持 SSE 流式传输和普通 JSON 响应两种模式
- **多内容类型处理**：支持文本、图片、音频、文件等多种 content part 的转换

## 当前状态

目前已实现 MiniMax 模型与 OpenAI Codex 的兼容层，使 Codex 可以通过 Responses API 调用 MiniMax 模型（如 MiniMax-M2.5），包括工具调用的完整闭环。

## 快速开始

**安装依赖：**

```bash
pip install fastapi uvicorn httpx
```

**设置 API Key 并启动：**

```bash
export SCNET_API_KEY="your-api-key"
python kllmgate.py
```

服务默认监听 `0.0.0.0:8500`，将 Responses API 请求发送到 `http://localhost:8500/responses` 即可。

## 未来规划

- [ ] 支持 OpenAI ↔ Anthropic 协议互转
- [ ] 支持 Chat Completions ↔ Responses API 双向转换
- [ ] 统一不同厂商的工具调用协议（MiniMax / OpenAI / Anthropic）
- [ ] 多模型后端路由
- [ ] 后端模型负载均衡
- [ ] 项目结构重构为可扩展的模块化架构
