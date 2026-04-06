# kllmgate

[![GitHub](https://img.shields.io/badge/GitHub-kllmgate-blue)](https://github.com/kuanghy/kllmgate)

通用 LLM API 协议转换网关，在不同模型厂商的 API 协议之间进行透明转换，让客户端无需适配即可调用不同厂商的模型。

## 当前能力

- `openai.chat` ↔ `openai.responses`
- `openai.chat` ↔ `anthropic.messages`
- `openai.responses` ↔ `anthropic.messages`
- 同协议直通与同协议工具适配
- 流式 / 非流式请求处理
- OpenAI Standard / MiniMax XML / Anthropic 三种工具调用格式适配

## 项目结构

项目已按现代 Python 包结构重构：

```text
kllmgate/
├── src/
│   └── kllmgate/
│       ├── __main__.py
│       ├── app.py
│       ├── config.py
│       ├── pipeline.py
│       ├── converters/
│       ├── toolcall/
│       └── upstream/
└── tests/
```

## 快速开始

### 1. 安装

```bash
pip install -e .
```

### 2. 编写配置

创建 `config.toml`：

```toml
# 服务配置（可选，CLI 参数可覆盖）
[server]
# host = "0.0.0.0"
# port = 8500
# log_level = "info"
# default_provider = "openai_official"

[providers.openai_official]
base_url = "https://api.openai.com/v1"
env_key = "OPENAI_API_KEY"
protocol = "openai"
wire_api = "chat"

[providers.anthropic_official]
base_url = "https://api.anthropic.com"
env_key = "ANTHROPIC_API_KEY"
protocol = "anthropic"
```

### 3. 设置环境变量并启动

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
python -m kllmgate --config config.toml
```

也可以使用安装后的命令行入口：

```bash
kllmgate --config config.toml
```

默认监听 `0.0.0.0:8500`。

## API 路由

- `POST /openai/chat/completions`
- `POST /openai/responses`
- `POST /anthropic/v1/messages`

客户端指定模型提供商有四种方式（按优先级从高到低）：

**方式一：`provider/model` 格式（推荐）**

```json
{
  "model": "openai_official/gpt-4.1"
}
```

**方式二：配置模型别名**

在 `config.toml` 中添加：

```toml
[model_aliases]
"MiniMax-M2.5" = "minimax_proxy/MiniMax-M2.5"
```

客户端直接使用裸模型名：

```json
{
  "model": "MiniMax-M2.5"
}
```

**方式三：`X-KLLMGate-Provider` 请求头**

适用于不支持 `provider/model` 格式的客户端（如 Codex）：

```bash
curl -H "X-KLLMGate-Provider: minimax_proxy" \
     -d '{"model": "MiniMax-M2.5", ...}' \
     http://localhost:8500/openai/chat/completions
```

该 header 仅在网关层消费，不会透传到上游。

**方式四：配置默认 Provider**

在 `[server]` 段设置 `default_provider` 作为兜底路由，客户端无需指定 provider：

```toml
[server]
default_provider = "openai_official"
```

```json
{
  "model": "gpt-4.1"
}
```
