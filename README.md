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
├── kllmgate/
│   ├── __main__.py
│   ├── app.py
│   ├── config.py
│   ├── pipeline.py
│   ├── converters/
│   ├── tools/
│   └── upstream/
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

客户端请求中的 `model` 必须使用 `provider/model` 形式，例如：

```json
{
  "model": "openai_official/gpt-4.1"
}
```

## 测试

```bash
python -m pytest tests -v --tb=short --color=no
```

当前测试状态：`184 passed`

## 说明

根目录旧文件 `kllmgate.py` 目前仅作为历史实现保留，不再是推荐入口。当前应使用包入口 `python -m kllmgate` 或命令 `kllmgate` 启动。
