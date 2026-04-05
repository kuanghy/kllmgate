"""CLI 入口"""

import argparse
import logging
import os
from pathlib import Path

import uvicorn

_SEARCH_PATHS = [
    Path("/etc/kllmgate/config.toml"),
    Path("/usr/local/etc/kllmgate/config.toml"),
    Path.home() / ".local/etc/kllmgate/config.toml",
    Path.home() / ".config/kllmgate/config.toml",
    Path("config.toml"),
]


def _resolve_config(cli_arg: str | None) -> str:
    """按优先级从低到高查找配置文件

    优先级：
    1. 系统路径（/etc → /usr/local/etc → ~/.local/etc → ~/.config）
    2. 当前工作目录 config.toml
    3. KLLMGATE_CONFIG 环境变量
    4. --config 命令行参数（最高）

    高优先级覆盖低优先级；如果都不存在，返回 "config.toml" 让
    config.py 在加载时报出明确错误。
    """
    if cli_arg is not None:
        return cli_arg

    env_path = os.environ.get("KLLMGATE_CONFIG")
    if env_path:
        return env_path

    found: str | None = None
    for path in _SEARCH_PATHS:
        if path.is_file():
            found = str(path)

    return found or "config.toml"


def main():
    parser = argparse.ArgumentParser(
        description="kllmgate - 通用 LLM API 协议转换网关",
    )
    parser.add_argument(
        "--config", default=None,
        help="配置文件路径（未指定时自动搜索标准路径）",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="监听地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--port", type=int, default=8500,
        help="监听端口（默认: 8500）",
    )
    parser.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
        help="日志级别（默认: info）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config_path = _resolve_config(args.config)
    logger = logging.getLogger(__name__)
    logger.info("Using config: %s", config_path)

    from .app import create_app

    app = create_app(config_path=config_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
