"""CLI 入口"""

import argparse
import logging
import os
import sys

import uvicorn

_DEFAULT_CONFIG = os.environ.get("KLLMGATE_CONFIG", "config.toml")


def main():
    parser = argparse.ArgumentParser(
        description="kllmgate - 通用 LLM API 协议转换网关",
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG,
        help=(
            "配置文件路径"
            "（默认: $KLLMGATE_CONFIG 或 config.toml）"
        ),
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

    from .app import create_app

    app = create_app(config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
