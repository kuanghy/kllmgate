"""CLI 入口"""

import argparse
import logging

import uvicorn

from . import log
from .app import create_app
from .config import load_config, resolve_config


def main():
    parser = argparse.ArgumentParser(
        description="kllmgate - 通用 LLM API 协议转换网关",
    )
    parser.add_argument(
        "--config", default=None,
        help="配置文件路径（未指定时自动搜索标准路径）",
    )
    parser.add_argument(
        "--host", default=None,
        help="监听地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="监听端口（默认: 8500）",
    )
    parser.add_argument(
        "--log-level", default=None,
        choices=["debug", "info", "warning", "error"],
        help="日志级别（默认: info）",
    )
    args = parser.parse_args()

    config_path = resolve_config(args.config)
    config = load_config(config_path)

    # CLI 参数 > 配置文件 > 硬编码默认值
    log_level = args.log_level or config.server.log_level
    host = args.host or config.server.host
    port = args.port or config.server.port

    log.setup(log_level)

    logger = logging.getLogger(__package__)
    logger.info("Using config: %s", config_path)

    app = create_app(config)
    uvicorn.run(
        app, host=host, port=port,
        log_config=log.UVICORN_LOG_CONFIG, log_level=log_level,
    )


if __name__ == "__main__":
    main()
