"""配置加载与校验的单元测试"""

import os
import textwrap

import pytest

from kllmgate.config import load_config
from kllmgate.errors import ConfigError


@pytest.fixture
def toml_file(tmp_path):
    """创建临时 TOML 配置文件的工厂 fixture"""
    def _write(content: str) -> str:
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent(content))
        return str(path)
    return _write


class TestLoadConfig:

    def test_basic_openai_provider(self, toml_file):
        path = toml_file("""\
            [providers.openai_official]
            base_url = "https://api.openai.com/v1"
            api_key = "sk-test"
            protocol = "openai"
            wire_api = "chat"
        """)
        config = load_config(path)
        assert config.model_aliases == {}
        assert len(config.providers) == 1
        cfg = config.providers["openai_official"]
        assert cfg.name == "openai_official"
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.api_key == "sk-test"
        assert cfg.protocol == "openai"
        assert cfg.wire_api == "chat"

    def test_anthropic_provider(self, toml_file):
        path = toml_file("""\
            [providers.anthropic_official]
            base_url = "https://api.anthropic.com"
            api_key = "sk-ant-test"
            protocol = "anthropic"
        """)
        config = load_config(path)
        cfg = config.providers["anthropic_official"]
        assert cfg.protocol == "anthropic"
        assert cfg.wire_api == "messages"

    def test_multiple_providers(self, toml_file):
        path = toml_file("""\
            [providers.openai]
            base_url = "https://api.openai.com/v1"
            api_key = "sk-1"
            protocol = "openai"

            [providers.anthropic]
            base_url = "https://api.anthropic.com"
            api_key = "sk-2"
            protocol = "anthropic"
        """)
        config = load_config(path)
        assert len(config.providers) == 2
        assert "openai" in config.providers
        assert "anthropic" in config.providers

    def test_default_values(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        config = load_config(path)
        cfg = config.providers["test"]
        assert cfg.wire_api == "chat"
        assert cfg.tool_style == "standard"
        assert cfg.timeout_seconds == 120
        assert cfg.max_retries == 2
        assert cfg.models is None

    def test_custom_values(self, toml_file):
        path = toml_file("""\
            [providers.minimax]
            base_url = "https://api.example.com/v1"
            api_key = "sk-test"
            protocol = "openai"
            wire_api = "chat"
            tool_style = "minimax_xml"
            timeout_seconds = 60
            max_retries = 5
            models = ["model-a", "model-b"]
        """)
        config = load_config(path)
        cfg = config.providers["minimax"]
        assert cfg.tool_style == "minimax_xml"
        assert cfg.timeout_seconds == 60
        assert cfg.max_retries == 5
        assert cfg.models == ["model-a", "model-b"]

    def test_base_url_trailing_slash_stripped(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com/v1/"
            api_key = "sk-test"
            protocol = "openai"
        """)
        config = load_config(path)
        assert config.providers["test"].base_url == "https://api.example.com/v1"

    def test_env_key_config(self, toml_file, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "sk-from-env")
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            env_key = "MY_API_KEY"
            protocol = "openai"
        """)
        config = load_config(path)
        cfg = config.providers["test"]
        assert cfg.env_key == "MY_API_KEY"
        assert cfg.resolve_api_key() == "sk-from-env"


class TestLoadConfigValidation:

    def test_invalid_protocol_raises(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "invalid_proto"
        """)
        with pytest.raises(ConfigError, match="protocol"):
            load_config(path)

    def test_anthropic_with_chat_wire_api_raises(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "anthropic"
            wire_api = "chat"
        """)
        with pytest.raises(ConfigError, match="wire_api"):
            load_config(path)

    def test_anthropic_with_responses_wire_api_raises(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "anthropic"
            wire_api = "responses"
        """)
        with pytest.raises(ConfigError, match="wire_api"):
            load_config(path)

    def test_missing_base_url_raises(self, toml_file):
        path = toml_file("""\
            [providers.test]
            api_key = "sk-test"
            protocol = "openai"
        """)
        with pytest.raises(ConfigError, match="base_url"):
            load_config(path)

    def test_missing_protocol_raises(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
        """)
        with pytest.raises(ConfigError, match="protocol"):
            load_config(path)

    def test_no_api_key_and_no_env_key_raises(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            protocol = "openai"
        """)
        with pytest.raises(ConfigError, match="API key"):
            load_config(path)

    def test_empty_providers_raises(self, toml_file):
        path = toml_file("""\
            [providers]
        """)
        with pytest.raises(ConfigError, match="provider"):
            load_config(path)

    def test_missing_providers_section_raises(self, toml_file):
        path = toml_file("""\
            [server]
            host = "0.0.0.0"
        """)
        with pytest.raises(ConfigError, match="providers"):
            load_config(path)

    def test_file_not_found_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/config.toml")

    def test_env_key_missing_in_environment_raises(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            env_key = "MISSING_ENV_12345"
            protocol = "openai"
        """)
        with pytest.raises(ConfigError, match="API key"):
            load_config(path)

    def test_env_key_empty_in_environment_raises(self, toml_file, monkeypatch):
        monkeypatch.setenv("EMPTY_ENV_KEY", "")
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            env_key = "EMPTY_ENV_KEY"
            protocol = "openai"
        """)
        with pytest.raises(ConfigError, match="API key"):
            load_config(path)

    def test_openai_responses_wire_api(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.openai.com/v1"
            api_key = "sk-test"
            protocol = "openai"
            wire_api = "responses"
        """)
        config = load_config(path)
        cfg = config.providers["test"]
        assert cfg.wire_api == "responses"
        from kllmgate.models import ProtocolFormat
        assert cfg.protocol_format == ProtocolFormat.OPENAI_RESPONSES

    def test_model_aliases_parsed(self, toml_file):
        path = toml_file("""\
            [providers.scnet]
            base_url = "https://api.scnet.cn/v1"
            api_key = "sk-test"
            protocol = "openai"

            [model_aliases]
            "MiniMax-M2.5" = "scnet/MiniMax-M2.5"
            "gpt-4" = "scnet/gpt-4"
        """)
        config = load_config(path)
        assert config.model_aliases == {
            "MiniMax-M2.5": "scnet/MiniMax-M2.5",
            "gpt-4": "scnet/gpt-4",
        }

    def test_model_aliases_empty_by_default(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        config = load_config(path)
        assert config.model_aliases == {}

    def test_model_aliases_invalid_target_format_raises(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"

            [model_aliases]
            "gpt-4" = "no-slash"
        """)
        with pytest.raises(ConfigError, match="provider/model"):
            load_config(path)

    def test_model_aliases_unknown_provider_raises(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"

            [model_aliases]
            "gpt-4" = "nonexistent/gpt-4"
        """)
        with pytest.raises(ConfigError, match="nonexistent"):
            load_config(path)


class TestServerConfig:

    def test_server_defaults_when_section_omitted(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        config = load_config(path)
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8500
        assert config.server.log_level == "info"

    def test_server_custom_values(self, toml_file):
        path = toml_file("""\
            [server]
            host = "127.0.0.1"
            port = 9000
            log_level = "debug"

            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        config = load_config(path)
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000
        assert config.server.log_level == "debug"

    def test_server_partial_values(self, toml_file):
        path = toml_file("""\
            [server]
            port = 3000

            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        config = load_config(path)
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 3000
        assert config.server.log_level == "info"

    def test_invalid_log_level_raises(self, toml_file):
        path = toml_file("""\
            [server]
            log_level = "verbose"

            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        with pytest.raises(ConfigError, match="log_level"):
            load_config(path)

    def test_invalid_port_raises(self, toml_file):
        path = toml_file("""\
            [server]
            port = 0

            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        with pytest.raises(ConfigError, match="port"):
            load_config(path)

    def test_port_exceeding_max_raises(self, toml_file):
        path = toml_file("""\
            [server]
            port = 70000

            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        with pytest.raises(ConfigError, match="port"):
            load_config(path)


class TestDefaultProvider:

    def test_default_provider_parsed(self, toml_file):
        path = toml_file("""\
            [server]
            default_provider = "test"

            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        config = load_config(path)
        assert config.default_provider == "test"

    def test_default_provider_none_when_omitted(self, toml_file):
        path = toml_file("""\
            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        config = load_config(path)
        assert config.default_provider is None

    def test_default_provider_unknown_raises(self, toml_file):
        path = toml_file("""\
            [server]
            default_provider = "nonexistent"

            [providers.test]
            base_url = "https://api.example.com"
            api_key = "sk-test"
            protocol = "openai"
        """)
        with pytest.raises(ConfigError, match="nonexistent"):
            load_config(path)
