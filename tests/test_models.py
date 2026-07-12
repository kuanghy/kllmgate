"""ProviderConfig 和 ProtocolFormat 的单元测试"""

import os

import pytest

from kllmgate.models import ProtocolFormat, ProviderConfig
from kllmgate.errors import ConfigError


class TestProtocolFormat:

    def test_enum_values(self):
        assert ProtocolFormat.OPENAI_CHAT == "openai.chat"
        assert ProtocolFormat.OPENAI_RESPONSES == "openai.responses"
        assert ProtocolFormat.ANTHROPIC_MESSAGES == "anthropic.messages"

    def test_is_string_enum(self):
        assert isinstance(ProtocolFormat.OPENAI_CHAT, str)
        assert ProtocolFormat.OPENAI_CHAT == "openai.chat"

    def test_from_value(self):
        assert ProtocolFormat("openai.chat") is ProtocolFormat.OPENAI_CHAT
        assert ProtocolFormat("anthropic.messages") is ProtocolFormat.ANTHROPIC_MESSAGES

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ProtocolFormat("invalid")


class TestProviderConfig:

    def test_minimal_openai_chat(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            protocol="openai",
        )
        assert cfg.name == "test"
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.api_key == "sk-test"
        assert cfg.env_key is None
        assert cfg.wire_api == "chat"
        assert cfg.tool_style == "standard"
        assert cfg.timeout_seconds == 120
        assert cfg.max_retries == 2
        assert cfg.models is None

    def test_minimal_anthropic(self):
        cfg = ProviderConfig(
            name="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            protocol="anthropic",
        )
        assert cfg.wire_api == "messages"
        assert cfg.protocol_format == ProtocolFormat.ANTHROPIC_MESSAGES

    def test_openai_responses_wire_api(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            protocol="openai",
            wire_api="responses",
        )
        assert cfg.protocol_format == ProtocolFormat.OPENAI_RESPONSES

    def test_openai_chat_protocol_format(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            protocol="openai",
            wire_api="chat",
        )
        assert cfg.protocol_format == ProtocolFormat.OPENAI_CHAT

    def test_custom_tool_style(self):
        cfg = ProviderConfig(
            name="minimax",
            base_url="https://api.example.com",
            api_key="sk-test",
            protocol="openai",
            tool_style="minimax_xml",
        )
        assert cfg.tool_style == "minimax_xml"

    def test_custom_timeout_and_retries(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com",
            api_key="sk-test",
            protocol="openai",
            timeout_seconds=60,
            max_retries=5,
        )
        assert cfg.timeout_seconds == 60
        assert cfg.max_retries == 5

    def test_models_whitelist(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com",
            api_key="sk-test",
            protocol="openai",
            models=["gpt-4.1", "gpt-4o-mini"],
        )
        assert cfg.models == ["gpt-4.1", "gpt-4o-mini"]

    def test_resolve_api_key_direct(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com",
            api_key="sk-direct",
            env_key="SOME_ENV",
            protocol="openai",
        )
        assert cfg.resolve_api_key() == "sk-direct"

    def test_resolve_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com",
            env_key="TEST_API_KEY",
            protocol="openai",
        )
        assert cfg.resolve_api_key() == "sk-from-env"

    def test_resolve_api_key_direct_takes_priority(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com",
            api_key="sk-direct",
            env_key="TEST_API_KEY",
            protocol="openai",
        )
        assert cfg.resolve_api_key() == "sk-direct"

    def test_resolve_api_key_missing_raises(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com",
            protocol="openai",
        )
        with pytest.raises(ConfigError, match="API key"):
            cfg.resolve_api_key()

    def test_resolve_api_key_empty_env_raises(self, monkeypatch):
        monkeypatch.setenv("EMPTY_KEY", "")
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com",
            env_key="EMPTY_KEY",
            protocol="openai",
        )
        with pytest.raises(ConfigError, match="API key"):
            cfg.resolve_api_key()

    def test_resolve_api_key_missing_env_raises(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com",
            env_key="NONEXISTENT_KEY_12345",
            protocol="openai",
        )
        with pytest.raises(ConfigError, match="API key"):
            cfg.resolve_api_key()

    def test_base_url_trailing_slash_preserved(self):
        """ProviderConfig 不做 URL 清理，由 config.py 负责"""
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com/v1/",
            api_key="sk-test",
            protocol="openai",
        )
        assert cfg.base_url == "https://api.example.com/v1/"


class TestResolveForInbound:

    def test_single_openai_ignores_inbound_anthropic(self):
        cfg = ProviderConfig(
            name="test",
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            protocol="openai",
            wire_api="chat",
        )
        resolved = cfg.resolve_for_inbound(ProtocolFormat.ANTHROPIC_MESSAGES)
        assert resolved.protocol == "openai"
        assert resolved.protocol_format == ProtocolFormat.OPENAI_CHAT
        assert resolved.base_url == "https://api.example.com/v1"

    def test_auto_openai_inbound_uses_openai_subsection(self):
        from kllmgate.models import ProtocolEndpointConfig

        cfg = ProviderConfig(
            name="dual",
            protocol="auto",
            api_key="sk-top",
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
                wire_api="chat",
                tool_style="minimax_xml",
            ),
            anthropic=ProtocolEndpointConfig(
                base_url="https://anthropic.example.com",
                wire_api="messages",
            ),
        )
        resolved = cfg.resolve_for_inbound(ProtocolFormat.OPENAI_CHAT)
        assert resolved.base_url == "https://openai.example.com/v1"
        assert resolved.protocol_format == ProtocolFormat.OPENAI_CHAT
        assert resolved.tool_style == "minimax_xml"
        assert resolved.resolve_api_key() == "sk-top"

    def test_auto_anthropic_inbound_uses_anthropic_subsection(self):
        from kllmgate.models import ProtocolEndpointConfig

        cfg = ProviderConfig(
            name="dual",
            protocol="auto",
            api_key="sk-top",
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
            ),
            anthropic=ProtocolEndpointConfig(
                base_url="https://anthropic.example.com",
                api_key="sk-ant",
            ),
        )
        resolved = cfg.resolve_for_inbound(ProtocolFormat.ANTHROPIC_MESSAGES)
        assert resolved.base_url == "https://anthropic.example.com"
        assert resolved.protocol_format == ProtocolFormat.ANTHROPIC_MESSAGES
        assert resolved.resolve_api_key() == "sk-ant"

    def test_auto_responses_inbound_still_uses_openai_subsection(self):
        from kllmgate.models import ProtocolEndpointConfig

        cfg = ProviderConfig(
            name="dual",
            protocol="auto",
            api_key="sk-top",
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
                wire_api="chat",
            ),
        )
        resolved = cfg.resolve_for_inbound(ProtocolFormat.OPENAI_RESPONSES)
        assert resolved.protocol_format == ProtocolFormat.OPENAI_CHAT
        assert resolved.base_url == "https://openai.example.com/v1"

    def test_auto_missing_subsection_raises(self):
        from kllmgate.models import ProtocolEndpointConfig

        cfg = ProviderConfig(
            name="dual",
            protocol="auto",
            api_key="sk-top",
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
            ),
        )
        with pytest.raises(ConfigError, match="anthropic"):
            cfg.resolve_for_inbound(ProtocolFormat.ANTHROPIC_MESSAGES)

    def test_auto_models_override(self):
        from kllmgate.models import ProtocolEndpointConfig

        cfg = ProviderConfig(
            name="dual",
            protocol="auto",
            api_key="sk-top",
            models=["shared"],
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
                models=["openai-only"],
            ),
            anthropic=ProtocolEndpointConfig(
                base_url="https://anthropic.example.com",
            ),
        )
        assert cfg.resolve_for_inbound(
            ProtocolFormat.OPENAI_CHAT,
        ).models == ["openai-only"]
        assert cfg.resolve_for_inbound(
            ProtocolFormat.ANTHROPIC_MESSAGES,
        ).models == ["shared"]

    def test_auto_protocol_format_property_raises(self):
        cfg = ProviderConfig(
            name="dual",
            base_url="",
            protocol="auto",
            api_key="sk-top",
        )
        with pytest.raises(ConfigError, match="auto"):
            _ = cfg.protocol_format

    def test_positional_args_name_base_url_protocol(self):
        cfg = ProviderConfig(
            "test",
            "https://api.example.com/v1",
            "openai",
        )
        assert cfg.name == "test"
        assert cfg.base_url == "https://api.example.com/v1"
        assert cfg.protocol == "openai"
        assert cfg.protocol_format == ProtocolFormat.OPENAI_CHAT

    def test_subsection_env_key_falls_back_to_top_api_key(
        self, monkeypatch,
    ):
        from kllmgate.models import ProtocolEndpointConfig

        monkeypatch.delenv("MISSING_SUB_KEY", raising=False)
        cfg = ProviderConfig(
            name="dual",
            base_url="",
            protocol="auto",
            api_key="sk-top",
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
                env_key="MISSING_SUB_KEY",
            ),
        )
        resolved = cfg.resolve_for_inbound(ProtocolFormat.OPENAI_CHAT)
        assert resolved.resolve_api_key() == "sk-top"

    def test_subsection_env_key_wins_when_present(self, monkeypatch):
        from kllmgate.models import ProtocolEndpointConfig

        monkeypatch.setenv("SUB_KEY", "sk-sub")
        cfg = ProviderConfig(
            name="dual",
            base_url="",
            protocol="auto",
            api_key="sk-top",
            openai=ProtocolEndpointConfig(
                base_url="https://openai.example.com/v1",
                env_key="SUB_KEY",
            ),
        )
        resolved = cfg.resolve_for_inbound(ProtocolFormat.OPENAI_CHAT)
        assert resolved.resolve_api_key() == "sk-sub"
