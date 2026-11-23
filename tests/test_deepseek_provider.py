from types import SimpleNamespace

from nmind.bridge.providers import DeepSeekProvider, build_provider
from nmind.bot.chat_service import ChatRequest, ChatService
from nmind.config import Config


def test_deepseek_provider_uses_safe_defaults_without_key():
    cfg = Config(cloud={"mode": "ask", "provider": "deepseek", "api_key_env": "TEST_NMIND_KEY"})
    assert isinstance(build_provider("ask", cfg), type(build_provider("off", Config())))


def test_deepseek_provider_is_explicit_and_chat_source_is_distinct():
    class FakeProvider:
        name = "deepseek"

        def ask(self, req):
            return SimpleNamespace(ok=True, provider=self.name, text="这是外部问答结果。")

    out = ChatService(cloud_provider=FakeProvider()).reply(
        ChatRequest.new("s", "什么是Python？"))
    assert out.response_source == "cloud_teacher"
    assert "N-Cell" in out.note


def test_deepseek_adapter_constants_do_not_contain_secret():
    assert DeepSeekProvider.DEFAULT_BASE_URL == "https://api.deepseek.com"
    assert DeepSeekProvider.DEFAULT_MODEL == "deepseek-chat"
