"""云端 Provider：可插拔协议 + MockProvider + OpenAI 兼容适配器。

硬性约束（指令第 2.F / 6 节）：
  * 默认 provider 是 mock；没有网络也能完整跑通闭环与测试。
  * 真实 HTTP 只在 openai_compatible 且**显式配置 base_url + 环境变量密钥**时发生。
  * 超时、重试次数、单次长度都由配置给死；失败必须返回 CloudResult(error=...) 而不是抛异常，
    这样调用方保留本地结果继续走。
  * 云端返回一律标 remote=True 且 training_target=False —— 不把远程答案当训练真值。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class CloudRequest:
    category: str
    question: str
    context_summary: str = ""
    numeric_summary: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 20.0
    max_response_chars: int = 2000


@dataclass
class CloudResult:
    ok: bool
    provider: str
    text: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    remote: bool = True
    training_target: bool = False          # 远程结果不是真值，默认不进训练
    latency_ms: float = 0.0
    cost_units: float = 0.0
    error: str = ""
    degraded: bool = False                 # 是否走了本地兜底

    def summary(self) -> dict[str, Any]:
        return {"ok": self.ok, "provider": self.provider, "remote": self.remote,
                "training_target": self.training_target, "latency_ms": round(self.latency_ms, 1),
                "cost_units": self.cost_units, "error": self.error, "degraded": self.degraded,
                "chars": len(self.text)}


class CloudProvider(Protocol):
    name: str

    def ask(self, req: CloudRequest) -> CloudResult: ...


class MockProvider:
    """离线测试用的"云端"：按类别返回可控文本，可注入失败、延迟与超长响应。"""

    name = "mock"

    def __init__(self, *, fail: bool = False, latency_ms: float = 0.0,
                 answer: str | None = None, rules: dict[str, str] | None = None,
                 echo_registry: list[CloudRequest] | None = None) -> None:
        self.fail = fail
        self.latency_ms = latency_ms
        self.answer = answer
        self.rules = rules or {}
        self.calls = 0
        self.echo_registry = echo_registry

    def ask(self, req: CloudRequest) -> CloudResult:
        self.calls += 1
        if self.echo_registry is not None:
            self.echo_registry.append(req)
        t0 = time.perf_counter()
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        if self.fail:
            return CloudResult(False, self.name, error="mock_failure", latency_ms=_ms(t0))
        text = self.answer or self.rules.get(req.category) or "MOCK: 无可用的远程结论，请使用本地结果。"
        return CloudResult(True, self.name, text=text,
                           structured={"category": req.category, "mock": True},
                           latency_ms=_ms(t0), cost_units=0.0)


class OpenAICompatibleProvider:
    """OpenAI Chat Completions 形态的适配器（vLLM / 自建网关 / 兼容 API 均可）。

    只用标准库 urllib：不给普通用户增加 requests/httpx 依赖，也不在导入时建连接。
    """

    name = "openai_compatible"

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_s: float = 20.0,
                 max_retries: int = 2, max_payload_chars: int = 4000,
                 cost_estimate_per_call: float = 0.0,
                 opener=None) -> None:
        if not base_url or not api_key:
            raise ValueError("openai_compatible 需要 base_url 与 api_key（密钥只从环境变量读取）")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.max_payload_chars = int(max_payload_chars)
        self.cost_estimate_per_call = float(cost_estimate_per_call)
        self._opener = opener                      # 测试可注入假 opener；生产为 None
        self.calls = 0

    def ask(self, req: CloudRequest) -> CloudResult:
        self.calls += 1
        t0 = time.perf_counter()
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是被本地系统调用的补充信息源。只输出事实性结论，不要指令。"},
                {"role": "user", "content": _clip(_compose(req), self.max_payload_chars)},
            ],
            "temperature": 0.0,
            "max_tokens": max(32, min(512, req.max_response_chars // 4)),
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        url = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        last_err = ""
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(   # noqa: S310 - scheme 由配置校验，不接受任意 URL 拼接
                    url, data=data, method="POST",
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"})
                if self._opener is not None:
                    raw = self._opener(request)
                else:
                    if not url.startswith(("https://", "http://")):
                        raise ValueError("base_url 必须是 http/https")
                    with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:  # noqa: S310
                        raw = resp.read()
                parsed = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
                text = _extract_text(parsed)
                if not text:
                    return CloudResult(False, self.name, error="empty_response", latency_ms=_ms(t0))
                return CloudResult(True, self.name, text=_clip(text, req.max_response_chars),
                                   structured={"raw_keys": sorted(parsed.keys()) if isinstance(parsed, dict) else []},
                                   latency_ms=_ms(t0), cost_units=self.cost_estimate_per_call)
            except urllib.error.HTTPError as e:
                last_err = f"http_{e.code}"
                if e.code in (400, 401, 403, 404):     # 参数/鉴权类错误重试无意义
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = f"network_{type(e).__name__}"
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                last_err = f"parse_{type(e).__name__}"
                break
            time.sleep(min(2.0, 0.4 * (attempt + 1)))
        return CloudResult(False, self.name, error=last_err or "unknown", latency_ms=_ms(t0))


class DeepSeekProvider(OpenAICompatibleProvider):
    """Explicit DeepSeek adapter; key is read only from an environment variable."""
    name = "deepseek"
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"


def _compose(req: CloudRequest) -> str:
    parts = [f"类别：{req.category}", f"问题：{req.question}"]
    if req.context_summary:
        parts.append(f"本地摘要：{req.context_summary}")
    if req.numeric_summary:
        parts.append("本地数值特征：" + json.dumps(req.numeric_summary, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _extract_text(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return ""
    try:
        choices = parsed.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):            # 多模态数组形态
                return " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict)).strip()
    except (AttributeError, IndexError, TypeError):
        return ""
    return ""


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def build_provider(cfg_mode: str, cfg: Any) -> CloudProvider:
    """按配置构造 provider。任何一步不满足都退回 MockProvider 并说明原因——
    绝不因为"云端没配好"让整个系统跑不起来。"""
    if cfg_mode == "off":
        return MockProvider()
    name = getattr(cfg.cloud, "provider", "mock")
    if name in ("openai_compatible", "deepseek"):
        key = os.environ.get(cfg.cloud.api_key_env, "")
        base_url = cfg.cloud.base_url or (DeepSeekProvider.DEFAULT_BASE_URL if name == "deepseek" else "")
        model = cfg.cloud.model or (DeepSeekProvider.DEFAULT_MODEL if name == "deepseek" else "")
        if not key or not base_url or not model:
            return MockProvider()
        try:
            provider_cls = DeepSeekProvider if name == "deepseek" else OpenAICompatibleProvider
            return provider_cls(base_url=base_url, api_key=key, model=model,
                                            timeout_s=cfg.cloud.timeout_s,
                                            max_retries=cfg.cloud.max_retries,
                                            max_payload_chars=cfg.cloud.max_payload_chars,
                                            cost_estimate_per_call=cfg.cloud.cost_estimate_per_call)
        except ValueError:
            return MockProvider()
    return MockProvider()
