"""N-MIND 8B 神经主体：DeepSeek-R1-0528-Qwen3-8B 本地加载 / 推理 / 神经活动采集。

总纲（路线 C）关键约束：
- 直接继承官方预训练权重，**不**随机初始化、**不**从零训练基础模型。
- 架构是 Qwen3 稠密 Transformer + DeepSeek 推理训练，不是 MoE/MLA。
- 必须能访问内部张量、梯度、权重、层与学习机制；禁止只调 API 冒充神经主体。
- 教室/教师/检索可关；关掉后学生须能独立回答。

依赖：transformers / torch / accelerate / bitsandbytes / peft（QLoRA 时）
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

HERE = Path(__file__).resolve().parent
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def _strip_thinking(text: str) -> str:
    """去掉 R1 的 <think>...</think> 内部推理，只保留最终答案。"""
    if "<think>" in text:
        # 思考链未闭合（被 max_new_tokens 截断）时整段丢弃
        if "</think>" not in text:
            return ""
        return text.split("</think>", 1)[1].strip()
    return text.strip()
LOCAL_MODEL_DIR = Path(
    __import__("os").environ.get(
        "NMIND_MODEL_DIR", str(HERE / "models" / "DeepSeek-R1-0528-Qwen3-8B")
    )
)


@dataclass
class LoadConfig:
    model_id: str = MODEL_ID
    local_dir: Path = LOCAL_MODEL_DIR
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    device_map: str = "auto"
    max_memory_gb: float = 10.5          # 12.82GB 显存留 ~2GB 余量给 KV/碎片
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"
    low_cpu_mem_usage: bool = True
    # 4bit 预量化缓存（P1 修复 424 秒冷启动）
    quantized_cache_dir: Path | None = None   # None 时默认 local_dir.parent / "4bit-cache"
    prefer_cache: bool = True                 # True=优先直接加载已量化权重，跳过逐层量化


@dataclass
class NeuralSnapshot:
    """真实神经活动快照（不是模型生成的解释）。"""

    ts: str
    step: int
    layer_norm_means: list[float] = field(default_factory=list)
    layer_norm_stds: list[float] = field(default_factory=list)
    top_token_probs: list[dict] = field(default_factory=list)
    hidden_norm: float = 0.0
    vram_gb: float = 0.0
    adapter_active: bool = False


class Neuro8B:
    """N-MIND 8B 神经主体封装。"""

    def __init__(self, cfg: LoadConfig | None = None):
        self.cfg = cfg or LoadConfig()
        self.model = None
        self.tokenizer = None
        self.device = None
        self._hooks: list[Any] = []
        self._last_snapshot: dict[str, Any] = {}
        self._step = 0
        self.load_info: dict[str, Any] = {}

    # ------------------------------------------------------------ 4bit 预量化缓存

    @property
    def cache_dir(self) -> Path:
        if self.cfg.quantized_cache_dir is not None:
            return self.cfg.quantized_cache_dir
        return self.cfg.local_dir.parent / "DeepSeek-R1-0528-Qwen3-8B-4bit-cache"

    def _cache_ready(self) -> bool:
        """缓存完整性检查：READY 标记 + config + 至少一个 safetensors。"""
        d = self.cache_dir
        return ((d / "READY").exists() and (d / "config.json").exists()
                and any(d.glob("*.safetensors")))

    def save_quantized_cache(self) -> dict:
        """把已加载的 4bit 模型保存为预量化权重，供后续秒级加载。

        原理（transformers 官方）：Bnb4bitDeserialize + Params4bit.from_prequantized
        会直接重建量化权重，不再逐层量化 → 消除 424 秒冷启动。
        兼容性：保存的是**冻结的 4bit 基础权重**，QLoRA 适配器单独保存/加载，互不干扰。
        """
        if self.model is None:
            raise RuntimeError("模型未加载，无法保存缓存")
        if not getattr(self.model.config, "quantization_config", None):
            raise RuntimeError("当前模型非 4bit 量化，不能保存预量化缓存")
        d = self.cache_dir
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        self.model.save_pretrained(str(d), safe_serialization=True)
        self.tokenizer.save_pretrained(str(d))
        (d / "READY").write_text(
            json.dumps({"model_id": self.cfg.model_id,
                        "quant_type": self.cfg.bnb_4bit_quant_type,
                        "compute_dtype": self.cfg.bnb_4bit_compute_dtype,
                        "double_quant": self.cfg.bnb_4bit_use_double_quant,
                        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "save_sec": round(time.perf_counter() - t0, 1),
                        "note": "预量化 NF4 权重，加载走 from_prequantized 不重新量化；"
                                "基础权重冻结，与 QLoRA 适配器兼容。"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        info = {"path": str(d), "save_sec": round(time.perf_counter() - t0, 1),
                "n_shards": len(list(d.glob("*.safetensors")))}
        print(f"[cache] 预量化缓存已保存 {d} ({info['n_shards']} 分片, "
              f"{info['save_sec']}s)", flush=True)
        return info

    # ------------------------------------------------------------ 加载

    def load(self) -> dict:
        """加载模型。优先读 4bit 预量化缓存（秒级），无缓存才逐层量化。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        src = str(self.cfg.local_dir) if self.cfg.local_dir.exists() else self.cfg.model_id
        if not self.cfg.local_dir.exists():
            raise FileNotFoundError(
                f"本地模型不存在: {self.cfg.local_dir}\n"
                f"请先下载 {self.cfg.model_id} 到该目录，禁止静默改用其他模型。"
            )

        t0 = time.perf_counter()
        use_cache = self.cfg.load_in_4bit and self.cfg.prefer_cache and self._cache_ready()
        model_src = str(self.cache_dir) if use_cache else src
        if use_cache:
            print(f"[cache] 命中预量化缓存，跳过逐层量化: {self.cache_dir}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_src, trust_remote_code=self.cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant = None
        if self.cfg.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.cfg.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.cfg.bnb_4bit_use_double_quant,
            )

        max_memory = {0: f"{int(self.cfg.max_memory_gb)}GiB"}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_src,
            quantization_config=quant,
            device_map=self.cfg.device_map,
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.cfg.attn_implementation,
            trust_remote_code=self.cfg.trust_remote_code,
            low_cpu_mem_usage=self.cfg.low_cpu_mem_usage,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.load_info = {
            "status": "MODEL_LOADED",
            "model_id": self.cfg.model_id,
            "source": model_src,
            "from_prequantized_cache": use_cache,
            "architecture": type(self.model.config).__name__,
            "model_type": getattr(self.model.config, "model_type", "?"),
            "n_layers": getattr(self.model.config, "num_hidden_layers", 0),
            "hidden_size": getattr(self.model.config, "hidden_size", 0),
            "vocab_size": getattr(self.model.config, "vocab_size", 0),
            "quantized_4bit": self.cfg.load_in_4bit,
            "quant_type": self.cfg.bnb_4bit_quant_type if quant else None,
            "device": str(self.device),
            "load_sec": round(time.perf_counter() - t0, 2),
            "peak_vram_gb": self.vram_gb(),
            "param_count": getattr(self.model.config, "num_parameters", None),
            "note": "Qwen3 稠密 Transformer + DeepSeek 推理训练；非 MoE/MLA。",
        }
        return self.load_info

    # ------------------------------------------------------------ 推理

    @torch.no_grad()
    def chat(self, messages: list[dict[str, str]], max_new_tokens: int = 256,
             temperature: float = 0.7, top_p: float = 0.9,
             capture_neural: bool = True, enable_thinking: bool = True,
             keep_thinking: bool = False) -> dict:
        """同步中文聊天，返回文本 + 真实神经快照。

        enable_thinking=False：尝试 Qwen3 no_think 模板（R1 蒸馏模型可能忽略）。
        keep_thinking=False：输出时剥离 <think>...</think>，只返回最终答案。
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先 load()")

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except (TypeError, ValueError):
            # 老模板不支持 enable_thinking 时降级
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        in_len = enc["input_ids"].shape[-1]

        hooks = self._install_hooks() if capture_neural else []
        try:
            t0 = time.perf_counter()
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            gen_sec = time.perf_counter() - t0
        finally:
            for h in hooks:
                h.remove()

        new = out[:, in_len:]
        reply = self.tokenizer.decode(new[0], skip_special_tokens=True).strip()
        # DeepSeek-R1 蒸馏模型强制输出 <think>...</think>，即使 enable_thinking=False
        # 也会带思考链。按参数决定是否保留。
        if not keep_thinking:
            reply = _strip_thinking(reply)
        self._step += 1
        snap = self._read_snapshot()
        return {
            "reply": reply,
            "input_tokens": in_len,
            "output_tokens": new.shape[-1],
            "gen_sec": round(gen_sec, 3),
            "tokens_per_s": round(new.shape[-1] / max(1e-6, gen_sec), 1),
            "neural": snap,
        }

    def chat_stream(self, messages: list[dict[str, str]], max_new_tokens: int = 2048,
                    temperature: float = 0.7, top_p: float = 0.9,
                    keep_thinking: bool = False) -> Iterator[dict]:
        """真实流式生成，逐 chunk 产出结构化事件。

        yield 形如 {"delta": "...", "state": "thinking|answer|done|truncated"}
        **P1 修复要点**（不再只是加大 max_new_tokens）：
        - 跨 chunk 边界：<think> / </think> 可能被切开，用状态机 + 拖尾缓存处理
        - 思考链未闭合即到上限：明确标记 truncated，不把未完成内容当最终答案
        - 空白回复：截断但已有部分答案时照常输出并标记 truncated，不吞掉
        - 支持继续生成：调用方可拿 state=truncated 后用续写提示重试
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        if self.model is None:
            raise RuntimeError("模型未加载")

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_tokens = int(enc["input_ids"].shape[-1])
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **enc, streamer=streamer, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
        )
        generation = {"result": None, "error": None}
        def run_generation():
            try:
                generation["result"] = self.model.generate(
                    **kwargs, return_dict_in_generate=True)
            except BaseException as exc:  # 传递线程异常，禁止静默结束
                generation["error"] = repr(exc)
                try:
                    streamer.end()
                except Exception:
                    pass
        t = Thread(target=run_generation, daemon=True)
        t0 = time.perf_counter()
        t.start()

        # 状态机：THINK(吞) -> ANSWER(吐) -> DONE / TRUNCATED
        state = "think" if not keep_thinking else "answer"
        pending = ""          # 跨 chunk 拖尾缓存（可能是不完整的标签前缀）
        think_buf = ""        # 缓存思考内容（检测是否真的出现 <think> 标记）
        saw_think_open = False  # 是否见过 <think> 开头
        emitted_any = False
        total = 0

        for piece in streamer:
            if not piece:
                continue
            total += 1
            if keep_thinking:
                yield {"delta": piece, "state": "answer"}
                emitted_any = True
                continue

            buf = pending + piece
            pending = ""

            if state == "think":
                # 检测 <think> 开头（可能跨 chunk），确认是 R1 思考链
                if "<think" in buf or "<think"[:min(len(buf), 6)] == buf[:min(len(buf), 6)]:
                    saw_think_open = True
                think_buf += buf
                # 找 </think>（可能跨 chunk）：保留尾部候选
                idx = think_buf.find("</think>")
                if idx >= 0:
                    state = "answer"
                    tail = think_buf[idx + len("</think>"):]
                    think_buf = ""
                    if tail:
                        yield {"delta": tail, "state": "answer"}
                        emitted_any = True
                else:
                    # 保留可能是标签开头的尾巴（最长 7 字符 "</think"）
                    for k in range(min(len(think_buf), 8), 0, -1):
                        if "</think"[:k] == think_buf[-k:]:
                            pending = think_buf[-k:]
                            think_buf = think_buf[:-k]
                            break
            else:  # answer
                yield {"delta": buf, "state": "answer"}
                emitted_any = True

        t.join(timeout=10)
        if generation["error"]:
            self._write_stream_trace(messages, input_tokens, total, None, "error", generation["error"], t0)
            raise RuntimeError("模型生成线程异常：" + generation["error"][:240])

        result = generation["result"]
        generated_tokens = 0
        last_token_id = None
        if result is not None and hasattr(result, "sequences"):
            seq = result.sequences
            generated_tokens = max(0, int(seq.shape[-1]) - input_tokens)
            if seq.shape[-1] > 0:
                last_token_id = int(seq[0, -1])
        elapsed = time.perf_counter() - t0
        eos_ids = self.tokenizer.eos_token_id
        eos_set = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
        finish_reason = "eos" if last_token_id in eos_set else ("length" if generated_tokens >= max_new_tokens else "unknown")

        # 收尾判定
        if state == "think":
            if saw_think_open:
                # 真的出现 <think> 但没闭合 → 截断，内部推理绝不外泄
                self._write_stream_trace(messages, input_tokens, generated_tokens, last_token_id, "truncated", "unfinished_thinking", t0)
                yield {"delta": "[思考未完成，模型未给出最终答案。请继续生成]",
                       "state": "truncated", "emitted": emitted_any}
            else:
                # 从未出现 <think>：普通模型输出，全部当答案透传（含 think_buf）
                whole = think_buf + pending
                if whole:
                    yield {"delta": whole, "state": "answer"}
                yield {"delta": "", "state": "done", "chunks": total}
        else:
            # R1 可能在达到上限时仍处于答案阶段；不能把半句话伪装成正常结束。
            # 这里按实际产出的 streamer piece 数做保守判定，前端据此显示截断提示。
            if finish_reason == "length":
                self._write_stream_trace(messages, input_tokens, generated_tokens, last_token_id, "truncated", "max_new_tokens", t0)
                yield {"delta": "[本段达到安全容量，内容未标记为完成，将由上层续写]",
                       "state": "truncated", "emitted": emitted_any, "chunks": total}
            else:
                self._write_stream_trace(messages, input_tokens, generated_tokens, last_token_id, "done", finish_reason, t0)
                yield {"delta": "", "state": "done", "chunks": total, "finish_reason": finish_reason}

    def _write_stream_trace(self, messages, input_tokens, generated_tokens, last_token_id, state, reason, started):
        """只记录可审计元数据，不写入聊天正文。"""
        try:
            path = HERE / "results" / "diagnostics" / "chat_stream.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            vram = {}
            if torch.cuda.is_available():
                vram = {"allocated": int(torch.cuda.memory_allocated()), "reserved": int(torch.cuda.memory_reserved())}
            payload = {"ts": time.time(), "model_id": MODEL_ID,
                       "messages_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                       "message_count": len(messages), "input_tokens": input_tokens,
                       "generated_tokens": int(generated_tokens or 0), "last_token_id": last_token_id,
                       "state": state, "finish_reason": reason, "elapsed_sec": round(time.perf_counter()-started,3),
                       "cuda": vram}
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------ 神经活动采集

    def _install_hooks(self) -> list[Any]:
        """挂 forward hook 采集真实 hidden state 统计（层级聚合，不逐参数渲染）。"""
        self._last_snapshot = {"layer_norm_means": [], "layer_norm_stds": []}

        # 按层聚合：每个 decode step 都会触发 hook，必须只保留该层最后一次统计，
        # 否则 36 层 × 20 token 会输出 720 个"层"的假数据。
        self._last_snapshot["_layer_means"] = {}
        self._last_snapshot["_layer_stds"] = {}

        def make_hook(idx: int):
            def hook(module, _inp, output):
                t = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(t, torch.Tensor):
                    return
                with torch.no_grad():
                    flat = t.detach().float().flatten()
                    if flat.numel() == 0:
                        return
                    sample = flat[::max(1, flat.numel() // 4096)]
                    self._last_snapshot["_layer_means"][idx] = round(
                        sample.mean().item(), 6)
                    self._last_snapshot["_layer_stds"][idx] = round(
                        sample.std().item(), 6)
            return hook

        layers = self._get_layers()
        for i, layer in enumerate(layers):
            if i % 2 == 0:        # 每两层采一次，降低开销
                self._hooks.append(
                    layer.register_forward_hook(make_hook(i)))
        return self._hooks

    def _get_layers(self) -> list[Any]:
        m = self.model
        for attr in ("model", "base_model", "transformer"):
            if hasattr(m, attr):
                inner = getattr(m, attr)
                if hasattr(inner, "layers"):
                    return list(inner.layers)
        if hasattr(m, "layers"):
            return list(m.layers)
        return []

    def _read_snapshot(self) -> dict:
        means = self._last_snapshot.get("_layer_means", {})
        stds = self._last_snapshot.get("_layer_stds", {})
        idxs = sorted(means)
        s = {
            "layer_norm_means": [means[i] for i in idxs],
            "layer_norm_stds": [stds[i] for i in idxs],
            "layer_indices": idxs,
            "step": self._step,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "vram_gb": self.vram_gb(),
            "adapter_active": any(
                "lora" in n.lower() for n, _ in self.model.named_parameters()
            ) if self.model else False,
            "n_layers_sampled": len(idxs),
            "n_layers_total": len(self._get_layers()),
        }
        return s

    # ------------------------------------------------------------ 资源

    def vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)

    def resource_status(self) -> dict:
        import os
        st = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device) if self.device else None,
            "peak_vram_gb": self.vram_gb(),
            "model_loaded": self.model is not None,
        }
        if torch.cuda.is_available():
            st["gpu"] = torch.cuda.get_device_name(0)
            st["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        st["cpu_cores"] = os.cpu_count()
        st["adapter_active"] = bool(
            self.model and any("lora" in n.lower()
                               for n, _ in self.model.named_parameters()))
        return st

    def unload(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-test", action="store_true", help="只加载模型并报资源")
    ap.add_argument("--ask", default="", help="单轮提问后退出")
    args = ap.parse_args()

    bot = Neuro8B()
    print(json.dumps(bot.load(), ensure_ascii=False, indent=2), flush=True)
    if args.ask:
        r = bot.chat([{"role": "user", "content": args.ask}])
        print("\n[reply]\n" + r["reply"], flush=True)
        print("\n[resource]\n" + json.dumps(bot.resource_status(),
                                            ensure_ascii=False, indent=2), flush=True)
