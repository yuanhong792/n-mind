"""V1.7 P5: interactive Chinese chat driven ONLY by the trained N-MIND network.
No pre-baked answers, no name->color rules, no online LLM fallback. Real model output shown,
including failures. Commands: /reset clears context, /exit quits.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2
from nmind.language.bpe_decoder import generate_bpe, effective_token_ids, utf8_prefix_status

V17 = ROOT / "experiments/language_v1/v17"
LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]

class ChatSession:
    def __init__(self, candidate):
        torch.set_num_threads(1)
        self.tok = ByteBPETokenizer.train([p+a for p,a in LEGACY], vocab_size=4096)
        self.model, self.meta = load_language_v2(V17/candidate, self.tok)
        self.model.eval(); self.model.use_router_load_update = False
        self.history = []   # list of (user, assistant) turns
        self.last = None
    def build_prompt(self, user_text):
        # 多轮上下文: 逐轮拼接, 保持训练时的 "用户：...\n助手：..." 格式
        parts = []
        for u, a in self.history:
            parts.append(f"用户：{u}\n助手：{a}")
        parts.append(f"用户：{user_text}\n助手：")
        return "\n".join(parts)
    def reply(self, user_text):
        prompt = self.build_prompt(user_text)
        t0 = time.perf_counter()
        g = generate_bpe(self.model, self.tok, prompt, max_new_tokens=24, utf8_safe=True)
        dt = time.perf_counter()-t0
        self.history.append((user_text, g.text))
        info = {"prompt_tokens": len(self.tok.encode(prompt)), "gen_tokens": len(g.token_ids),
                "seconds": round(dt,3), "finish_reason": g.finish_reason,
                "utf8_status": utf8_prefix_status(bytes.fromhex(g.raw_byte_hex)),
                "rejected_utf8_tokens": len(g.rejected_utf8_tokens), "raw_byte_hex": g.raw_byte_hex}
        self.last = info
        return g.text, info
    def reset(self): self.history = []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="candidate_v17_best.pt")
    ap.add_argument("--batch", help="可选: 从 jsonl 文件批量测试(每行 {input: ...}), 非交互")
    ap.add_argument("--batch-out", default="V17_CHAT_TRANSCRIPT.jsonl")
    args = ap.parse_args()
    s = ChatSession(args.candidate)
    print(f"[N-MIND V1.7] 已加载 {args.candidate} | 参数量 {s.model.n_params()} | 指令: /reset 清空, /exit 退出")
    if args.batch:
        out = []
        for line in open(args.batch, encoding="utf-8"):
            if not line.strip(): continue
            u = json.loads(line).get("input","")
            a, info = s.reply(u)
            out.append({"input":u,"output":a,"info":info})
            print(f"用户：{u}\n助手：{a}  <{info['finish_reason']}, {info['seconds']}s, {info['gen_tokens']}tok>")
        (V17/args.batch_out).write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in out)+"\n",encoding="utf-8")
        print(f"[已保存 {len(out)} 组对话 -> {args.batch_out}]")
        return
    while True:
        try:
            u = input("用户：").strip()
        except EOFError:
            break
        if not u: continue
        if u == "/exit": print("[退出]"); break
        if u == "/reset": s.reset(); print("[上下文已清空]"); continue
        a, info = s.reply(u)
        print(f"助手：{a}")
        print(f"  [{info['finish_reason']} | {info['seconds']}s | prompt {info['prompt_tokens']}tok / gen {info['gen_tokens']}tok | UTF-8 {info['utf8_status']} | 拒非法 {info['rejected_utf8_tokens']}]")

if __name__=="__main__": main()
