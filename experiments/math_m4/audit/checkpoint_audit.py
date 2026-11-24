"""M3.1 P0：Checkpoint 证据链审计（零梯度，仅加载与评测）。

任务书指出 M3 报告可能出现"实际加载 Final 却标记 Best"。
本脚本对 M3 现存 16 个权重逐一记录：
  实际路径 / 文件 SHA256 / 模型张量 SHA256 / meta 角色与步数 /
  Best-Final 张量关系 / 固定小面板重评测 macro 与 meta 声明的差异。
不修改任何 M3 文件；缺权重标 NOT_AVAILABLE。

运行：python checkpoint_audit.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
M3 = HERE.parent / "math_m3"
sys.path.insert(0, str(M3))
sys.path.insert(0, str(M3 / "models"))

from tokenizer_m3 import M3Tokenizer  # noqa: E402
from evaluate_m3 import load_rows, build_fixed_panel, evaluate_panel  # noqa: E402
from train_m3 import MODELS  # noqa: E402

ARCH_CFG = {
    "R1_gru_reference": {"d_model": 64, "hidden": 128},
    "R2_tiny_transformer": {"d_model": 64, "nhead": 4, "layers": 2},
    "M3-A_low_rank_neural": {"d": 96, "r": 16, "n_layers": 1},
    "M3-B_adaptive_computation": {"d": 96, "r": 16, "hidden_mult": 2},
}


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(20)
    tok = M3Tokenizer()
    panel = build_fixed_panel(load_rows("val"), per_task=5)  # 85 条，仅证据链复核
    d = M3 / "checkpoints"
    entries = []
    for p in sorted(d.glob("*.pt")):
        try:
            blob = torch.load(p, weights_only=False)
            sd = blob["model"]
            meta = blob.get("meta", {})
            arch = meta.get("arch")
            if arch not in MODELS or arch not in ARCH_CFG:
                entries.append({"file": p.name, "status": "UNKNOWN_ARCH", "arch": arch})
                continue
            model = MODELS[arch](tok.vocab_size, **ARCH_CFG[arch])
            model.load_state_dict(sd, strict=True)
            model.eval()
            ev = evaluate_panel(model, tok, panel)
            entries.append({
                "file": p.name, "status": "OK",
                "file_sha256": file_sha(p), "tensor_sha256": tensor_sha(sd),
                "n_tensors": len(sd), "arch": arch, "seed": meta.get("seed"),
                "meta_role": meta.get("role"), "meta_step": meta.get("step"),
                "meta_macro": meta.get("macro"), "meta_global_step": meta.get("global_step"),
                "meta_init_sha": meta.get("init_sha"), "meta_data_sha": meta.get("data_sha"),
                "meta_tokenizer_sha": meta.get("tokenizer_sha"),
                "remeasured_macro": ev["macro_accuracy"], "remeasured_n": ev["n"],
                "macro_delta": round(ev["macro_accuracy"] - float(meta.get("macro", 0.0)), 4),
            })
        except Exception as exc:
            entries.append({"file": p.name, "status": "ERROR", "error": str(exc)})

    # Best/Final 张量关系
    relations = []
    for e in entries:
        if e.get("status") != "OK" or e.get("meta_role") != "best":
            continue
        fin = next((x for x in entries if x.get("arch") == e["arch"] and x.get("seed") == e["seed"]
                    and x.get("meta_role") == "final" and x.get("status") == "OK"), None)
        if fin:
            relations.append({"arch": e["arch"], "seed": e["seed"],
                              "best_file": e["file"], "final_file": fin["file"],
                              "tensor_identical": e["tensor_sha256"] == fin["tensor_sha256"],
                              "best_macro": e["meta_macro"], "final_macro": fin["meta_macro"],
                              "role_labels_match_steps": e["meta_step"] <= fin["meta_step"]})
    out = {
        "status": "COMPLETE", "n_checkpoints": len(entries),
        "panel": f"val 每任务5条共{len(panel)}条，仅用于证据链复核，非能力排名",
        "note": "权重路径与 SHA 为准；meta 声明的 macro 与重评测差异如实记录",
        "checkpoints": entries, "best_final_relations": relations,
    }
    p = HERE / "M31_CHECKPOINT_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    errs = [e for e in entries if e.get("status") != "OK"]
    print(f"[ckpt] n={len(entries)} errors={len(errs)} relations={len(relations)}")
    print(f"[ckpt] 写出 {p}")


if __name__ == "__main__":
    main()
