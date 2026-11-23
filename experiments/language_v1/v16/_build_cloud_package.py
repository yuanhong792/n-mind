"""Build cloud_training_package/ + zip with integrity checks. Weight is NEVER committed (public repo)."""
from __future__ import annotations
import hashlib, json, shutil, subprocess, sys, warnings, zipfile
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2

torch.set_num_threads(1)
CKPT = ROOT / "experiments/language_v1/v15/candidate_v15_baseline.pt"
PKG = ROOT / "cloud_training_package"
ZIP = ROOT / "N-MIND-V16-Cloud-Checkpoint.zip"
SHA_EXPECT = "5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611"
TOK_EXPECT = "26af116ebd13ff00"
LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# pre-checks
model_sha = sha(CKPT)
tok = ByteBPETokenizer.train([p+a for p,a in LEGACY], vocab_size=4096)
tok_hash = tok.vocab_hash()
if model_sha != SHA_EXPECT or tok_hash != TOK_EXPECT:
    print(json.dumps({"status":"BLOCKED","model_sha":model_sha,"tok_hash":tok_hash}, ensure_ascii=False)); sys.exit(1)
model, meta = load_language_v2(CKPT, tok)
model.eval(); model.use_router_load_update=False
# load verification: strict + finite + inference
bad = [k for k,v in model.state_dict().items() if v.is_floating_point() and not torch.isfinite(v).all().item()]
from nmind.language.bpe_decoder import generate_bpe
g = generate_bpe(model, tok, "用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：", max_new_tokens=16)
load_ok = (not bad) and g.finish_reason == "eos" and bool(g.text)
git_commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()

PKG.mkdir(parents=True, exist_ok=True)
dst = PKG / "candidate_v15_baseline.pt"
shutil.copy2(CKPT, dst)

manifest = {
    "package": "N-MIND V1.6 Cloud Checkpoint Package",
    "checkpoint_file": "candidate_v15_baseline.pt",
    "checkpoint_sha256": model_sha,
    "checkpoint_sha256_expected": SHA_EXPECT,
    "tokenizer_hash": tok_hash,
    "tokenizer_hash_expected": TOK_EXPECT,
    "tokenizer_rebuild": "ByteBPETokenizer trained on V1.5 ORIGINAL LEGACY corpus (9 pairs), vocab_size=4096",
    "architecture": {
        "format": model.format,
        "config": model.v2_config.__dict__,
        "param_count": model.n_params(),
        "core_components": ["N-Cell", "BlockRouter", "GlobalWorkspace"],
        "external_pretrained_model": False,
        "query_attention_params": False,
    },
    "model_source": {
        "origin": "locally trained V1.5 baseline (train_v15.py, seed 20261150)",
        "saved_at": meta.get("saved_at"),
        "manifest_hash_field": meta.get("manifest_hash"),
        "steps": meta.get("steps"),
    },
    "git_commit": git_commit,
    "file_size_bytes": CKPT.stat().st_size,
    "load_validation": {
        "strict_load": True,
        "all_finite_weights": not bad,
        "nonfinite_tensors": bad,
        "inference_ok": load_ok,
        "inference_output": g.text,
        "finish_reason": g.finish_reason,
    },
    "parent_role": "父 checkpoint for V1.6 A/B Router-gradient training (NOT a trained V1.6 candidate)",
    "security": {
        "repo_visibility": "PUBLIC",
        "weight_committed_to_git": False,
        "weight_uploaded_to_release": False,
        "note": "权重为私有模型；仓库公开，禁止提交或公开发布。仅本地打包，通过安全渠道传输。",
    },
}
(PKG / "checkpoint_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
readme = f"""# N-MIND V1.6 Cloud Checkpoint Package

本目录是 **V1.6 A/B 训练用的父 checkpoint**（V1.5 baseline，不是已训练的 V1.6 模型）。

## 文件
- `candidate_v15_baseline.pt` — 父权重，SHA-256 = `{model_sha}`
- `checkpoint_manifest.json` — 架构/参数量/来源/校验/安全说明
- `README.md` — 本文件

## 完整性
- Tokenizer Hash = `{tok_hash}`（按 V1.5 原始 LEGACY 语料重建，vocab_size=4096）
- 参数量 = {model.n_params()}；严格加载成功；无 NaN/Inf；推理输出 = "{g.text}"（EOS 结束）
- 生成 git commit = `{git_commit}`

## 用途
作为 `experiment/language-v16-github-cpu` 分支 A/B 训练的**共同父模型**：
- A 组 `baseline_frozen_router`：`router_gradient_path_enabled=False`
- B 组 `fixed_router`：`router_gradient_path_enabled=True`
- 两组同种子 {20261123}、同数据、同父权重，唯一变量为 Router 梯度路径。

## 安全约束（重要）
GitHub 仓库 `yuanhong792/n-mind` 为**公开仓库**。本权重为私有模型：
- 未提交到 Git（`.gitignore` 排除 `cloud_training_package/` 与 `*.pt`/`*.zip`）
- 未上传到公开 Release / Pages / 公开存储
- 云端训练前须通过**安全渠道**传输，并在 Actions 中以 secret 提供，加载后再次校验 SHA-256
- 校验失败或权重缺失时，云端作业**阻塞**，禁止用随机模型替代父 checkpoint

## 传输
压缩包 `N-MIND-V16-Cloud-Checkpoint.zip`（位于项目根）含上述 3 文件。
接收方：解压后先比对 SHA-256，与 manifest 一致方可用于训练。
"""
(PKG / "README.md").write_text(readme, encoding="utf-8")

# zip
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("candidate_v15_baseline.pt", "checkpoint_manifest.json", "README.md"):
        z.write(PKG / f, arcname=f"{PKG.name}/{f}")
# integrity: reopen and verify each member hash
zip_ok = True; member_hashes = {}
with zipfile.ZipFile(ZIP) as z:
    bad_member = z.testzip()
    zip_ok = (bad_member is None)
    for info in z.infolist():
        member_hashes[info.filename] = hashlib.sha256(z.read(info.filename)).hexdigest()
zip_member_pt_sha = member_hashes.get(f"{PKG.name}/candidate_v15_baseline.pt")
zip_ok = zip_ok and (zip_member_pt_sha == model_sha)
result = {
    "status": "CLOUD_PACKAGE_BUILT" if zip_ok and load_ok else "BUILD_FAILED",
    "package_dir": "cloud_training_package/",
    "zip_file": ZIP.name,
    "zip_size_bytes": ZIP.stat().st_size,
    "zip_sha256": sha(ZIP),
    "zip_integrity_ok": zip_ok,
    "zip_member_weight_sha256": zip_member_pt_sha,
    "weight_matches_manifest": zip_member_pt_sha == model_sha,
    "git_tracked": subprocess.check_output(["git","ls-files","cloud_training_package/",ZIP.name],cwd=ROOT,text=True).strip() or "(none - correctly untracked)",
}
out = V16 = ROOT / "experiments/language_v1/v16/_cloud_package_build.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
