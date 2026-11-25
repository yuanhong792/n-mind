"""FLY 数据入口：复刻 MQAR 的唯一哈希校验加载逻辑，不改 src/nmind。

本模块不导入 mqar_task/nmind/torch，直接在文件层读取 npz + meta 并校验 SHA-256，
避免 FLY 机制测试被训练栈依赖拖住；哈希口径与 mqar_task.load_dataset 完全一致。
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_MQAR_DIR = ROOT / "experiments" / "phase2_baselines" / "mqar_data"
FLY_SEAL_DIR = HERE / "data" / "seal2_like"
TOK_PATH = ROOT / "experiments" / "m5_language" / "tokenizer_m52.json"

NAME_CHARS = "小明红号位置哪里收到好了白了"
NAME_LEN = 4
POOL_SIZE = 2048
TEST_N = 1000
DATA_SEED = 20271201


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_vocab():
    blob = json.loads(TOK_PATH.read_text(encoding="utf-8"))
    vocab = blob["vocab"]
    id2tok = {int(v): k for k, v in vocab.items()}
    return vocab, id2tok, blob.get("vocab_hash")


def load_mqar(N: int, split: str, out_dir: Path | None = None):
    """与 mqar_task.load_dataset 相同的哈希校验入口。"""
    out_dir = out_dir or DEFAULT_MQAR_DIR
    meta_path = out_dir / f"mqar_N{N}_{split}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fname = meta.get("file") or f"mqar_N{N}_{split}.npz"
    path = out_dir / fname
    digest = sha256_file(path)
    if digest != meta["sha256"]:
        raise RuntimeError(f"DATA_HASH_MISMATCH: {path.name} sha256={digest} != meta {meta['sha256']}")
    z = np.load(path)
    ds = {k: z[k] for k in z.files}
    ds.update({k: v for k, v in meta.items() if k not in ds})
    ds["sha256"] = digest
    return ds, meta


def parse_ids(ids: np.ndarray):
    """从 token id 序列解析键值对与查询；与 mqar_task.parse_seq 同口径。"""
    _, id2tok, _ = _load_vocab()
    toks = [id2tok[int(i)] for i in ids]
    if toks[0] != "<bos>":
        raise RuntimeError("MQAR 序列缺少 <bos>")
    lines, cur = [], []
    for t in toks[1:]:
        if t == "\n":
            lines.append(cur)
            cur = []
        else:
            cur.append(t)
    lines.append(cur)
    pairs = []
    for ln in lines[:-1]:
        s = "".join(ln)
        k, v = s.split("：", 1)
        pairs.append((k, v))
    qln = "".join(lines[-1])
    q = qln[3:] if qln.startswith("用户：") else qln
    kq = q.split("？", 1)[0]
    return {"pairs": pairs, "query_key": kq, "n_pairs": len(pairs)}


def seq_to_key_values(ids: np.ndarray):
    parsed = parse_ids(ids)
    ks = [k for k, _ in parsed["pairs"]]
    vs = [v for _, v in parsed["pairs"]]
    q = parsed["query_key"]
    if q not in ks:
        raise RuntimeError(f"查询键不在序列中: {q}")
    return ks, vs, ks.index(q)


def seq_fingerprint(ks, vs, qi):
    return tuple(zip(ks, vs)) + (qi,)


def decode_candidates(arr: np.ndarray) -> list[list[str]]:
    _, id2tok, _ = _load_vocab()
    return [["".join(id2tok[int(t)] for t in row if t not in {"<bos>", "<eos>"})
             for row in seq] for seq in arr]


def examples_from_dataset(ds: dict, limit: int = 200):
    ids = ds["test"][:limit]
    cand_tokens = ds["test_cand"][:limit]
    true = ds["test_true"][:limit]
    cands = decode_candidates(cand_tokens)
    out = []
    for i in range(len(ids)):
        ks, vs, qi = seq_to_key_values(ids[i])
        out.append({
            "pairs": list(zip(ks, vs)),
            "query_key": ks[qi],
            "candidates": cands[i],
            "true_idx": int(true[i]),
        })
    return out


def gen_names(rng: random.Random, n: int, used: set[str]) -> list[str]:
    out = []
    while len(out) < n:
        s = "".join(rng.choice(NAME_CHARS) for _ in range(NAME_LEN))
        if s not in used:
            used.add(s)
            out.append(s)
    return out


def build_pools():
    rng = random.Random(DATA_SEED)
    used: set[str] = set()
    return {tag: gen_names(rng, POOL_SIZE, used)
            for tag in ("a_key", "a_val", "b_key", "b_val", "bt_key", "bt_val")}


def gen_seqs(rng, keys, vals, n, N, exclude):
    out, seen = [], set()
    while len(out) < n:
        ks = rng.sample(keys, N)
        vs = rng.sample(vals, N)
        qi = rng.randrange(N)
        fp = seq_fingerprint(ks, vs, qi)
        if fp in exclude or fp in seen:
            continue
        seen.add(fp)
        out.append((ks, vs, qi))
    return out


def encode_seq(ks, vs, qi):
    vocab, _, _ = _load_vocab()
    ctx = "\n".join(f"{k}：{v}" for k, v in zip(ks, vs))
    prompt = ctx + f"\n用户：{ks[qi]}？"
    pids = [vocab["<bos>"]] + [vocab[ch] for ch in prompt]
    aids = [vocab[ch] for ch in vs[qi]] + [vocab["<eos>"]]
    return pids + aids, len(pids)


def generate_seal2_like(N: int, seed: int, out_dir: Path = FLY_SEAL_DIR):
    """生成 seal2-like 封存集（仅 N=8/16 使用），排除既有 train/dev/test 指纹。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    base, _ = load_mqar(N, "heldout_seq")
    exclude = set()
    for part in ("train", "dev", "test"):
        for ids in base[part]:
            ks, vs, qi = seq_to_key_values(ids)
            exclude.add(seq_fingerprint(ks, vs, qi))

    pools = build_pools()
    rng = random.Random(seed)
    seqs = gen_seqs(rng, pools["a_key"], pools["a_val"], TEST_N, N, exclude)

    ids, plens = [], []
    for ks, vs, qi in seqs:
        full, plen = encode_seq(ks, vs, qi)
        ids.append(full)
        plens.append(plen)
    if len(set(plens)) != 1:
        raise RuntimeError(f"seal2-like 序列长度不一致: {plens[:5]}")
    arr_ids = np.asarray(ids, dtype=np.int32)
    vocab, _, _ = _load_vocab()
    cand = np.asarray(
        [[vocab[ch] for ch in v] for ks, vs, qi in seqs for v in vs], dtype=np.int32
    ).reshape(len(seqs), N, NAME_LEN)
    true = np.asarray([qi for ks, vs, qi in seqs], dtype=np.int64)

    ds = {
        "N": N,
        "split": "heldout_seq_seal2_like",
        "prompt_len": int(plens[0]),
        "L": int(arr_ids.shape[1]),
        "test": arr_ids,
        "test_cand": cand,
        "test_true": true,
        "n_test": len(seqs),
    }
    name = f"mqar_N{N}_heldout_seq_seal2_like.npz"
    path = out_dir / name
    np.savez_compressed(path, **{k: v for k, v in ds.items() if isinstance(v, np.ndarray)})
    meta = {k: v for k, v in ds.items() if not isinstance(v, np.ndarray)}
    meta.update({
        "file": name,
        "sha256": sha256_file(path),
        "seed": seed,
        "excluded_fingerprints": len(exclude),
        "seq_overlap_with_train_dev_test": 0,
    })
    (out_dir / f"mqar_N{N}_heldout_seq_seal2_like.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded, loaded_meta = load_mqar(N, "heldout_seq_seal2_like", out_dir)
    return {"path": str(path), "sha256": meta["sha256"], "loaded_sha256": loaded["sha256"],
            "seed": seed, "N": N, "n_test": len(seqs),
            "excluded_fingerprints": len(exclude), "meta": loaded_meta}
