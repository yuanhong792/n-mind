"""MQAR 任务构造（阶段 2.5）：键值关联记忆，词表不扩。

键/值均为现有词表字符组成的 4 字随机名（池各 ≥2048），序列内不重复；
查询：给键答值。两种划分：heldout_seq（同名字池、不同序列）与
heldout_name（测试名字池与训练互不相交）。数据落盘 npz + sha256，
训练器与审计脚本读同一文件，哈希不一致即报错。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys_path = str(ROOT / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from nmind.language.tokenizer import CharTokenizer  # noqa: E402

TOK_PATH = ROOT / "experiments" / "m5_language" / "tokenizer_m52.json"
DATA_DIR = ROOT / "experiments" / "phase2_baselines" / "mqar_data"

N_LIST = (2, 4, 8, 16)
SPLITS = ("heldout_seq", "heldout_name")
POOL_SIZE = 2048
NAME_LEN = 4
TRAIN_N, DEV_N, TEST_N = 20000, 1000, 1000
NAME_CHARS = "小明红号位置哪里收到好了白了"  # 15 个非结构字符，不含 在：。\n？用户助手
DATA_SEED = 20271201
RANDOM_LEVEL = {N: 1.0 / N for N in N_LIST}
SHORTCUT_TOL = 0.05


def load_tok() -> CharTokenizer:
    return CharTokenizer.load(str(TOK_PATH))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gen_names(rng: random.Random, n: int, used: set[str]) -> list[str]:
    out = []
    while len(out) < n:
        s = "".join(rng.choice(NAME_CHARS) for _ in range(NAME_LEN))
        if s not in used:
            used.add(s)
            out.append(s)
    return out


def build_pools() -> dict:
    """四组互不相交的名字池：A(key/val)、B 训练(key/val)、B 测试(key/val)。"""
    rng = random.Random(DATA_SEED)
    used: set[str] = set()
    pools = {}
    for tag in ("a_key", "a_val", "b_key", "b_val", "bt_key", "bt_val"):
        pools[tag] = gen_names(rng, POOL_SIZE, used)
    assert len(used) == 6 * POOL_SIZE
    return pools


def encode_seq(tok: CharTokenizer, ks: list[str], vs: list[str], qi: int) -> tuple[list[int], int]:
    ctx = "\n".join(f"{k}：{v}" for k, v in zip(ks, vs))
    prompt = ctx + f"\n用户：{ks[qi]}？"
    pids = tok.encode(prompt, add_bos=True)
    aids = tok.encode(vs[qi], add_eos=True)
    return pids + aids, len(pids)


def seq_fingerprint(ks, vs, qi) -> tuple:
    return tuple(zip(ks, vs)) + (qi,)


def gen_seqs(rng, keys, vals, n, N, exclude: set) -> list[tuple]:
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


def build_dataset(tok: CharTokenizer, pools: dict, N: int, split: str) -> dict:
    if split == "heldout_seq":
        keys, vals = pools["a_key"], pools["a_val"]
        tkeys, tvals = keys, vals
    else:
        keys, vals = pools["b_key"], pools["b_val"]
        tkeys, tvals = pools["bt_key"], pools["bt_val"]
    rng = random.Random(DATA_SEED + 1000 * N + (0 if split == "heldout_seq" else 7))
    tr = gen_seqs(rng, keys, vals, TRAIN_N, N, set())
    dv = gen_seqs(rng, keys, vals, DEV_N, N, {seq_fingerprint(*s) for s in tr})
    te = gen_seqs(rng, tkeys, tvals, TEST_N, N, {seq_fingerprint(*s) for s in tr})

    def enc_batch(seqs):
        ids, plens = [], []
        for ks, vs, qi in seqs:
            full, plen = encode_seq(tok, ks, vs, qi)
            ids.append(full)
            plens.append(plen)
        assert len(set(plens)) == 1, plens[:5]
        return np.asarray(ids, dtype=np.int32), int(plens[0])

    tr_ids, plen = enc_batch(tr)
    dv_ids, plen_d = enc_batch(dv)
    te_ids, plen_t = enc_batch(te)
    assert plen == plen_d == plen_t

    def cand(seqs):
        c = np.asarray([tok.encode(v) for ks, vs, qi in seqs for v in vs],
                       dtype=np.int32).reshape(len(seqs), N, NAME_LEN)
        t = np.asarray([qi for ks, vs, qi in seqs], dtype=np.int64)
        return c, t

    dv_c, dv_t = cand(dv)
    te_c, te_t = cand(te)

    def swap(seqs):
        ids = []
        ans = []
        for ks, vs, qi in seqs:
            full, plen2 = encode_seq(tok, vs, ks, qi)  # 键值互换，问旧值答旧键
            ids.append(full)
            ans.append(tok.encode(ks[qi]))
        return np.asarray(ids, dtype=np.int32), np.asarray(ans, dtype=np.int32)

    te_sw, te_sw_ans = swap(te)
    L = tr_ids.shape[1]
    assert dv_ids.shape[1] == te_ids.shape[1] == te_sw.shape[1] == L
    return {
        "N": N, "split": split, "prompt_len": plen, "L": L,
        "train": tr_ids, "dev": dv_ids, "test": te_ids,
        "dev_cand": dv_c, "dev_true": dv_t,
        "test_cand": te_c, "test_true": te_t,
        "test_swap": te_sw, "test_swap_ans": te_sw_ans,
        "n_train": TRAIN_N, "n_dev": DEV_N, "n_test": TEST_N,
    }


def parse_seq(tok: CharTokenizer, ids: np.ndarray) -> dict:
    """从 id 序列解析键值对与查询（审计用）。"""
    toks = [tok.id2tok[int(i)] for i in ids]
    assert toks[0] == "<bos>"
    body = toks[1:]
    nl = "\n"
    lines = []
    cur = []
    for t in body:
        if t == nl:
            lines.append(cur)
            cur = []
        else:
            cur.append(t)
    lines.append(cur)
    # 最后一段是 查询行：用户：kq？ 之后可能直接结束（答案段不含）
    pairs = []
    for ln in lines[:-1]:
        s = "".join(ln)
        k, v = s.split("：", 1)
        pairs.append((k, v))
    qln = "".join(lines[-1])
    q = qln[3:] if qln.startswith("用户：") else qln
    kq = q.split("？", 1)[0]
    return {"pairs": pairs, "query_key": kq, "n_pairs": len(pairs)}


def possible_sequences(N: int, pool: int = POOL_SIZE) -> int:
    """有序不重复键序列 × 有序不重复值序列 × N 个查询位。"""
    return math.perm(pool, N) ** 2 * N


def save_dataset(tok: CharTokenizer, ds: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"mqar_N{ds['N']}_{ds['split']}.npz"
    path = out_dir / name
    arrays = {k: v for k, v in ds.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays)
    meta = {k: v for k, v in ds.items() if not isinstance(v, np.ndarray)}
    meta["file"] = name
    meta["sha256"] = sha256_file(path)
    meta["tokenizer"] = str(TOK_PATH)
    meta["tokenizer_vocab_hash"] = tok.vocab_hash()
    meta["vocab_size"] = tok.vocab_size
    meta["name_chars"] = NAME_CHARS
    meta["pool_size"] = POOL_SIZE
    meta["possible_sequences"] = possible_sequences(ds["N"])
    meta["possible_over_train"] = possible_sequences(ds["N"]) / TRAIN_N
    meta["template"] = "<bos>{k1:v1\\n...\\n用户：kq？} + vq<eos>"
    (out_dir / f"mqar_N{ds['N']}_{ds['split']}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_dataset(N: int, split: str, out_dir: Path | None = None) -> tuple[dict, dict]:
    """训练器/审计共用的唯一加载入口：校验 sha256，不一致抛错。"""
    out_dir = out_dir or DATA_DIR
    name = f"mqar_N{N}_{split}.npz"
    path = out_dir / name
    meta = json.loads((out_dir / f"mqar_N{N}_{split}.meta.json").read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if digest != meta["sha256"]:
        raise RuntimeError(f"DATA_HASH_MISMATCH: {name} sha256={digest} != meta {meta['sha256']}")
    z = np.load(path)
    ds = {k: z[k] for k in z.files}
    ds.update({k: v for k, v in meta.items() if k not in ds})
    ds["sha256"] = digest
    return ds, meta
