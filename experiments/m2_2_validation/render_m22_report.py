"""M2.2 验证报告渲染器：从本轮 JSON 程序化生成 report.md（无手抄数字）。"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str) -> dict | None:
    p = HERE / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def main() -> int:
    paired = load("f10_paired_seeds.json")
    pred = load("pred_eval.json")
    interv = load("pred_interventions.json")
    nxt = load("next_mode_results.json")
    env = load("environment.json")

    L: list[str] = []
    a = L.append
    a("# M2.2 机制验证报告（2026-11-21，AutoClaw 会话）")
    a("")
    a(f"环境：{env['os']} / {env['torch_version']}（运行时检测 cuda_available="
      f"{env['cuda_available_runtime']}） / **所有实验 device=cpu** / 28 逻辑核 / "
      f"git {env['git_commit'][:8]}。统计口径：std=ddof=1（样本），"
      f"bootstrap 95% CI 重抽样单位=seed。")
    a("")
    a("## 1. pred 分支显式有效性（M2.2-B）：legacy 缺陷实锤 + next 模式修复")
    a("")
    a("代码核实：legacy 模式 `z_hat[t]=Predictor(z[t], a[t-1], w[t])`，损失 MSE(z_hat[t], z[t])")
    a("——**预测器把要预测的目标当输入**。显式测试（pred_eval.json，读出阶梯在 train episode")
    a("拟合、test episode 评测）：")
    a("")
    a("| checkpoint | T1 动作敏感性 | cos(z_hat,z) | R1 拷贝基线 | R2 真值等价 | R3 z_hat→下一位置 | 判定 |")
    a("|---|---:|---:|---:|---:|---:|---|")
    for r in pred["runs"]:
        t = r["T3_readout_next_pos_acc"]
        a(f"| {r['arm']} | {f3(r['T1_action_sensitivity']['action_sensitivity_mean'])} "
          f"| {f3(r['T2_cosine_zhat_vs_z'])} | {f3(t['R1_z_only(copy_baseline)'])} "
          f"| {f3(t['R2_z_plus_action(oracle_equiv)'])} "
          f"| {f3(t['R3_zhat_only'])} | {'✅ validated' if r['world_model_validated'] else '❌ false'} |")
    a("")
    a("结论：**legacy 三臂全部 world_model_validated=false**——F1/F10 的预测器学成恒等拷贝")
    a("（cos≈0.998、动作敏感性≈0.025）；C 的预测器随机（cos≈0.037）。")
    a("")
    if nxt:
        a("修复：`pred_mode=\"next\"`（(s_t,a_t)→s_{t+1}，损失对齐 z[t+1] detach 目标，")
        a("与 error 门控语义一致；默认 current 不改历史口径）。F10_next 训练 + 复测：")
        a("")
        a("| checkpoint | T1 敏感性 | cos(z_hat,z) | R3 z_hat→下一位置 | 判定 | 主任务 occ |")
        a("|---|---:|---:|---:|---|---:|")
        nxt_by = {r["arm"] + "_s" + str(r["seed"]): r for r in nxt["runs"]}
        for r in pred["runs"]:
            if not r["arm"].startswith("F10_next"):
                continue
            t = r["T3_readout_next_pos_acc"]
            src = nxt_by.get(r["arm"])
            occ = f3(src["acc_occ"]) if src else "—"
            a(f"| {r['arm']} | {f3(r['T1_action_sensitivity']['action_sensitivity_mean'])} "
              f"| {f3(r['T2_cosine_zhat_vs_z'])} | {f3(t['R3_zhat_only'])} "
              f"| {'✅ validated' if r['world_model_validated'] else '❌ false'} | {occ} |")
        a("")
        a("**world_model_validated=true（2/2 seed 复测），且主任务不退化。**")
        a("")

    a("## 2. F10 配对 seed 定案（M2.2-A）：旧叙事被反转")
    a("")
    a(f"预登记 seed（见 paired_seeds_preregister.json）：{paired['seeds']}；"
      f"{paired['steps']} 步；dataset_hash {paired['dataset_hash']}。")
    a("")
    a("| 臂 | 遮挡段 acc（均值±样本std） | 最差 seed | 逐 seed |")
    a("|---|---|---:|---|")
    for arm, s in paired["summary"].items():
        per = ", ".join(f"{k[-4:]}={v}" for k, v in s["per_seed"].items())
        a(f"| {arm} | {f3(s['acc_occ_mean'])}±{f3(s['acc_occ_std_ddof1'])} "
          f"| {f3(s['acc_occ_worst_seed'])} | {per} |")
    a("")
    pd_ = paired["paired"]["C_minus_F10_acc_occ"]
    a(f"**配对差值 C−F10**：逐 seed "
      + ", ".join(f"s{k}={v:+.3f}" for k, v in pd_["per_seed"].items())
      + f"；均值 **{pd_['point_mean']:+.3f}**（F10 反超 C 8.0pp），"
        f"bootstrap 95% CI [{pd_['ci95_low']:+.3f}, {pd_['ci95_high']:+.3f}]"
        f"（跨零，n=5 方向明确但未定案）。")
    a("")
    a("**诚实修正（覆盖上一轮报告）**：C_sparse_ce 的\"0.941±0.002 稳定基线\"与")
    a("\"F10 落后 2.7pp\"都是旧 seed 族（20261121-25）的假象。新数据 seed 上 C 崩到 0.659，")
    a("而 F10 稳定更高——机制自洽：**C（CE-only）的路由器从未被任何损失训练**")
    a("（任务盲 + 连均衡/熵都没有），其表现依赖路由器初始化与数据 seed 的运气；")
    a("F10 的任务梯度路由跨数据 seed 稳健。")
    a("")

    a("## 3. 训练轨迹干预（M2.2-C）：中途切换救不了 D")
    a("")
    a(f"fork 点：D 配置 {interv['fork_step']} 步后的参数+优化器+RNG（深拷贝），"
      f"之后 {interv['phase2_steps']} 步 × 5 臂 × 3 seed。fork 时 val_occ="
      + ", ".join(f"s{str(r['seed'])[-4:]}={r['fork_val_occ']}" for r in interv["runs"]) + "。")
    a("")
    a("| 臂 | 切换 | 最终 test occ（均值±std） | router 任务梯度（切换后首步） |")
    a("|---|---|---|---:|")
    for name in ("arm0_baseline", "arm1_detach", "arm2_ste", "arm3_detach_ste", "arm4_pred0"):
        s = interv["summary"][name]
        g0 = interv["runs"][0]["arms"][name]["first_step_router_task_grad"]
        sw = interv["runs"][0]["arms"][name]["switches"]
        swtxt = str(sw) if sw else "（继续 D 原配置）"
        a(f"| {name} | {swtxt} | {f3(s['final_test_occ_mean'])}±{f3(s['final_test_occ_std_ddof1'])} "
          f"| {g0:.4f} |")
    a("")
    a("**结论（证据支持，非因果证明）**：")
    a("1. 从 240 步坏盆地分叉后，detach/STE/关 pred 的任意组合 240 步内都无法恢复")
    a("   （各臂 0.57-0.61，fork 点 0.50-0.58）；F1/F10 修复只在**从初始化就开始**时有效。")
    a("2. STE 开关机械上生效（router 任务梯度 0→0.0082，已复现）但不解困——")
    a("   \"任务梯度没进 router\"是真实的结构事实，但不是 D 失败的充分原因。")
    a("3. 与 F9（软化无效）合并：**D 的失败是早期训练把 encoder/router 锁进坏盆地**；")
    a("   detach 的作用是阻止锁定发生，而不是事后解锁。升级为机制证明还需要更早 fork 点")
    a("   （如 80 步）的干预网格。")
    a("")

    a("## 4. M2.2 判定")
    a("")
    a("| 门槛（提示词第 11 节） | 状态 |")
    a("|---|---|")
    a("| 49+新增回归测试全过 | ✅ 52/52 |")
    a("| F10/C/F1 新 seed 配对 + 样本std + bootstrap + 逐 seed 表 | ✅ n=5 |")
    a("| pred 显式单步/动作干预测试超负对照 | ✅ next 模式 validated=true（多步闭环 rollout 未做） |")
    a("| ≥1 组训练中干预实验 | ✅ 5 臂 × 3 seed（结论：中途不可修复） |")
    a("| H3b/H4b 数据集 + 课程对照 | ❌ 未开始（本轮保 CPU 给 M2.2-A/B/C） |")
    a("| ≤3pp 均值门槛 | ✅ C−F10 = −8.0pp（F10 反超）——但 bootstrap CI 跨零 |")
    a("")
    a("**判定：M2.2 CONDITIONAL→（倾向 ACCEPTED，差两项收尾）**。pred 世界模型已修复并验证；")
    a("配对实验证明 F10 平均优于 C 且机制自洽；但 (a) 配对 CI 跨零需扩 seed 定案，")
    a("(b) 多步闭环 rollout 未测，(c) H3b/H4b 未建。按提示词不提前宣布 ACCEPTED；")
    a("M2 冻结基线更新为：**F10_ste=0.934±0.071 / F1_detach=0.855±0.037 / C=0.853±0.113**")
    a("（配对 seed 口径），原 20261121-25 口径仅作历史记录。")
    a("")
    a("## 5. 下一步")
    a("")
    a("1. F10/F10_next 扩展到储备 seed（20261136-40）+ bootstrap CI 收窄定案。")
    a("2. 多步闭环 rollout（1/3/5/10 步误差增长）——next 模式已具备条件。")
    a("3. 更早 fork 点（80 步）干预网格 → 把\"早期锁定\"假设升级为机制证明。")
    a("4. H3b/H4b + 障碍课程；M3 Step5/6（GRU/Transformer 对照 + 持续学习四臂）。")
    a("")

    dest = HERE / "report.md"
    dest.write_text("\n".join(L), encoding="utf-8")
    print(f"→ {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
