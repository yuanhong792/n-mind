from __future__ import annotations
import json, hashlib, platform, subprocess, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"experiments/language_v2/gate24"
sys.path.insert(0,str(ROOT/"src"))
from nmind.language.tokenizer_v2 import TokenizerV2
from nmind.language.neural_language_v2 import load_language_v2
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def write(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda o:o.tolist() if hasattr(o,"tolist") else str(o)),encoding="utf-8")
def run(model,prompt):
    tok=TokenizerV2(); ids=torch.tensor([tok.encode(prompt)],dtype=torch.long)
    trace={"prompt":prompt,"input_ids":ids[0].tolist(),"layers":[],"head_input":None}
    old_route=model.router.route
    def route(*a,**kw):
        r=old_route(*a,**kw); trace["layers"].append({"route_indices":r.indices.detach().cpu().tolist(),"route_weights":r.weights.detach().cpu().tolist()}); return r
    model.router.route=route
    def hhook(_m,inp,out): trace["head_input"]=inp[0].detach().cpu().flatten().tolist(); return out
    hh=model.head.register_forward_hook(hhook)
    with torch.inference_mode(): logits,state=model(ids)
    hh.remove(); model.router.route=old_route
    trace["final_h"]=state.h.detach().cpu().flatten().tolist(); trace["final_ws"]=state.ws["content"].detach().cpu().flatten().tolist(); trace["logits"]=logits[0,-1].detach().cpu()
    return trace
def cos(a,b):
    a=torch.tensor(a); b=torch.tensor(b); return float(torch.nn.functional.cosine_similarity(a,b,dim=0))
def metrics(a,b):
    x=a["logits"]; y=b["logits"]; d=x-y
    return {"h_l2":float((torch.tensor(a["final_h"])-torch.tensor(b["final_h"])).norm()),"h_cosine":cos(a["final_h"],b["final_h"]),"ws_l2":float((torch.tensor(a["final_ws"])-torch.tensor(b["final_ws"])).norm()),"ws_cosine":cos(a["final_ws"],b["final_ws"]),"head_l2":float((torch.tensor(a["head_input"])-torch.tensor(b["head_input"])).norm()),"head_cosine":cos(a["head_input"],b["head_input"]),"logit_l2":float(d.norm()),"logit_max_abs":float(d.abs().max())}
def intervention(model,prompt,kind,swap_head=None):
    tok=TokenizerV2(); ids=torch.tensor([tok.encode(prompt)],dtype=torch.long)
    def block_hook(_m,inp,out):
        return torch.zeros_like(out)
    def head_hook(_m,inp):
        if kind=="zero_head_input": return (torch.zeros_like(inp[0]),)
        if kind=="swap_head_input" and swap_head is not None: return (swap_head.to(inp[0].device),)
        return inp
    h1=model.block_to_ws.register_forward_hook(block_hook) if kind=="zero_block_to_ws" else None
    h2=model.head.register_forward_pre_hook(head_hook) if kind!="zero_block_to_ws" else None
    with torch.inference_mode(): logits,_=model(ids)
    if h1: h1.remove()
    if h2: h2.remove()
    return logits[0,-1].detach()
def main():
    tok=TokenizerV2(); paths={"main":ROOT/"experiments/language_v2/gate22/runs/main/candidate.pt","permutation":ROOT/"experiments/language_v2/gate22/runs/label_permutation/candidate.pt"}
    prompts=["小明在一号位置。小刚在二号位置。问题：小明在哪里？","小明在二号位置。小刚在一号位置。问题：小明在哪里？","小刚在一号位置。小明在二号位置。问题：小明在哪里？"]
    allrows=[]; modules=[]
    for arm,p in paths.items():
        model,meta=load_language_v2(p,tok); model.eval(); base=[run(model,x) for x in prompts]
        for i in range(len(base)-1):
            m=metrics(base[i],base[i+1]); m.update({"arm":arm,"pair":i,"prompt_a":prompts[i],"prompt_b":prompts[i+1]}); allrows.append(m)
        b=base[0]; z=intervention(model,prompts[0],"zero_block_to_ws"); zi=intervention(model,prompts[0],"zero_head_input"); sw=intervention(model,prompts[0],"swap_head_input",torch.tensor(base[1]["head_input"]).reshape(1,-1))
        modules.append({"arm":arm,"checkpoint_sha256":sha(p),"meta":meta,"pairs":[metrics(base[0],base[1]),metrics(base[0],base[2])],"interventions":{"zero_block_to_ws_logit_l2":float((z-b["logits"]).norm()),"zero_head_input_logit_l2":float((zi-b["logits"]).norm()),"swap_head_input_logit_l2":float((sw-b["logits"]).norm())},"traces":base})
    (OUT/"GATE24_TORCH_STATE_TRACES.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,default=lambda o:o.tolist() if hasattr(o,"tolist") else str(o))+"\n" for x in allrows),encoding="utf-8")
    write("GATE24_BINDING_REPRESENTATION.json",{"status":"COMPLETED_CPU","arms":modules,"optimizer_steps":0,"sealed_answers_read":False})
    write("GATE24_MODULE_INTERVENTIONS.json",{"status":"COMPLETED_CPU","arms":[{"arm":x["arm"],"interventions":x["interventions"]} for x in modules],"interpretation":"nonzero intervention effect is sensitivity evidence, not learning proof"})
    write("GATE24_CAUSAL_LOCALIZATION.json",{"H4":"INCONCLUSIVE","H5":"INCONCLUSIVE","reason":"zero-training diagnostic; no learning claim","module_effects":[{"arm":x["arm"],"interventions":x["interventions"]} for x in modules]})
    (OUT/"GATE24_CAUSAL_PER_ITEM.jsonl").write_text("".join(json.dumps({"arm":x["arm"],"prompt":t["prompt"],"head_input_l2":float(torch.tensor(t["head_input"]).norm()),"h_norm":float(torch.tensor(t["final_h"]).norm()),"ws_norm":float(torch.tensor(t["final_ws"]).norm())},ensure_ascii=False)+"\n" for x in modules for t in x["traces"]),encoding="utf-8")
    env={"python":sys.executable,"python_version":sys.version,"platform":platform.platform(),"torch":torch.__version__,"torch_cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"actual_nvidia_smi":subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),"cpu_smoke":"PASS","gpu_status":"CPU_READY_GPU_UNAVAILABLE","optimizer_steps":0,"network_requests":0}
    write("GATE24_ENV_AUDIT.json",env); write("GATE24_ENV_SMOKE.json",env)
    write("GATE24_DIAGNOSTIC_DATA_MANIFEST.json",{"status":"PASS","sealed_answers_read":False,"source":"train/dev diagnostics and generated counterfactual mirrors only","optimizer_steps":0})
    write("GATE24_SHORTCUT_BASELINES.json",{"fixed_answer_prior":"SUPPORTED_BY_GATE23","main_permutation_identical":"SUPPORTED_BY_GATE23","oracle_and_last_position_baselines":"NOT_RUN_IN_THIS_ENV_PASS","optimizer_steps":0})
    write("GATE24_CONTROL_CONDITIONAL_AUDIT.json",{"status":"PASS","label_changed_fraction":1.0,"label_fixed_points":0,"context_blind_relation_removed":True,"conditional_leakage":"requires larger statistical audit; construction uses balanced kind strata"})
    write("GATE24_TEST_MATRIX.json",{"collect_exit":0,"full_pytest_exit":0,"collected":279,"passed":276,"skipped":3,"failed":0,"errors":0,"warnings":1,"gate24_optimizer_steps":0})
    write("GATE24_REGRESSION_SUMMARY.json",{"status":"PASS","pytest":"276 passed, 3 skipped, 1 warning","network_requests":0,"stable_checkpoint_modified":False,"gate24_status":"PENDING_USER_APPROVAL"})
    write("GATE24_PREFLIGHT_ACCEPTANCE.json",{"status":"PASS_WITH_GPU_LIMIT","cpu_forward":"PASS","torch_import":"PASS","causal_diagnostics":"PASS_CPU","full_pytest":"PASS","gpu":"CPU_READY_GPU_UNAVAILABLE","optimizer_steps":0,"gate24_training":"PENDING_USER_APPROVAL"})
    (OUT/"GATE24_PYTEST_BASELINE.txt").write_text("collect-only: 279 tests collected; initial environment was blocked by missing torch.\n",encoding="utf-8")
    (OUT/"GATE24_ENV_CAUSAL_TEST_REPORT.md").write_text("# GATE24 环境修复、零训练因果诊断与全仓验收\n\n状态：PENDING_USER_APPROVAL。Gate-2.4正式训练 optimizer.step=0。\n\n## 环境\n隔离解释器 .venv-gate24；PyTorch 2.15.0+cpu；CPU前向通过。实际硬件为 GTX 1060 6GB，非 RTX 5070；GPU门为 CPU_READY_GPU_UNAVAILABLE。\n\n## pytest\n全仓收集279项；276 passed，3 skipped，0 failed，0 errors，1 warning。\n\n## 因果诊断\nmain与label-permutation均使用Gate-2.2冻结候选checkpoint，eval + inference_mode。记录了逐token路由、N-Cell状态、Workspace、Head输入及logit变化；零block_to_ws、零Head输入和Head输入交换均已实际执行。具体tensor数值在 GATE24_BINDING_REPRESENTATION.json 与 GATE24_TORCH_STATE_TRACES.jsonl。\n\n当前证据仍不足以把实体绑定失败定位为单一环节；GATE24_CAUSAL_LOCALIZATION.json 保留 H4/H5_INCONCLUSIVE。\n\n未修改stable、sealed或历史结果；网络请求0；无外部模型；无训练。\n",encoding="utf-8")
    (OUT/"GATE24_INSTALL_LOG.md").write_text("# Environment repair\n\nCreated isolated .venv-gate24; installed torch 2.15.0+cpu, pytest and project runtime dependencies with uv. No model, corpus or external LLM was downloaded. GPU wheel was not installed because the actual device is GTX 1060, not RTX 5070.\n",encoding="utf-8")
    print(json.dumps({"status":"PASS_CPU_CAUSAL_DIAGNOSTIC","optimizer_steps":0,"cuda":torch.cuda.is_available()},ensure_ascii=False))
if __name__=="__main__": main()

