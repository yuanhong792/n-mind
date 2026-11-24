"""Read-only checkpoint re-evaluation; never writes historical artifacts."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'experiments/m5_language'))
from nmind.language.tokenizer import CharTokenizer
from nmind.language.neural_dialogue import load_lm
from nmind.language.decoder import generate
from nmind.evaluation.phase1_io import validate_metadata
from run_m52 import build_s0, build_s1, SEED, ENTITIES, POSITIONS

def meta(seed, n, eff, data_fp, cfg):
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return {'torch_version':torch.__version__,'torch_build':getattr(torch,'__config__',None).show() if hasattr(getattr(torch,'__config__',None),'show') else 'unknown','python_version':platform.python_version(),'git_revision':rev,'data_fingerprint':data_fp,'config_hash':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16],'seed':seed,'sample_count':n,'effective_sample_count':eff}

def score(model,tok,prompt,answer):
    p=tok.encode(prompt); a=tok.encode(answer,add_eos=False); ids=p+a
    with torch.no_grad(): logits,_=model(torch.tensor([ids],dtype=torch.long))
    lp=torch.log_softmax(logits[0],-1); vals=[float(lp[len(p)-1+i,tid]) for i,tid in enumerate(a)]
    return vals

def run_one(path, tok, prompts, answers, candidates):
    model, ckmeta=load_lm(path,tok); model.eval(); rows=[]
    for p,a in zip(prompts,answers):
        vals=score(model,tok,p,a); cand=[sum(score(model,tok,p,c)) for c in candidates]
        ids=tok.encode(p)
        with torch.no_grad(): logits,st=model(torch.tensor([ids],dtype=torch.long))
        rec,_=generate(model,tok,tok.encode('助手：'),state=st,max_new_tokens=24,greedy=True)
        rows.append({'prompt':p,'target':a,'answer_token_logprobs':vals,'answer_log_likelihood':sum(vals),'candidate_scores':cand,'candidate_rank':1+sum(x>sum(vals) for x in cand),'nll':-sum(vals)/max(1,len(vals)),'generated':rec.text.strip(),'em':rec.text.strip()==a.strip()})
    return rows,ckmeta

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-items',type=int,default=80); a=ap.parse_args()
    here=ROOT/'experiments/m5_language'; tok=CharTokenizer.load(here/'tokenizer_m52.json')
    tasks={}
    for name,builder,seed in [('s0',build_s0,SEED+2),('s1',build_s1,SEED+5)]: tasks[name]=builder(a.max_items,seed)
    models={'stable_s0':here/'m52_s0_checkpoint.pt','s1':here/'m52_s1_checkpoint.pt','gateb001':ROOT/'experiments/bot_runtime_r2/candidates/rejected/gateb001.pt','gateb002':ROOT/'experiments/bot_runtime_r2/candidates/rejected/002.pt','gateb003':ROOT/'experiments/bot_runtime_r2/candidates/rejected/003.pt'}
    outdir=ROOT/'experiments/phase1_reeval'/datetime.now().strftime('%Y%m%d'); outdir.mkdir(parents=True,exist_ok=True)
    for label,path in models.items():
        if not path.exists(): continue
        block={}
        for fam,(p,ans) in tasks.items():
            c=[f'{e}在{x}。' for e in ENTITIES for x in POSITIONS if e==ans[0][:3] or True]
            rows,ck=run_one(path,tok,p,ans,c); fp=hashlib.sha256(('\n'.join(p)+'\n'+'\n'.join(ans)).encode()).hexdigest()[:16]
            md=meta(0,len(rows),len(rows),fp,{'family':fam,'protocol':'freegen-v1'}); validate_metadata(md)
            block[fam]={'metadata':md,'checkpoint_metadata':ck,'samples':rows}
        (outdir/f'{label}.json').write_text(json.dumps(block,ensure_ascii=False,indent=2),encoding='utf8')
    print(outdir)
if __name__=='__main__': main()
