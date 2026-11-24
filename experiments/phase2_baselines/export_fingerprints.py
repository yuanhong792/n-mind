from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'experiments/m5_language'))
from run_m52 import build_s0,build_s1,SEED
from nmind.evaluation.phase1_archive import generate_archive
def fp(s): return hashlib.sha256(s.encode()).hexdigest()[:16]
def main():
    out=ROOT/'experiments/phase2_baselines/fingerprints_20261124'; out.mkdir(parents=True,exist_ok=True); all_rows=[]
    for name,builder,seed in [('m52_s0_train',build_s0,SEED),('m52_s0_sealed',build_s0,SEED+2),('m52_s1_train',build_s1,SEED),('m52_s1_sealed',build_s1,SEED+5)]:
        p,a=builder(400 if 'train' in name else 80,seed); rows=sorted({fp(x+'→'+y) for x,y in zip(p,a)}); (out/f'{name}.json').write_text(json.dumps(rows,indent=2),encoding='utf8'); all_rows += [(name,x) for x in rows]
    arch,audit=generate_archive(families=('position_qa',),n_each=500,seed=20262401); rows=sorted({t.fingerprint for t in arch['position_qa']}); (out/'new_archive_position_qa.json').write_text(json.dumps(rows,indent=2),encoding='utf8'); all_rows += [('new_archive_position_qa',x) for x in rows]
    master={'generated':'2026-11-24','rows':all_rows,'unique_total':len(set(x[1] for x in all_rows)),'archive_audit':audit}
    (out/'MASTER.json').write_text(json.dumps(master,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps({'unique_total':master['unique_total'],'archive_overlap':audit['overlap_count']},indent=2))
if __name__=='__main__': main()
