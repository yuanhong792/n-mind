"""A vocabulary-closed, combination-held-out binding task generator."""
from __future__ import annotations
import hashlib,itertools,random
from pathlib import Path
from nmind.language.tokenizer import CharTokenizer

ENTITIES=("小明","小红","明小","红小")
POSITIONS=("1号位置","2号位置","3号位置","4号位置")
QUESTIONS=("{e}在哪里？","{e}的位置在哪里？")
ACKS=("","好的。","明白了。","收到。")

def _fp(prompt,target): return hashlib.sha256((prompt+'→'+target).encode()).hexdigest()[:16]
def generate_tasks(n=600,seed=20270201,split='all',tokenizer_path=None):
    rows=[]; seen=set(); rng=random.Random(seed)
    combos=list(itertools.product(ENTITIES,POSITIONS,ENTITIES,POSITIONS))
    combos=[x for x in combos if x[0]!=x[2] and x[1]!=x[3]]; rng.shuffle(combos)
    for a,pa,b,pb in combos:
        for order in (0,1):
            ents=((a,pa),(b,pb)) if not order else ((b,pb),(a,pa))
            distractors=[e for e in ENTITIES if e not in (a,b)]
            full_ents=list(ents)+[(distractors[0],POSITIONS[(POSITIONS.index(pa)+2)%4]),(distractors[1],POSITIONS[(POSITIONS.index(pb)+2)%4])]
            for qent,_qp in full_ents:
                for qfmt in QUESTIONS:
                    for ack in ACKS:
                        ctx=''.join(f'{e}在{p}。' for e,p in full_ents)
                        q=qfmt.format(e=qent); prompt=f'用户：{ctx}\n{ack}用户：{q}\n助手：'
                        target=f'{qent}在{dict(full_ents)[qent]}。'; key=(tuple(sorted((a,pa,b,pb))),qent,qfmt,ack)
                        if key in seen: continue
                        seen.add(key); rows.append({'prompt':prompt,'target':target,'combo':key,'fingerprint':_fp(prompt,target),'query_entity':qent,'question_type':'entity' if qfmt.startswith('{e}在哪里') else 'attribute'})
                        if len(rows)>=n: break
                if len(rows)>=n: break
            if len(rows)>=n: break
        if len(rows)>=n: break
    if tokenizer_path:
        tok=CharTokenizer.load(tokenizer_path); kept=[]
        for r in rows:
            ids=tok.encode(r['prompt']+r['target'])
            if sum(i==3 for i in ids)==0: kept.append(r)
        rows=kept
    return rows

def split_by_combo(rows, test_fraction=.25):
    combos=sorted({r['combo'] for r in rows}); random.Random(20270202).shuffle(combos); cut=max(1,int(len(combos)*test_fraction)); test=set(combos[:cut])
    return [r for r in rows if r['combo'] not in test],[r for r in rows if r['combo'] in test]
