from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from nmind.bot.chat_service import ChatRequest, ChatService
from nmind.bot.knowledge_search import KnowledgeSearch
from nmind.bot.knowledge_store import KnowledgeStore

def main():
    qs=[
      {'id':'G2-01','question':'Python列表和元组有什么区别？','expected_sources':['tutorial/datastructures.html']},
      {'id':'G2-02','question':'Python中for循环和while循环分别适合什么场景？','expected_sources':['tutorial/controlflow.html']},
      {'id':'G2-03','question':'Python如何定义函数？','expected_sources':['tutorial/controlflow.html']},
      {'id':'G2-04','question':'Python模块如何导入？','expected_sources':['tutorial/modules.html']},
      {'id':'G2-05','question':'Python如何处理异常？','expected_sources':['tutorial/errors.html']},
      {'id':'G2-06','question':'Python如何读取文件？','expected_sources':['tutorial/inputoutput.html']},
      {'id':'G2-07','question':'Python中的类和对象是什么关系？','expected_sources':['tutorial/classes.html']},
      {'id':'G2-08','question':'Python字典数据结构有什么特点？','expected_sources':['tutorial/datastructures.html']},
      {'id':'G2-09','question':'range函数如何使用？','expected_sources':['tutorial/controlflow.html']},
      {'id':'G2-10','question':'Python官方教程能否证明N-MIND项目2027年的实验计划？','expected_sources':[]},
    ]
    store=KnowledgeStore(ROOT/'results/online_v1_gate1/knowledge.db'); search=KnowledgeSearch(store)
    svc=ChatService(model=None,tokenizer=None,knowledge=search); rows=[]
    for q in qs:
        t=time.perf_counter(); r=svc.reply(ChatRequest.new('gate2',q['question'])); hits=search.search(q['question'],limit=3)
        urls=[h['citation']['url'] for h in hits]; expected=q.get('expected_sources',[])
        found=(any(x in ' '.join(urls) for x in expected) if expected else r.response_source in ('none','rule'))
        rows.append({'id':q['id'],'question':q['question'],'response':r.text,'response_source':r.response_source,'top3':[h['citation'] for h in hits],'expected_sources':expected,'evidence_found':found,'latency_ms':round((time.perf_counter()-t)*1000,2),'neural_called':r.response_source=='neural'})
    out={'status':'COMPLETE','questions_total':len(rows),'questions':rows,'supported_evidence_rate':sum(x['evidence_found'] for x in rows[:9])/9,'out_of_scope_abstain':rows[9]['response_source'] in ('none','rule'),'optimizer_steps':0}
    for p in (ROOT/'ONLINE_V1_GATE2_CHAT_EVAL.json',ROOT/'results/online_v1_gate2/ONLINE_V1_GATE2_CHAT_EVAL.json'): p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    store.close(); print(json.dumps({'status':'COMPLETE','supported_evidence_rate':out['supported_evidence_rate'],'out_of_scope_abstain':out['out_of_scope_abstain']},ensure_ascii=False))
if __name__=='__main__': main()
