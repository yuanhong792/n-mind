import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nmind.bot.chat_service import ChatRequest,ChatService
from nmind.bot.knowledge_search import KnowledgeSearch
from nmind.bot.knowledge_store import KnowledgeStore
Q=[('R01','try和except如何配合处理错误？'),('R02','json.dumps通常把什么转换成什么？'),('R03','Path.exists()用于检查什么？'),('R04','logging中的日志级别有什么作用？'),('R05','集合和列表在成员关系检查上有什么不同？')]
s=KnowledgeStore(ROOT/'results/online_v1_gate1/knowledge.db');k=KnowledgeSearch(s);svc=ChatService(model=None,tokenizer=None,knowledge=k);rows=[]
for qid,q in Q:
    r=svc.reply(ChatRequest.new('g31',q)); rows.append({'id':qid,'question':q,'response_source':r.response_source,'citations':[c.to_dict() for c in r.citations],'has_evidence':bool(r.citations)})
out={'status':'COMPLETE','questions_total':len(rows),'questions':rows,'evidence_rate':sum(x['has_evidence'] for x in rows)/len(rows),'optimizer_steps':0}
(ROOT/'ONLINE_V1_GATE31_GENERALIZATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');s.close();print(json.dumps({'evidence_rate':out['evidence_rate']},ensure_ascii=False))
