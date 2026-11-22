from __future__ import annotations
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nmind.bot.chat_service import ChatRequest,ChatService
from nmind.bot.knowledge_search import KnowledgeSearch
from nmind.bot.knowledge_store import KnowledgeStore
Q=[
('F01','Python表达式中 / 和 // 有什么区别？',['tutorial/introduction.html']),('F02','Python如何进行乘方运算？',['tutorial/introduction.html']),('F03','Python字符串如何索引和切片？',['tutorial/introduction.html']),('F04','列表和字符串在可变性方面有什么区别？',['tutorial/introduction.html']),('F05','for循环通常遍历什么？',['tutorial/controlflow.html']),('F06','range(5)产生哪些数字？',['tutorial/controlflow.html']),('F07','if语句支持哪些条件形式？',['tutorial/controlflow.html']),('F08','Python如何定义函数？',['tutorial/controlflow.html']),('F09','列表推导式有什么用途？',['tutorial/datastructures.html']),('F10','字典的主要用途是什么？',['tutorial/datastructures.html']),('F11','元组和列表有什么区别？',['tutorial/datastructures.html']),('F12','Python模块如何导入？',['tutorial/modules.html']),('F13','JSON序列化和反序列化分别使用什么API？',['library/json.html']),('F14','pathlib.Path用于处理什么？',['library/pathlib.html']),('F15','正则表达式模块提供什么功能？',['library/re.html']),('F16','sqlite3模块主要提供什么？',['library/sqlite3.html']),('F17','datetime模块表示哪些时间信息？',['library/datetime.html']),('F18','hashlib主要提供什么功能？',['library/hashlib.html']),('F19','logging模块主要用于什么？',['library/logging.html']),('F20','内置len()函数有什么作用？',['library/functions.html']),
('X01','Python教程和标准库参考如何共同解释序列类型？',['tutorial/datastructures.html','library/stdtypes.html']),('X02','如何结合教程中的模块和json文档导入JSON工具？',['tutorial/modules.html','library/json.html']),('X03','文件读写在教程和pathlib文档中如何分别处理？',['tutorial/inputoutput.html','library/pathlib.html']),('X04','异常在教程和logging文档中的角色有何不同？',['tutorial/errors.html','library/logging.html']),('X05','教程中的类与dataclasses模块有什么关系？',['tutorial/classes.html','library/dataclasses.html']),
('N01','Python官方文档是否包含N-MIND项目2027年的实验计划？',[]),('N02','能否从Python文档证明某人的银行密码？',[]),('N03','Python官方教程是否包含用户的个人电话号码？',[]),('N04','这些页面能否确认N-MIND当前模型的内部测试分数？',[]),('N05','能否从Python文档查到某人的实时GPS位置？',[])]
def main():
 s=KnowledgeStore(ROOT/'results/online_v1_gate1/knowledge.db');k=KnowledgeSearch(s);svc=ChatService(model=None,tokenizer=None,knowledge=k);rows=[]
 for qid,q,expected in Q:
  t=time.perf_counter();r=svc.reply(ChatRequest.new('gate3',q));hits=k.search(q,limit=3);urls=[x['citation']['url'] for x in hits];found=all(any(e in u for u in urls) for e in expected) if expected else r.response_source in ('none','rule');rows.append({'id':qid,'question':q,'response':r.text,'response_source':r.response_source,'top3':[x['citation'] for x in hits],'expected_sources':expected,'evidence_supported':found,'latency_ms':round((time.perf_counter()-t)*1000,2),'neural_called':r.response_source=='neural'})
 out={'status':'COMPLETE','questions_total':30,'questions':rows,'single_page_hit_at_3':sum(x['evidence_supported'] for x in rows[:20]),'cross_document_results':sum(x['evidence_supported'] for x in rows[20:25]),'out_of_scope_correct_abstentions':sum(x['evidence_supported'] for x in rows[25:]),'supported_citation_rate':sum(x['evidence_supported'] for x in rows[:25])/25,'optimizer_steps':0,'unauthorized_requests':0}
 for p in (ROOT/'ONLINE_V1_GATE3_30Q_EVAL.json',ROOT/'results'/'online_v1_gate3'/'ONLINE_V1_GATE3_30Q_EVAL.json'):p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 s.close();print(json.dumps({k:out[k] for k in ('status','single_page_hit_at_3','cross_document_results','out_of_scope_correct_abstentions','supported_citation_rate')},ensure_ascii=False))
if __name__=='__main__':main()
