"""Execute the approved Gate-3 expansion to 30 pages."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
from urllib import robotparser
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from nmind.bot.knowledge_ingest import ingest_extracted, EXTRACTOR_VERSION
from nmind.bot.knowledge_store import KnowledgeStore
from nmind.bot.knowledge_search import KnowledgeSearch
from nmind.online.safe_transport import fetch_readonly
ROBOTS='https://docs.python.org/robots.txt'
PAGES=[
('https://docs.python.org/zh-cn/3.11/tutorial/appetite.html','tutorial'),('https://docs.python.org/zh-cn/3.11/tutorial/interpreter.html','tutorial'),('https://docs.python.org/zh-cn/3.11/tutorial/stdlib2.html','tutorial'),('https://docs.python.org/zh-cn/3.11/tutorial/venv.html','tutorial'),('https://docs.python.org/zh-cn/3.11/tutorial/whatnow.html','tutorial'),('https://docs.python.org/zh-cn/3.11/tutorial/interactive.html','tutorial'),('https://docs.python.org/zh-cn/3.11/tutorial/floatingpoint.html','tutorial'),
('https://docs.python.org/zh-cn/3.11/library/functions.html','library'),('https://docs.python.org/zh-cn/3.11/library/stdtypes.html','library'),('https://docs.python.org/zh-cn/3.11/library/json.html','library'),('https://docs.python.org/zh-cn/3.11/library/re.html','library'),('https://docs.python.org/zh-cn/3.11/library/sqlite3.html','library'),('https://docs.python.org/zh-cn/3.11/library/urllib.request.html','library'),('https://docs.python.org/zh-cn/3.11/library/datetime.html','library'),('https://docs.python.org/zh-cn/3.11/library/collections.html','library'),('https://docs.python.org/zh-cn/3.11/library/dataclasses.html','library'),('https://docs.python.org/zh-cn/3.11/library/hashlib.html','library'),('https://docs.python.org/zh-cn/3.11/library/logging.html','library'),('https://docs.python.org/zh-cn/3.11/library/os.html','library'),('https://docs.python.org/zh-cn/3.11/library/sys.html','library')]
ALLOWED=('docs.python.org/robots.txt','docs.python.org/zh-cn/3.11/tutorial/','docs.python.org/zh-cn/3.11/library/')
def main():
    out=ROOT/'results'/'online_v1_gate3'; out.mkdir(parents=True,exist_ok=True); db=ROOT/'results'/'online_v1_gate1'/'knowledge.db'; store=KnowledgeStore(db); audit=[]; total=0; last=None
    def get(url,kind):
        nonlocal last,total
        if last is not None: time.sleep(max(0,5-(time.monotonic()-last)))
        last=time.monotonic(); r=fetch_readonly(url,allowed_sites=ALLOWED,max_bytes=524288,timeout_s=10,max_redirects=0); raw=(r.get('content') or '').encode(); total+=len(raw)
        row={'request_no':len(audit)+1,'kind':kind,'url':url,'method':'GET','status':r.get('status'),'http_status':r.get('http_status'),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest() if raw else '','elapsed_ms':r.get('elapsed_ms'),'error_code':r.get('error_code'),'redirects':0,'network_authorized':True}; audit.append(row); return r,row
    rr,rrrow=get(ROBOTS,'robots_policy_check');
    if rr.get('status')!='fetched': raise RuntimeError('robots check failed')
    rp=robotparser.RobotFileParser(); rp.parse((rr.get('content') or '').splitlines()); allowed={u:rp.can_fetch('N-MIND-readonly/1.0',u) for u,_ in PAGES}; rrrow['robots_allowed']=allowed
    if not all(allowed.values()): raise RuntimeError('robots disallowed candidate')
    ready=0; dedup=0
    for url,kind in PAGES:
        r,row=get(url,'page')
        if r.get('status')!='fetched': continue
        ing=ingest_extracted(store,url=url,html=r['content'],fetched_at_utc=time.strftime('%Y-%m-%dT%H:%M:%S%z'),trust_state='public-read-only',http_status=r['http_status'],mime='text/html'); ready+=ing.get('status')=='READY'; dedup+=ing.get('deduplicated',0); row.update({'knowledge_status':ing['status'],'doc_id':ing.get('doc_id'),'content_sha256':ing.get('content_sha256'),'extractor_version':EXTRACTOR_VERSION,'robots_policy':'allowed','license_url':'https://docs.python.org/3.11/license.html'})
    text='\n'.join(json.dumps(x,ensure_ascii=False) for x in audit)+'\n'; (out/'ONLINE_V1_GATE3_REQUEST_AUDIT.jsonl').write_text(text,encoding='utf-8');
    with (ROOT/'ONLINE_V1_REQUEST_AUDIT.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(text)
    stats=KnowledgeSearch(store).stats(); metrics={'status':'COMPLETE','gate1_gate2_requests_used':13,'gate3_requests':len(audit),'cumulative_requests':13+len(audit),'new_pages_attempted':20,'new_pages_ready':ready,'documents':stats['documents'],'all_versions':stats['all_versions'],'chunks':stats['chunks'],'deduplicated':dedup,'bytes_gate3':total,'optimizer_steps':0,'unauthorized_requests':0,'stable_checkpoint_modified':False,'robots_all_allowed':True};
    for p in (ROOT/'ONLINE_V1_GATE3_KNOWLEDGE_METRICS.json',out/'ONLINE_V1_GATE3_KNOWLEDGE_METRICS.json'):p.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    store.close(); print(json.dumps(metrics,ensure_ascii=False));
if __name__=='__main__': main()
