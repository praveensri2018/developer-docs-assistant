import json,time
from pathlib import Path
from core import Assistant
ROOT=Path(__file__).parent
assistant=Assistant()
assistant.ingest([(p.name,1,p.read_text()) for p in sorted((ROOT/'samples').glob('*.md'))])
cases=[('How do I create an order?','orders.md'),('When do authentication tokens expire?','orders.md'),('Which fields are required in order items?','orders.md'),('What is the API request rate limit?','operations.md'),('What happens after a deployment fails?','operations.md'),('Where is the service health endpoint?','operations.md'),('Who won the tennis championship?',None),('What is the capital of Mongolia?',None)]
rows=[]
for question,expected in cases:
    start=time.perf_counter();result=assistant.ask(question)
    hit=any(s['source']==expected for s in result['sources']) if expected else result['status']=='insufficient_evidence'
    rows.append({'question':question,'expected_source':expected,'passed':hit,'status':result['status'],'latency_ms':round((time.perf_counter()-start)*1000,2),'sources':[s['source'] for s in result['sources']]})
report={'mode':'extractive; no LLM calls','scope':'8 hand-authored smoke cases on 2 fictional documents; not generalization or answer-quality evaluation','retrieval_hit_at_3':sum(r['passed'] for r in rows[:6])/6,'abstention_accuracy':sum(r['passed'] for r in rows[6:])/2,'cases':rows}
(ROOT/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
