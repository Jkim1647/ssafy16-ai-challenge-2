"""어려운 평가셋(H184)·val667 조합 분석: 프롬프트·회전 TTA·타일·27B·122B 앙상블 정확도와 read 오답 예시.

Colab에서 /content/h184_gold.json 과 Drive ai2_35b_out 예측 파일이 있을 때 실행한다(HANDOFF 35B 비교표의 출처).
"""
import json, math, os
O='/content/drive/MyDrive/ai2_35b_out/'
g=json.load(open('/content/h184_gold.json'))
def load(f):
    p=O+f
    return {json.loads(l)['id']:json.loads(l) for l in open(p,encoding='utf-8') if l.strip()} if os.path.exists(p) else None
def sm(lp):
    m=max(lp); e=[math.exp(x-m) for x in lp]; z=sum(e); return [x/z for x in e]
F={'ours':'Qwen3.6-35B-A3B_ours_devH184_predictions.jsonl','dev_ours':'Qwen3.6-35B-A3B_ours_dev_predictions.jsonl',
   'read':'Qwen3.6-35B-A3B_read_devH184_predictions.jsonl','en':'Qwen3.6-35B-A3B_en_devH184_predictions.jsonl',
   'r90':'Qwen3.6-35B-A3B_ours_devH184_r90_predictions.jsonl','r180':'Qwen3.6-35B-A3B_ours_devH184_r180_predictions.jsonl',
   'r270':'Qwen3.6-35B-A3B_ours_devH184_r270_predictions.jsonl','readtiles':'Qwen3.6-35B-A3B_read_devH184_tiles2x2x2_predictions.jsonl',
   '27b':'Qwen3.6-27B_ours_dev_predictions.jsonl','tiles':'Qwen3.6-35B-A3B_ours_devH184_tiles2x2x2_predictions.jsonl','122b':'Qwen3.5-122B-A10B-GPTQ-Int4_ours_devH184_vllm_predictions.jsonl'}
P={k:load(v) for k,v in F.items()}
ids=sorted(g)
full=lambda k: P[k] is not None and all(i in P[k] for i in ids)
base=P['ours'] if full('ours') else P['dev_ours']
def acc(fn): return round(sum(fn(i)==g[i] for i in ids)/len(ids),4)
pred=lambda k:(lambda i:P[k][i]['pred'])
for k in F:
    if P[k] and all(i in P[k] for i in ids): print('H184',k,acc(pred(k)))
def ens(ks,w=None):
    w=w or [1]*len(ks)
    return lambda i:'abcd'[max(range(4),key=lambda j:sum(wk*sm(P[k][i]['logprobs'])[j] for k,wk in zip(ks,w)))]
P['base']=base
rot=[k for k in ['base','r90','r180','r270'] if full(k)]
if len(rot)==4:
    print('H184 rotTTA maxconf', acc(lambda i:max((P[k][i] for k in rot),key=lambda r:r['confidence'])['pred']))
    print('H184 rotTTA base-unless-other>0.9&base<0.6', acc(lambda i: (lambda b,o: o['pred'] if (b['confidence']<0.6 and o['confidence']>0.9) else b['pred'])(P['base'][i], max((P[k][i] for k in rot[1:]),key=lambda r:r['confidence']))))
for ks in [['base','read'],['read','27b'],['base','read','27b'],['read','122b'],['base','122b'],['read','readtiles'],['base','tiles'],['read','tiles'],['tiles','27b']]:
    if all(full(k) for k in ks): print('H184 ens',ks,acc(ens(ks)))
# val667 read vs ours
v={k:load(f) for k,f in {'ours':'Qwen3.6-35B-A3B_ours_val667_predictions.jsonl','tiles':'Qwen3.6-35B-A3B_ours_val667_tiles2x2x2_predictions.jsonl','read':'Qwen3.6-35B-A3B_read_val667_predictions.jsonl','27b':'Qwen3.6-27B_ours_val667_predictions.jsonl','122b':'Qwen3.5-122B-A10B-GPTQ-Int4_ours_val667_vllm_predictions.jsonl'}.items()}
vid=sorted(v['ours']); vg={i:v['ours'][i]['gold'] for i in vid}
va=lambda fn: round(sum(fn(i)==vg[i] for i in vid)/len(vid),4)
for k in v:
    if v[k] and len(v[k])==len(vid): print('val667',k,va(lambda i:v[k][i]['pred']))
def vens(ks): return lambda i:'abcd'[max(range(4),key=lambda j:sum(sm(v[k][i]['logprobs'])[j] for k in ks))]
for ks in [['ours','read'],['read','27b'],['27b','ours','122b'],['27b','122b'],['read','27b','122b'],['ours','tiles'],['read','tiles'],['tiles','27b']]:
    if all(v[k] and len(v[k])==len(vid) for k in ks): print('val667 ens',ks,va(vens(ks)))
if full('read'):
    bad=[i for i in ids if P['read'][i]['pred']!=g[i]][:5]
    for i in bad: print('READ-ERR',i,g[i],P['read'][i]['pred'],P['read'][i].get('transcript','')[:80].replace('\n',' '))
