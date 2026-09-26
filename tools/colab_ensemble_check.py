"""35B read+tiles 와 27B LoRA(1장 / read+tiles) 앙상블을 val667·H184에서 비교한다(HANDOFF 앙상블 검토표의 출처).
"""
import json, math, os
O='/content/drive/MyDrive/ai2_35b_out/'
def load(p):
    p=O+p
    return {json.loads(l)['id']:json.loads(l) for l in open(p,encoding='utf-8') if l.strip()} if os.path.exists(p) else None
def sm(lp):
    m=max(lp); e=[math.exp(x-m) for x in lp]; z=sum(e); return [x/z for x in e]
g=json.load(open('/content/h184_gold.json')); hid=sorted(g)
V={'35rt':'Qwen3.6-35B-A3B_read_val667_tiles2x2x2_predictions.jsonl','27L':'lora/27b_lora_r8/val_predictions.jsonl',
   '27Lrt':'Qwen3.6-27B_read_val667_tiles2x2x2_lora_predictions.jsonl','35t':'Qwen3.6-35B-A3B_ours_val667_tiles2x2x2_predictions.jsonl'}
H={'35rt':'Qwen3.6-35B-A3B_read_devH184_tiles2x2x2_predictions.jsonl','27L':'lora/27b_lora_r8/hard_predictions.jsonl',
   '27Lrt':'Qwen3.6-27B_read_devH184_tiles2x2x2_lora_predictions.jsonl','35t':'Qwen3.6-35B-A3B_ours_devH184_tiles2x2x2_predictions.jsonl'}
v={k:load(p) for k,p in V.items()}; h={k:load(p) for k,p in H.items()}
vid=sorted(v['35rt']); vg={i:v['35rt'][i]['gold'] for i in vid}
def acc(P,ids,gold,ks,w=None):
    w=w or [1]*len(ks)
    return round(sum('abcd'[max(range(4),key=lambda j:sum(wk*sm(P[k][i]['logprobs'])[j] for k,wk in zip(ks,w)))]==gold[i] for i in ids)/len(ids),4)
for ks,w in [(['35rt'],None),(['27L'],None),(['27Lrt'],None),(['35rt','27L'],None),(['35rt','27L'],[2,1]),(['35rt','27Lrt'],None),(['35rt','35t','27L'],None)]:
    ok=all(v[k] and all(i in v[k] for i in vid) for k in ks) and all(h[k] and all(i in h[k] for i in hid) for k in ks)
    print('ENS', ks, w, 'val667', acc(v,vid,vg,ks,w) if ok else '-', 'H184', acc(h,hid,g,ks,w) if ok else '-')
