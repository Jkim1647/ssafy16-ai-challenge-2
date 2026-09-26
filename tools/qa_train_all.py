# -*- coding: utf-8 -*-
"""
train_all_20260922.csv 최종 무결성 재검증 + A/AB/ABC 학습 CSV 생성 (2026-09-22).

- canonical image id = path stem (train_0001, dev_0545 ...). 문자열 id가 아니라
  실제 이미지 파일 기준으로 누수를 검사한다.
- 이미지 SHA256(파일 바이트)로 split 간 exact 중복(id가 달라도)을 잡는다.
- known_duplicates + duplicate_candidates로 group 기준 누수도 검사한다.
- 파괴적 수정 없음. 산출물만 새로 쓴다. test exact-pixel 중복은 보고만 한다.
실행: baseline/Scripts/python.exe tools/qa_train_all.py
"""
from __future__ import annotations
import pandas as pd, json, os, hashlib, sys
from collections import defaultdict, Counter
from PIL import Image

ROOT = r'C:\ssafy\AI_2_CHALLENGE'
DATA = os.path.join(ROOT, 'ssafy-16-2-ai')
OUT  = os.path.join(ROOT, 'runs', 'dev_validation_v1')
TRAIN_ALL = os.path.join(OUT, 'train_all_20260922.csv')

def norm(s): return str(s).strip().lower()
def stem(p): return os.path.splitext(os.path.basename(str(p)))[0]

_hcache = {}
def fhash(path):
    if path in _hcache: return _hcache[path]
    try:
        with open(path, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        h = None
    _hcache[path] = h
    return h

def imgpath(rel):  # path 컬럼(예: dev/dev_0545.jpg) → 절대경로
    return os.path.join(DATA, str(rel).replace('/', os.sep))

REP = []
def p(*a): REP.append(' '.join(str(x) for x in a))

# ---------------- 로드 ----------------
u = pd.read_csv(TRAIN_ALL, encoding='utf-8-sig', dtype=str, keep_default_na=False)
u['cid'] = u['path'].map(stem)          # canonical image id
gold = json.load(open(os.path.join(OUT, 'hard_eval_gold_v3.json')))
gold_cid = set(stem(k) for k in gold)   # dev_XXXX 331
val667 = json.load(open(os.path.join(ROOT, 'data_meta', 'splits', 'val667_ids.json')))['val_groups']
val_cid = set(stem(x) for x in val667)  # train_XXXX 667
# H184 (gold_v2) = gold_v3의 hard subset
gold_v2 = json.load(open(os.path.join(OUT, 'hard_eval_gold_v2.json')))
h184_cid = set(stem(k) for k in gold_v2)

p('='*72); p('train_all_20260922.csv 최종 무결성 재검증'); p('='*72)
p(f'행 수: {len(u)} | 컬럼: {list(u.columns)}')
p()

# ---------------- 1. id 중복 ----------------
iddup = u[u['id'].duplicated(keep=False)]
p('[1] id 중복 :', len(iddup))

# ---------------- 2. 결측 / answer 유효성 / 정답 선택지 존재 ----------------
need = ['id','path','question','a','b','c','d','answer']
miss = {c: int((u[c].map(lambda x: str(x).strip()=='')).sum()) for c in need}
p('[2] 필드 결측(빈문자열):', miss)
bad_ans = u[~u['answer'].map(norm).isin(list('abcd'))]
p('    answer∉{a,b,c,d} :', len(bad_ans))
def chosen_empty(r):
    a = norm(r['answer'])
    return a in list('abcd') and str(r[a]).strip() == ''
empty_choice = u[u.apply(chosen_empty, axis=1)]
p('    정답 선택지 텍스트 공백 :', len(empty_choice))
if len(empty_choice): p('      예:', empty_choice['id'].head(10).tolist())

# ---------------- 3. 이미지 존재 / decode ----------------
missing_img, decode_fail = [], []
u['abspath'] = u['path'].map(imgpath)
uniq_paths = u['abspath'].unique()
for ap in uniq_paths:
    if not os.path.exists(ap):
        missing_img.append(ap); continue
    try:
        with Image.open(ap) as im:
            im.load()
    except Exception as e:
        decode_fail.append((ap, str(e)[:60]))
p('[3] 참조 고유 이미지 파일:', len(uniq_paths))
p('    존재하지 않는 파일 :', len(missing_img))
p('    decode 실패        :', len(decode_fail))
if missing_img: p('      예:', missing_img[:5])
if decode_fail: p('      예:', decode_fail[:5])

# ---------------- 4. 이미지 hash 중복 (train_all 내부) ----------------
u['ihash'] = u['abspath'].map(fhash)
hash_groups = u.groupby('ihash')['cid'].apply(lambda s: sorted(set(s)))
multi = {h: c for h, c in hash_groups.items() if len(c) > 1 and h is not None}
p('[4] train_all 내부 이미지 hash 중복 그룹(서로 다른 cid가 같은 픽셀):', len(multi))
for h, cids in list(multi.items())[:8]:
    p('      ', cids)

# ---------------- 5. source별 건수/answer 분포 ----------------
p('[5] source별 건수 / answer 분포')
for s in ['train_orig','dev_pseudo','codex_regen']:
    sub = u[u['source']==s]
    ad = dict(Counter(sub['answer'].map(norm)))
    p(f'    {s:12s} n={len(sub):5d}  ans={ad}')

# ---------------- group union-find (known_dup + dup_candidates) ----------------
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

kd = pd.read_csv(os.path.join(ROOT,'data_meta','known_duplicates.csv'), encoding='utf-8-sig')
for _, r in kd.iterrows():
    union(stem(r['stem_a']), stem(r['stem_b']))
dc = pd.read_csv(os.path.join(OUT,'duplicate_candidates.csv'), encoding='utf-8-sig')
exact_pairs = dc[dc['kind']=='exact_pixels']
for _, r in dc.iterrows():
    union(stem(r['id_a']), stem(r['id_b']))
def grp(cid): return find(cid)

# ---------------- 6. 누수 검사 (id / hash / group) ----------------
gold_paths = {c: os.path.join(DATA,'dev',c+'.jpg') for c in gold_cid}
val_paths  = {c: os.path.join(DATA,'train',c+'.jpg') for c in val_cid}
gold_hashes = {fhash(pth) for pth in gold_paths.values()} - {None}
val_hashes  = {fhash(pth) for pth in val_paths.values()} - {None}
train_hashes = set(u['ihash']) - {None}
train_cids   = set(u['cid'])
train_groups = {grp(c) for c in train_cids}

def leak_block(name, ecid, ehashes):
    id_ov   = train_cids & ecid
    hash_ov = train_hashes & ehashes
    grp_ov  = {c for c in ecid if grp(c) in train_groups}
    # hash 겹침을 낸 학습 행
    hrows = sorted(set(u[u['ihash'].isin(ehashes)]['cid'])) if hash_ov else []
    p(f'[6] {name} 누수  id={len(id_ov)}  imagehash={len(hash_ov)}  group={len(grp_ov)}')
    if id_ov:  p('       id:', sorted(id_ov)[:10])
    if hash_ov:p('       hash 겹친 학습 cid:', hrows[:10])
    if grp_ov: p('       group:', sorted(grp_ov)[:10])
    return dict(id=sorted(id_ov), hash_train_cids=hrows, group=sorted(grp_ov))

leak_gold = leak_block('gold_v3(331)', gold_cid, gold_hashes)
leak_val  = leak_block('val667(667)',  val_cid,  val_hashes)

# ---------------- 7. test exact-pixel 중복 (보고만) ----------------
te = pd.read_csv(os.path.join(DATA,'test.csv'), encoding='utf-8-sig', dtype=str, keep_default_na=False)
te_cid = set(stem(x) for x in te['id'])
te_hashes = {}
for c in te_cid:
    te_hashes[c] = fhash(os.path.join(DATA,'test',c+'.jpg'))
te_hash_set = set(te_hashes.values()) - {None}
test_hash_ov = train_hashes & te_hash_set
test_rows = sorted(set(u[u['ihash'].isin(te_hash_set)]['cid'])) if test_hash_ov else []
# duplicate_candidates 기준 exact_pixels 중 test와 얽힌 쌍
ep_test = exact_pairs[(exact_pairs['id_a'].str.contains('test')) | (exact_pairs['id_b'].str.contains('test'))]
p('[7] test exact-pixel 중복 (보고만, 자동수정 안 함)')
p('    file-hash 기준 train_all∩test :', len(test_hash_ov), '| 얽힌 학습 cid:', test_rows[:10])
p('    duplicate_candidates exact_pixels 중 test 얽힌 쌍:')
for _, r in ep_test.iterrows():
    p('      ', r['id_a'], '==', r['id_b'])

# ---------------- 8. source 상호 중복 ----------------
p('[8] source 상호 중복 (같은 cid가 여러 source에)')
cid_src = u.groupby('cid')['source'].apply(lambda s: sorted(set(s)))
cross = {c: s for c, s in cid_src.items() if len(s) > 1}
p('    같은 이미지가 2개 이상 source에 등장:', len(cross))
for c, s in list(cross.items())[:8]: p('      ', c, s)

# ---------------- A/AB/ABC CSV ----------------
assert not (train_cids & gold_cid), 'FATAL gold 누수'
assert not (train_cids & val_cid),  'FATAL val 누수'
cols = ['id','path','question','a','b','c','d','answer','source']
A   = u[u['source']=='train_orig'][cols]
AB  = u[u['source'].isin(['train_orig','dev_pseudo'])][cols]
ABC = u[cols]
A.to_csv(os.path.join(OUT,'train_A.csv'),   index=False, encoding='utf-8')
AB.to_csv(os.path.join(OUT,'train_AB.csv'), index=False, encoding='utf-8')
ABC.to_csv(os.path.join(OUT,'train_ABC.csv'),index=False, encoding='utf-8')
p()
p('[생성] train_A.csv=%d  train_AB.csv=%d  train_ABC.csv=%d' % (len(A),len(AB),len(ABC)))

# ---------------- QA json ----------------
qa = dict(
  rows=len(u), id_dup=len(iddup), field_missing=miss,
  answer_invalid=len(bad_ans), empty_choice=len(empty_choice),
  missing_img=len(missing_img), decode_fail=len(decode_fail),
  internal_hash_dup_groups=len(multi),
  source_counts={s:int((u['source']==s).sum()) for s in ['train_orig','dev_pseudo','codex_regen']},
  leak_gold=leak_gold, leak_val=leak_val,
  test_hash_overlap=len(test_hash_ov), test_exact_pairs=len(ep_test),
  cross_source_same_image=len(cross),
  datasets=dict(A=len(A), AB=len(AB), ABC=len(ABC)),
  h184_in_gold=len(h184_cid),
)
json.dump(qa, open(os.path.join(OUT,'qa_train_all_20260922.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
open(os.path.join(OUT,'QA_TRAIN_ALL_20260922.txt'),'w',encoding='utf-8-sig').write('\n'.join(REP))
print('OK. report -> QA_TRAIN_ALL_20260922.txt / qa_train_all_20260922.json')
