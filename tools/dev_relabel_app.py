"""dev 재라벨링 로컬 화면 (teammate_handoff 패킷용).

    python tools/dev_relabel_app.py            # http://127.0.0.1:8765
    python tools/dev_relabel_app.py --port 8766

표준 라이브러리만 쓴다. 127.0.0.1에만 바인딩한다(이미지·라벨을 외부에 노출하지 않는다).

지키는 규칙 (teammate_handoff/TEAMMATE_HANDOFF.md):
  - 작성 가능한 11개 열만 저장한다. id·path·question·a~d·auto_type·assignment·is_calibration,
    행 순서, 행 수는 절대 바꾸지 않는다.
  - approved는 필수 여섯 조건을 모두 만족해야 저장된다.
  - readability가 partial/unreadable이면 human_answer를 넣을 수 없다.
  - 9B 초안(llm_draft.csv)은 기본 숨김. calibration 파일에서는 아예 제공하지 않는다.

저장은 매번 임시 파일에 쓴 뒤 교체한다(중간에 꺼져도 labels.csv가 반쯤 쓰이지 않는다).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
PACKET = REPO / "teammate_handoff"
DATA = REPO / "ssafy-16-2-ai"
FILES = {
    "calibration": PACKET / "calibration" / "labels.csv",
    "teammate": PACKET / "teammate" / "labels.csv",
    # 다른 작업자 담당분(1,317). 팀 합의로 같은 화면에서 검수한다. 반환물과 섞지 않도록 별도 폴더.
    "other": PACKET / "other" / "labels.csv",
    # 사람≠9B 불일치 가린 재판정(tools/relabel_merge_compare.py가 생성). 초안을 절대 보여 주지 않는다.
    "recheck": PACKET / "recheck" / "labels.csv",
}
WRITABLE = [
    "human_answer", "confidence", "readability", "quality_reason", "data_usage",
    "ambiguity_reason", "evidence", "reviewer", "llm_assistance", "review_status", "review_note",
]
ALLOWED = {
    "human_answer": {"", "a", "b", "c", "d"},
    "confidence": {"", "high", "medium", "low"},
    "readability": {"", "clear", "partial", "unreadable"},
    "quality_reason": {"", "none", "blur", "low_resolution", "glare", "occlusion", "bad_crop", "other"},
    "data_usage": {"", "vqa_train", "vision_auxiliary_candidate", "exclude"},
    "review_status": {"", "approved", "needs_review", "rejected"},
}
LOCK = threading.Lock()


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_table(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    tmp = path.with_suffix(".csv.tmp")
    # 사람이 Excel로 여는 반환 파일이라 utf-8-sig (CLAUDE.md 인코딩 규칙 2, 원본 패킷도 BOM 포함)
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write(buf.getvalue())
    os.replace(tmp, path)


def validate(f: dict[str, str]) -> str | None:
    for key, allowed in ALLOWED.items():
        if f.get(key, "") not in allowed:
            return f"{key}={f.get(key)!r} 는 허용값이 아니다"
    if f.get("readability") in {"partial", "unreadable"} and f.get("human_answer"):
        return "partial/unreadable 이면 human_answer를 비워야 한다"
    if f.get("readability") in {"partial", "unreadable"} and f.get("data_usage") == "vqa_train":
        return "partial/unreadable 이면 data_usage는 vision_auxiliary_candidate 또는 exclude여야 한다"
    if f.get("review_status"):
        empty = [k for k in ("readability", "quality_reason", "data_usage") if not f.get(k)]
        if empty:
            return "검토 상태를 넣으면 이 칸도 채워야 한다: " + ", ".join(empty)
        if f["review_status"] != "approved" and not (f.get("ambiguity_reason") or f.get("review_note")):
            return "needs_review/rejected는 ambiguity_reason 또는 review_note에 이유를 적어야 한다"
    if f.get("review_status") == "approved":
        need = {
            "human_answer": f.get("human_answer") in {"a", "b", "c", "d"},
            "confidence": f.get("confidence") in {"high", "medium"},
            "readability": f.get("readability") == "clear",
            "quality_reason": f.get("quality_reason") == "none",
            "data_usage": f.get("data_usage") == "vqa_train",
            "evidence": bool(f.get("evidence", "").strip()),
        }
        bad = [k for k, ok in need.items() if not ok]
        if bad:
            return "approved 조건 불충족: " + ", ".join(bad)
    if f.get("review_status") and not f.get("reviewer", "").strip():
        return "reviewer를 적어야 한다"
    return None


def load_draft(kind: str, name: str = "llm_draft.csv") -> dict[str, dict[str, str]]:
    if kind in {"calibration", "recheck"}:
        return {}                     # 기준 맞추기·가린 재판정 단계라 초안을 주지 않는다
    path = FILES[kind].with_name(name)
    if not path.exists():
        return {}
    _, rows = read_table(path)
    return {r["id"]: r for r in rows}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 조용히
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if url.path == "/api/rows":
            kind = parse_qs(url.query).get("file", ["teammate"])[0]
            if kind not in FILES:
                return self._json({"error": "unknown file"}, 400)
            if not FILES[kind].exists():
                return self._json({"rows": [], "draft": {}, "cdraft": {}, "writable": WRITABLE})
            with LOCK:
                _, rows = read_table(FILES[kind])
            draft = load_draft(kind)
            return self._json({"rows": rows, "draft": draft, "cdraft": load_draft(kind, "claude_draft.csv"),
                               "writable": WRITABLE})
        if url.path.startswith("/img/"):
            rel = unquote(url.path[len("/img/"):])
            target = (DATA / rel).resolve()
            if DATA.resolve() / "dev" not in target.parents or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return self._send(200, target.read_bytes(), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self._send(404, b"not found", "text/plain")
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._json({"error": f"요청을 읽지 못했다(UTF-8 JSON이어야 함): {exc}"}, 400)
        kind, row_id, fields = payload.get("file"), payload.get("id"), payload.get("fields", {})
        if kind not in FILES:
            return self._json({"error": "unknown file"}, 400)
        clean = {k: str(fields.get(k, "")).strip() for k in WRITABLE}
        if err := validate(clean):
            return self._json({"error": err}, 422)
        with LOCK:
            header, rows = read_table(FILES[kind])
            hit = [r for r in rows if r["id"] == row_id]
            if len(hit) != 1:
                return self._json({"error": f"id {row_id} 를 찾지 못했다"}, 404)
            hit[0].update(clean)
            write_table(FILES[kind], header, rows)
        return self._json({"ok": True})


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>dev 재라벨링</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f6f4;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#dcdcd6;--acc:#2f5bd3;--ok:#1f8a4c;--warn:#b7791f;--bad:#c0392b;--sel:#e8eefc}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#20201e;--ink:#ecece8;--muted:#9b9b94;--line:#3a3a36;--acc:#7c9cf5;--ok:#4fbf7f;--warn:#e0a84a;--bad:#ef6b5b;--sel:#27304a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,"Malgun Gothic",sans-serif}
header{display:flex;gap:12px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
header b{font-size:15px}select,input,textarea,button{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(360px,1fr);height:calc(100vh - 50px)}
#imgbox{overflow:auto;background:#000;position:relative}#imgbox img{display:block;margin:auto;max-width:100%;max-height:100%;cursor:zoom-in}
#imgbox.zoomed img{max-width:none;max-height:none;margin:0;cursor:zoom-out}#imgbox.contrast img{filter:contrast(1.8) brightness(1.05)}
#zoombar{position:sticky;top:0;left:0;z-index:2;display:flex;gap:4px;padding:4px;background:rgba(0,0,0,.55);width:max-content}#zoombar button{padding:2px 8px;background:#222;color:#eee;border-color:#555}#zl{color:#eee;font-size:12px;align-self:center;min-width:38px;text-align:center}
#side{overflow:auto;padding:12px 14px;border-left:1px solid var(--line);background:var(--panel)}
.q{font-size:16px;font-weight:600;margin:4px 0 10px}.meta{color:var(--muted);font-size:12px}
.ch{display:flex;gap:8px;align-items:flex-start;width:100%;text-align:left;margin:4px 0;padding:8px 10px}
.ch.on{background:var(--sel);border-color:var(--acc);font-weight:600}.ch kbd{min-width:18px}
.grid{display:grid;grid-template-columns:110px 1fr;gap:6px 8px;align-items:center;margin-top:10px}
.grid label{color:var(--muted);font-size:12px}textarea{width:100%;min-height:44px;resize:vertical}
.row{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.st-approved{color:var(--ok)}.st-needs_review{color:var(--warn)}.st-rejected{color:var(--bad)}
#msg{min-height:20px;margin-top:8px;font-weight:600}#draft{margin-top:8px;padding:8px;border:1px dashed var(--line);border-radius:6px;font-size:13px}
#cdraft{margin:8px 0;padding:8px 10px;border-left:4px solid var(--acc);background:var(--sel);border-radius:4px;font-size:13px}#cdraft.warn{border-left-color:var(--warn)}#cdraft .bad{color:var(--bad);font-weight:700}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:11px}.help{color:var(--muted);font-size:12px;margin-top:12px}
@media (max-width:820px){main{grid-template-columns:1fr;height:auto}#imgbox{height:60vh}}
</style></head><body>
<header>
 <b>dev 재라벨링</b><span class="meta">정답 <kbd>human_answer</kbd> = 아래 선택지 클릭</span>
 <select id="file"><option value="calibration">calibration (50)</option><option value="teammate">teammate (1,316)</option><option value="other">other · 석웅 담당분 (1,317)</option><option value="recheck">recheck · 가린 재판정</option></select>
 <select id="filter"><option value="all">전체</option><option value="todo">미작성</option><option value="needs_review">needs_review</option><option value="rejected">rejected</option><option value="low">confidence=low</option><option value="d_hi" class="dopt">미작성 · 9B 고확신(≥0.9)</option><option value="d_mid" class="dopt">미작성 · 9B 중간(0.7~0.9)</option><option value="d_lo" class="dopt">미작성 · 9B 저확신(&lt;0.7)</option><option value="c_todo" class="copt">검수 대기 · Claude 초안 전체</option><option value="c_dis" class="copt">검수 대기 · Claude≠9B</option><option value="c_ok" class="copt">검수 대기 · Claude 승인</option><option value="c_rev" class="copt">검수 대기 · Claude 보류/거절</option></select>
 <button id="prev">◀</button><span id="pos"></span><button id="next">▶</button>
 <button id="jump">첫 미작성으로</button>
 <span id="stats" class="meta"></span>
 <span style="margin-left:auto" class="meta">검토자 <input id="reviewer" size="8"></span>
</header>
<main>
 <div id="imgbox"><div id="zoombar"><button id="zout">−</button><span id="zl">맞춤</span><button id="zin">+</button><button id="zfit">맞춤</button><button id="zcon">대비</button></div><img id="img" alt=""></div>
 <div id="side">
  <div class="meta" id="rowmeta"></div>
  <div class="q" id="q"></div>
  <div id="cdraft" hidden></div>
  <div id="choices"></div>
  <div class="row">
   <button id="pApprove" title="판독성 clear · 품질 none · 용도 vqa_train · 상태 approved">승인 프리셋 <kbd>Enter</kbd></button>
   <button id="pReview">애매 → needs_review <kbd>R</kbd></button>
   <button id="pReject">판독불가 → rejected <kbd>X</kbd></button>
  </div>
  <div class="grid">
   <label>확신도<br>confidence</label><select id="confidence"><option value=""></option><option value="high">high — 확실</option><option value="medium">medium — 대체로 확실</option><option value="low">low — 불확실</option></select>
   <label>판독성<br>readability</label><select id="readability"><option value=""></option><option value="clear">clear — 선명하게 읽힘</option><option value="partial">partial — 일부만 읽힘</option><option value="unreadable">unreadable — 판독 불가</option></select>
   <label>품질 문제<br>quality_reason</label><select id="quality_reason"><option value=""></option><option value="none">none — 문제 없음</option><option value="blur">blur — 흐림</option><option value="low_resolution">low_resolution — 저해상도</option><option value="glare">glare — 빛 반사</option><option value="occlusion">occlusion — 가려짐</option><option value="bad_crop">bad_crop — 잘림</option><option value="other">other — 기타</option></select>
   <label>데이터 용도<br>data_usage</label><select id="data_usage"><option value=""></option><option value="vqa_train">vqa_train — VQA 학습 후보</option><option value="vision_auxiliary_candidate">vision_auxiliary_candidate — 보조비전 후보</option><option value="exclude">exclude — 제외</option></select>
   <label>검토 상태<br>review_status</label><select id="review_status"><option value=""></option><option value="approved">approved — 승인</option><option value="needs_review">needs_review — 재검토 필요</option><option value="rejected">rejected — 거절</option></select>
   <label>근거<br>evidence</label><textarea id="evidence" placeholder="정답 근거가 된 이미지 속 글자·위치 (한 문장)"></textarea>
   <label>애매한 이유<br>ambiguity_reason</label><input id="ambiguity_reason" placeholder="복수 정답 / 대상 여러 개 / 일부만 읽힘 등">
   <label>메모<br>review_note</label><input id="review_note" placeholder="LLM과 판단이 다르면 여기에">
   <label>LLM 사용<br>llm_assistance</label><input id="llm_assistance" placeholder="none 또는 모델명 (초안 보기 시 자동)">
  </div>
  <div class="row"><button class="pri" id="save">저장 후 다음 <kbd>Ctrl+S</kbd></button><button id="clearAns">정답 지우기</button><button id="showDraft">9B 초안 보기</button><button id="applyC" hidden>Claude 초안 다시 적용 <kbd>C</kbd></button><label class="meta"><input type="checkbox" id="autoC" checked> 자동 적용</label></div>
  <div id="draft" hidden></div>
  <div id="msg"></div>
  <div class="help"><kbd>1</kbd>~<kbd>4</kbd> 정답 (<kbd>C</kbd>는 Claude 초안 다시 적용) · <kbd>+</kbd><kbd>-</kbd> 확대/축소(Ctrl+휠) · 이미지 클릭 = 그 지점 3배 · <kbd>0</kbd> 맞춤 · <kbd>V</kbd> 대비 강조 · <kbd>←</kbd><kbd>→</kbd> 이동(입력칸 밖) · 초안은 먼저 스스로 판단한 뒤 여세요</div>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
const FIELDS=["confidence","readability","quality_reason","data_usage","review_status","evidence","ambiguity_reason","review_note","llm_assistance"];
let rows=[],draft={},cdraft={},view=[],idx=0,ans="";
const store={get(k){try{return localStorage.getItem(k)}catch(e){return null}},set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};
$("reviewer").value=store.get("reviewer")||"";$("reviewer").onchange=()=>store.set("reviewer",$("reviewer").value.trim());
$("file").value=store.get("file")||"calibration";
function msg(t,c){$("msg").textContent=t;$("msg").style.color=c||"var(--ink)"}
async function load(){store.set("file",$("file").value);
 const r=await fetch("/api/rows?file="+$("file").value).then(r=>r.json());rows=r.rows;draft=r.draft;cdraft=r.cdraft||{};const hasC=Object.keys(cdraft).length;document.querySelectorAll(".copt").forEach(o=>o.hidden=!hasC);$("applyC").hidden=!hasC;if(!hasC&&$("filter").value.startsWith("c_"))$("filter").value="all";$("showDraft").hidden=!Object.keys(draft).length;document.querySelectorAll(".dopt").forEach(o=>o.hidden=!Object.keys(draft).length);if(!Object.keys(draft).length&&$("filter").value.startsWith("d_"))$("filter").value="all";applyFilter(true)}
function applyFilter(keep){const f=$("filter").value;
 const dp=r=>{const d=draft[r.id];return d?parseFloat(d.llm_prob):NaN};
 view=rows.map((r,i)=>i).filter(i=>{const r=rows[i];if(f.startsWith("c_")){const c=cdraft[r.id];if(r.review_status||!c)return false;return f==="c_todo"||(f==="c_dis"&&c.disagree_9b==="1")||(f==="c_ok"&&c.review_status==="approved")||(f==="c_rev"&&c.review_status!=="approved")}if(f.startsWith("d_")){if(r.review_status)return false;const p=dp(r);return f==="d_hi"?p>=0.9:f==="d_mid"?(p>=0.7&&p<0.9):p<0.7}return f==="all"||(f==="todo"&&!r.review_status)||(f==="low"&&r.confidence==="low")||r.review_status===f});
 if(!keep||idx>=view.length)idx=0;stats();show()}
function stats(){const c={approved:0,needs_review:0,rejected:0,low:0,todo:0};rows.forEach(r=>{if(!r.review_status)c.todo++;else c[r.review_status]++;if(r.confidence==="low")c.low++});
 $("stats").textContent=`완료 ${rows.length-c.todo}/${rows.length} · 승인 ${c.approved} · 보류 ${c.needs_review} · 거절 ${c.rejected} · low ${c.low}`}
function show(){if(!view.length){$("pos").textContent="0/0";$("q").textContent="해당하는 행이 없습니다";$("choices").innerHTML="";$("img").removeAttribute("src");return}
 const r=rows[view[idx]];$("pos").textContent=`${idx+1}/${view.length}`;
 $("rowmeta").textContent=`${r.id} · ${r.auto_type}`+(r.review_status?` · 현재: ${r.review_status}`:"");
 $("q").textContent=r.question;$("img").src="/img/"+encodeURIComponent(r.path).replace(/%2F/g,"/");setZoom(0);
 ans=r.human_answer||"";FIELDS.forEach(k=>$(k).value=r[k]||"");$("draft").hidden=true;msg("");renderC(r);
 if(!r.review_status&&cdraft[r.id]&&$("autoC").checked)applyC();else renderChoices(r)}
function renderC(r){const c=cdraft[r.id],box=$("cdraft");if(!c){box.hidden=true;return}box.hidden=false;const dis=c.disagree_9b==="1";box.className=dis?"warn":"";
 box.innerHTML="";const t=document.createElement("div");t.innerHTML=`<b>Claude 초안</b> · ${c.review_status} · 정답 <b>${c.answer||"—"}</b> · ${c.confidence} · ${c.readability}/${c.quality_reason}`+(dis?` · <span class="bad">9B는 ${c.llm_answer} (확률 ${c.llm_prob}) — 반드시 직접 대조</span>`:"")+(c.rule_error?` · <span class="bad">규칙 위반: ${c.rule_error}</span>`:"");box.append(t);
 const e=document.createElement("div");e.className="meta";e.textContent="근거: "+(c.evidence||"-")+(c.ambiguity_reason?" · 이유: "+c.ambiguity_reason:"");box.append(e)}
function applyC(){const r=rows[view[idx]],c=cdraft[r.id];if(!c)return;ans=c.answer||"";
 const map={confidence:c.confidence,readability:c.readability,quality_reason:c.quality_reason,data_usage:c.data_usage,review_status:c.review_status,evidence:c.evidence,ambiguity_reason:c.ambiguity_reason,llm_assistance:c.model||"claude-opus-5"};
 Object.entries(map).forEach(([k,v])=>$(k).value=v||"");if(c.disagree_9b==="1"&&!$("review_note").value)$("review_note").value=`9B는 ${c.llm_answer}`;renderChoices(r);msg("Claude 초안을 채웠습니다 — 이미지와 대조한 뒤 저장하세요","var(--acc)")}
function renderChoices(r){$("choices").innerHTML="";["a","b","c","d"].forEach((k,i)=>{const b=document.createElement("button");b.className="ch"+(ans===k?" on":"");
 b.innerHTML=`<kbd>${i+1}</kbd><span><b>${k}.</b> </span>`;b.querySelector("span").append(document.createTextNode(r[k]));b.onclick=()=>{ans=k;renderChoices(r)};$("choices").append(b)})}
function preset(p){const r=rows[view[idx]];
 if(p==="approve"){if(!$("confidence").value)$("confidence").value="high";$("readability").value="clear";$("quality_reason").value="none";$("data_usage").value="vqa_train";$("review_status").value="approved"}
 if(p==="review"){$("review_status").value="needs_review";ans="";renderChoices(r);$("ambiguity_reason").focus()}
 if(p==="reject"){$("readability").value="unreadable";if(!$("quality_reason").value||$("quality_reason").value==="none")$("quality_reason").value="blur";$("data_usage").value="exclude";$("review_status").value="rejected";ans="";renderChoices(r)}}
async function save(){const r=rows[view[idx]];if(!$("reviewer").value.trim()){msg("검토자 이름을 먼저 입력하세요","var(--bad)");$("reviewer").focus();return}
 const f={human_answer:ans,reviewer:$("reviewer").value.trim()};FIELDS.forEach(k=>f[k]=$(k).value.trim());if(!f.llm_assistance)f.llm_assistance="none";
 const res=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file:$("file").value,id:r.id,fields:f})});
 const j=await res.json();if(!res.ok){msg(j.error,"var(--bad)");return}Object.assign(r,f);stats();msg("저장됨","var(--ok)");
 if($("filter").value!=="all"){applyFilter(true)}else if(idx<view.length-1){idx++;show();msg("저장됨","var(--ok)")}}
$("showDraft").onclick=()=>{const r=rows[view[idx]],d=draft[r.id];if(!d){$("draft").hidden=false;$("draft").textContent="이 행의 초안이 없습니다";return}
 $("draft").hidden=false;$("draft").innerHTML=`9B 후보: <b>${d.llm_answer}</b> (확률 ${d.llm_prob}, 2위와 차이 ${d.llm_margin}) · a ${d.p_a} / b ${d.p_b} / c ${d.p_c} / d ${d.p_d}<br><span class="meta">모델 답과 판단이 다르면 needs_review로 두고 review_note에 차이를 남기세요</span>`;
 $("llm_assistance").value=d.llm_model}
$("file").onchange=load;$("filter").onchange=()=>applyFilter(false);$("prev").onclick=()=>{if(idx>0){idx--;show()}};$("next").onclick=()=>{if(idx<view.length-1){idx++;show()}};
$("jump").onclick=()=>{const j=view.findIndex(i=>!rows[i].review_status);if(j>=0){idx=j;show()}else msg("미작성 행이 없습니다")};
$("save").onclick=save;$("applyC").onclick=applyC;$("clearAns").onclick=()=>{ans="";renderChoices(rows[view[idx]])};
$("pApprove").onclick=()=>preset("approve");$("pReview").onclick=()=>preset("review");$("pReject").onclick=()=>preset("reject");
let zoom=0;
function setZoom(z,cx,cy){const box=$("imgbox"),img=$("img");const fw=img.clientWidth||1;
 const rx=cx==null?0.5:(box.scrollLeft+cx)/Math.max(1,box.scrollWidth),ry=cy==null?0.5:(box.scrollTop+cy)/Math.max(1,box.scrollHeight);
 zoom=Math.max(0,Math.min(6,z));if(!zoom){box.classList.remove("zoomed");img.style.width="";$("zl").textContent="맞춤";return}
 if(!box.classList.contains("zoomed")){box.classList.add("zoomed")}
 const base=Math.min(box.clientWidth/img.naturalWidth,box.clientHeight/img.naturalHeight);img.style.width=Math.round(img.naturalWidth*base*zoom)+"px";
 $("zl").textContent=zoom+"×";box.scrollLeft=rx*box.scrollWidth-(cx==null?box.clientWidth/2:cx);box.scrollTop=ry*box.scrollHeight-(cy==null?box.clientHeight/2:cy)}
$("img").onclick=e=>{const b=$("imgbox").getBoundingClientRect();if(zoom)setZoom(0);else setZoom(3,e.clientX-b.left,e.clientY-b.top)};
$("zin").onclick=()=>setZoom((zoom||1)+1);$("zout").onclick=()=>setZoom(zoom<=2?0:zoom-1);$("zfit").onclick=()=>setZoom(0);$("zcon").onclick=()=>$("imgbox").classList.toggle("contrast");
$("imgbox").addEventListener("wheel",e=>{if(!e.ctrlKey)return;e.preventDefault();const b=$("imgbox").getBoundingClientRect();setZoom(e.deltaY<0?(zoom||1)+1:(zoom<=2?0:zoom-1),e.clientX-b.left,e.clientY-b.top)},{passive:false});
document.addEventListener("keydown",e=>{if((e.ctrlKey||e.metaKey)&&e.key==="s"){e.preventDefault();save();return}
 const t=e.target.tagName;if(t==="INPUT"||t==="TEXTAREA"||t==="SELECT")return;const k=e.key.toLowerCase();
 if("1234".includes(k)&&k.length===1){ans="abcd"[+k-1];renderChoices(rows[view[idx]])}else if("abd".includes(k)&&k.length===1){ans=k;renderChoices(rows[view[idx]])}
 else if(k==="c"&&!e.ctrlKey){applyC()}else if(k==="enter"){e.preventDefault();preset("approve")}else if(k==="r")preset("review");else if(k==="x")preset("reject");else if(k==="+"||k==="=")setZoom((zoom||1)+1);else if(k==="-")setZoom(zoom<=2?0:zoom-1);else if(k==="0")setZoom(0);else if(k==="v")$("imgbox").classList.toggle("contrast");
 else if(k==="arrowright")$("next").click();else if(k==="arrowleft")$("prev").click()});
load();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    for kind, path in FILES.items():
        if kind == "recheck":
            continue                  # 재판정 파일은 대조 후에만 생긴다
        if not path.exists():
            print(f"{kind} 파일이 없다: {path}")
            return 1
    if not (DATA / "dev").is_dir():
        print(f"dev 이미지 폴더가 없다: {DATA / 'dev'}")
        return 1
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"http://127.0.0.1:{args.port}  (Ctrl+C로 종료)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
