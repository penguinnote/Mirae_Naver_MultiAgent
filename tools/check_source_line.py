# -*- coding: utf-8 -*-
"""근거 줄(※ 근거) 오프라인 검증

사용법:  python check_source_line.py [프로젝트경로] [raw_*.json]

원래 목적: HCX 호출 없이 근거 줄만 다시 만든다.

raw_h3_v2.json은 옛 _source_line(검색 상위 5개 나열)으로 만들어졌다.
같은 답변·같은 검색결과에 새 _source_line을 먹여서
(1) 무관한 근거가 빠지는지 (2) 펀드코드가 사라지는지 본다.

근거는 retrieved_context 헤더의 chunk_id를 chunks_final.jsonl에 되짚어
런타임 evidence와 같은 모양(text는 EVIDENCE_CHARS까지)으로 되살린다.
헤더를 파싱해 만들면 '· 절 제목 · 기준일'이 붙은 줄을 놓친다.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))
import agent as A  # noqa: E402

meta = []
with open(ROOT / "dataset" / "chunks_final.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            meta.append(json.loads(line))
md_of = {m["chunk_id"]: m for m in meta}

# ── Retriever.__init__의 색인 구축부 재현 (Chroma 없이) ──
fund_names = {}
for m in meta:
    fc = m.get("fund_code")
    if fc:
        fund_names.setdefault(fc, m.get("fund_name"))

_best = {}
for m in meta:
    doc = m.get("doc_id") or ""
    mm = A._CID_PAGE_RE.search(m.get("chunk_id") or "")
    if not doc.startswith("doc") or not mm:
        continue
    key = (int(mm.group(1)), int(mm.group(2)))
    if doc not in _best or key < _best[doc][0]:
        _best[doc] = (key, m.get("text") or "")
for doc, (_k, text) in _best.items():
    t = A._make_title(text)
    if len(t) >= 3:
        A._DOC_TITLES[doc] = t
for fc, nm in fund_names.items():
    if nm:
        A._DOC_TITLES.setdefault(fc, A._doc_label({"fund_name": nm}))

fund_series = {}
for name in fund_names.values():
    flat = A._NS_RE.sub("", A.N(name or ""))
    for mm in A._SERIES_RE.finditer(flat):
        fam, num = mm.group(1), mm.group(2)
        if len(fam) >= 3:
            fund_series.setdefault(fam, {})[num] = name
name_only = {k: dict(v) for k, v in fund_series.items()}
SERIES_MIN_HITS = 5
if fund_series:
    _fam_re = re.compile("(" + "|".join(re.escape(f) for f in fund_series) + r")(\d{3,4})(?!\d)")
    hits = {}
    for m in meta:
        for mm in _fam_re.finditer(A._NS_RE.sub("", A.N(m.get("text") or ""))):
            hits[(mm.group(1), mm.group(2))] = hits.get((mm.group(1), mm.group(2)), 0) + 1
    for (fam, num), n in hits.items():
        if n >= SERIES_MIN_HITS:
            fund_series.setdefault(fam, {}).setdefault(num, fam + num)

print("=" * 74)
print("[1] 번호 계열 목록 — 역질문이 부르는 이름")
for fam in sorted(fund_series):
    on, fn = sorted(name_only.get(fam, {})), sorted(fund_series[fam])
    if on != fn:
        print(f"  {fam}")
        print(f"    투자설명서만 : {on}")
        print(f"    본문 포함 후 : {fn}")
        dropped = sorted(n for (f_, n), c in hits.items()
                         if f_ == fam and c < SERIES_MIN_HITS)
        print(f"    {SERIES_MIN_HITS}회 미만 제외 : {dropped}")

# ── retrieved_context → evidence 복원 ──
CID = re.compile(r"^\[([^\]\s]+)\]")


def ev_of(ctx):
    ev = []
    for line in (ctx or "").splitlines():
        m = CID.match(line.strip())
        if not m:
            continue
        md = md_of.get(m.group(1))
        if not md:
            continue
        ev.append({
            "chunk_id": md["chunk_id"], "text": md["text"][:A.EVIDENCE_CHARS],
            "doc_id": md.get("doc_id"), "fund_name": md.get("fund_name"),
            "fund_code": md.get("fund_code"), "page": md.get("page"),
            "section": md.get("section"),
        })
    return ev


RAW = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "raw_h3_v2.json"
raw = json.load(open(RAW, encoding="utf-8"))
print("=" * 74)
print("[2] 근거 줄 재계산 (옛 → 새)")
changed = kr = miss = 0
for r in raw["results"]:
    ans = r.get("answer") or ""
    old = ans.split("※ 근거:")[1].strip() if "※ 근거:" in ans else "(없음)"
    ev = ev_of(r.get("retrieved_context"))
    if not ev:
        miss += 1
        continue
    new = A._source_line(ev, ans, r.get("question") or "").replace("\n\n※ 근거: ", "") or "(없음)"
    mark = " " if old == new else "*"
    if old != new:
        changed += 1
    if re.search(r"KR[0-9A-Z]{8,}", new):
        kr += 1
        mark = "!"
    print(f"{mark} {r['question_id']}  (근거 {len(ev)}개 중 {new.count('·') + 1}개)")
    print(f"    old: {old}")
    print(f"    new: {new}")
print("=" * 74)
print(f"바뀐 항목 {changed}개 / 펀드코드 노출 {kr}건 / 근거 복원 실패 {miss}건")
