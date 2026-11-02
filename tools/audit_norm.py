"""채점기 정규화의 오탐 검사.

ns()는 공백을 전부 지우고 부분문자열로 맞춘다. 한국어 띄어쓰기가 자유로워
필요한 처리지만, 부작용이 둘 있다.

  ① 문장 경계를 넘어 붙는다.
     "…가능합니다 중도인출은…" → "가능합니다중도인출은"
     서로 다른 문장의 조각이 이어져 없는 표현이 생길 수 있다.
  ② 부정문 안에서 매치된다.
     정답 표현이 "…가 아닙니다" 안에 들어 있어도 맞은 것으로 센다.

원문 위치를 되짚어 두 경우를 찾아낸다.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("so", "score_official.py")
m = importlib.util.module_from_spec(spec)
sys.modules["so"] = m
spec.loader.exec_module(m)

GOLD = json.load(open(
    "/mnt/user-data/uploads/mirae_integrated_blind_v1_ANSWERS_ONLY/"
    "gold_private_integrated_v1.json", encoding="utf-8"))
RAW = json.load(open(
    "/mnt/user-data/uploads/dev--data_test/raw_penguinnote_integrated_v2.json",
    encoding="utf-8"))
raw_by = {r["question_id"]: r for r in RAW["results"]}

# 문장 끝으로 볼 문자들
SENT_END = set(".!?\n•")


def norm_with_map(s: str):
    """공백을 지우면서 원문 인덱스를 함께 남긴다."""
    out, idx = [], []
    for i, ch in enumerate(s or ""):
        if ch.isspace():
            continue
        out.append(ch.lower())
        idx.append(i)
    return "".join(out), idx


def find_spans(alt: str, text: str):
    """정규화 매치가 원문에서 차지한 구간들을 돌려준다."""
    n, idx = norm_with_map(text)
    a = m.ns(alt)
    if not a:
        return []
    spans = []
    for mm in re.finditer(re.escape(a), n):
        s, e = idx[mm.start()], idx[mm.end() - 1]
        spans.append((s, e, text[s:e + 1]))
    return spans


cross, negated = [], []

for gq in GOLD["questions"]:
    qid = gq["question_id"]
    r = raw_by.get(qid)
    if not r:
        continue
    ans = r.get("answer") or ""
    for req in gq.get("required", []):
        if req.get("type") != "text":
            continue
        for alt in req.get("alts", []):
            for s, e, orig in find_spans(alt, ans):
                # ① 매치 구간에 문장부호가 섞여 있으면 경계를 넘은 것
                if any(ch in SENT_END for ch in orig):
                    cross.append((qid, alt, orig[:70]))
                # ② 매치 직후 30자 안에 부정 표현이 오면 의심
                tail = ans[e + 1:e + 31]
                if re.search(r"(아닙|아니라|불가|없습니다|없음|않습니다|않음|"
                             r"제외|확인할 수 없)", tail):
                    negated.append((qid, alt, (orig + "…" + tail)[:90]))

print("=" * 72)
print("① 문장 경계를 넘어 매치된 항목")
print("=" * 72)
if not cross:
    print("  없음 ✅")
for qid, alt, orig in cross:
    print(f"  {qid}  alt={alt!r}")
    print(f"     원문: {orig!r}")

print()
print("=" * 72)
print("② 매치 직후에 부정 표현이 오는 항목 (사람 확인 필요)")
print("=" * 72)
if not negated:
    print("  없음 ✅")
for qid, alt, s in negated:
    print(f"  {qid}  alt={alt!r}")
    print(f"     {s!r}")
