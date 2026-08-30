# -*- coding: utf-8 -*-
"""정확도가 왜 90%에서 멈추는가 — 놓친 항목을 원인별로 분해한다.

정답지의 required 항목 하나하나에 대해 네 단계로 갈라 본다.

  코퍼스   그 표현이 dataset/chunks_final.jsonl 어딘가에 있는가
  검색     그 질문의 retrieved_context 안까지 들어왔는가
  답변     답변이 실제로 그 표현을 썼는가
  안정성   4번 실행에서 몇 번 맞혔는가 (0/4·4/4가 아니면 흔들리는 항목)

사용법:  python why90.py [프로젝트경로]
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))
import score_official as S  # noqa: E402

GOLD = json.load(open(ROOT / "gold_holdout_v3.json", encoding="utf-8"))
RUNS = ["raw_h3_v1.json", "raw_h3_v2.json", "raw_h3_v3.json", "raw_h3_v4.json"]

corpus = []
with open(ROOT / "dataset" / "chunks_final.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            corpus.append(json.loads(line).get("text") or "")
CORPUS = S.ns("\n".join(corpus))

runs = {}
for name in RUNS:
    p = ROOT / name
    if p.exists():
        runs[name] = {r["question_id"]: r for r in json.load(open(p, encoding="utf-8"))["results"]}

rows = []
for gq in GOLD["questions"]:
    qid = gq["question_id"]
    for i, req in enumerate(gq.get("required") or []):
        alts = req.get("alts") or [str(v) for v in req.get("values", [])]
        label = " / ".join(alts)[:44]
        w = req.get("weight", 1.0)

        in_corpus = any(S.ns(a) in CORPUS for a in alts)
        hits, in_ctx = [], False
        for name in RUNS:
            r = runs.get(name, {}).get(qid)
            if not r:
                continue
            hits.append(S.req_hit(req, r.get("answer") or ""))
            if S.req_hit(req, r.get("retrieved_context") or ""):
                in_ctx = True
        n_hit, n_run = sum(hits), len(hits)

        # evidence=False인 항목은 문서에서 인용할 말이 아니라 **에이전트가
        # 스스로 해야 할 말**(거절·한계 고지·안내 창구)이다. 코퍼스에
        # 없는 게 당연하므로 자료 부족으로 세면 안 된다.
        needs_doc = req.get("evidence", True)
        if n_hit == n_run:
            cause = "맞음"
        elif not needs_doc:
            cause = "⑤ 표현·태도 문제(자료 무관)"
        elif not in_corpus:
            cause = "① 코퍼스에 없음"
        elif not in_ctx:
            cause = "② 검색이 못 물어옴"
        elif n_hit == 0:
            cause = "③ 근거엔 있는데 안 씀"
        else:
            cause = "④ 실행마다 흔들림"
        rows.append({"qid": qid, "cat": gq.get("category"), "req": label, "w": w,
                     "corpus": in_corpus, "ctx": in_ctx, "hits": n_hit, "runs": n_run,
                     "cause": cause})

print("=" * 92)
print(f"{'문항':<7}{'카테고리':<10}{'요구 항목':<46}{'가중':>4} {'4회':>4}  원인")
print("-" * 92)
for r in rows:
    if r["cause"] == "맞음":
        continue
    print(f"{r['qid']:<7}{(r['cat'] or ''):<10}{r['req']:<46}{r['w']:>4} "
          f"{r['hits']}/{r['runs']:<2}  {r['cause']}")

print("=" * 92)
tot_w = sum(r["w"] for r in rows)
agg = {}
for r in rows:
    a = agg.setdefault(r["cause"], [0, 0.0])
    a[0] += 1
    a[1] += r["w"]
order = ["맞음", "⑤ 표현·태도 문제(자료 무관)", "④ 실행마다 흔들림",
         "③ 근거엔 있는데 안 씀", "② 검색이 못 물어옴", "① 코퍼스에 없음"]
print(f"{'원인':<24}{'항목':>5}{'가중합':>8}{'전체 대비':>10}")
for k in order:
    if k in agg:
        n, w = agg[k]
        print(f"{k:<24}{n:>5}{w:>8.1f}{w / tot_w * 100:>9.1f}%")
print("-" * 92)
print(f"{'합계':<24}{len(rows):>5}{tot_w:>8.1f}")

print()
print("실행별 A (같은 코드가 아니라 각 커밋 시점)")
for name in RUNS:
    if name not in runs:
        continue
    tot = hit = 0.0
    for gq in GOLD["questions"]:
        r = runs[name].get(gq["question_id"])
        for req in (gq.get("required") or []):
            w = req.get("weight", 1.0)
            tot += w
            if r and S.req_hit(req, r.get("answer") or ""):
                hit += w
    print(f"  {name:<18} {hit / tot * 100:5.1f}%")
