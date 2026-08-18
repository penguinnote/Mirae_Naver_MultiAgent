#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
평가셋 실행기 — 검색 정확도(Recall) 측정

지금 우리에겐 "표 청크 3,863개"라는 숫자는 있어도 "정확도 몇 %"라는 숫자가 없다.
발표에서 주장할 수 있는 건 후자다. 이 스크립트가 그 숫자를 만든다.

무엇을 재는가
------------
에이전트가 아직 없으므로 **검색 계층만** 잰다. 그래도 충분히 의미가 있다.
검색이 근거를 못 가져오면 그 뒤에 무슨 LLM을 붙여도 정답이 안 나오기 때문이다.
즉 여기서 나오는 수치가 **최종 정확도의 천장**이다.

지표가 두 개인 이유
------------------
처음엔 "내가 지정한 청크가 나왔나(chunk-id 일치)"만 쟀는데, 이게 실제보다 성능을
낮게 잡는다는 걸 발견했다. 같은 사실이 여러 문서에 중복돼 있기 때문이다.

  · 투자설명서 84개가 거의 같은 법정 문구를 공유한다 (환매 관련 청크만 83개 펀드에 513개)
  · 제도 문서도 "세액공제 900만원"이 doc6·doc41·doc27·doc55에 전부 나온다

내가 doc21을 정답으로 지정했는데 검색이 doc39를 가져왔다면, chunk-id 기준으로는
'실패'지만 doc39에도 똑같이 "만70세~79세는 4.4%"라고 쓰여 있다. LLM은 이걸로
정답을 만든다. 실제로 BM25 단독 기준 52.8% → 80.6%로 갈렸다.

그래서 주 지표를 바꿨다.

  근거 충족률 (주 지표)  : 상위 k개 청크를 합쳤을 때 answer_points가 전부 들어있나
                          → LLM이 정답을 만들 수 있는가. 이게 진짜 천장이다.
  청크 적중률 (보조 지표) : 내가 지정한 바로 그 청크가 나왔나
                          → 가장 좋은 출처를 집었는지. 인용 품질 진단용.
  MRR                   : 첫 적중이 몇 번째였나. 순위 품질.

engine_only=True 문항(집계·계산으로만 풀리는 것)은 검색 채점에서 뺀다.
문서에 답이 통문장으로 없으니 검색을 탓할 수 없다.

SQL 문항은 검색이 아니라 expected_sql이 실제로 결과를 뱉는지 확인한다.

사용법
------
data_test 폴더에서 실행한다.

    python run_eval.py --compare
        hybrid/vector/bm25 × k=5,10,20 을 한 번에 비교. 여기서 시작할 것.
        질의 임베딩을 문항당 1회만 하므로 따로 9번 돌리는 것보다 훨씬 빠르다.

    python run_eval.py
        하이브리드 k=5 상세 (문항별 ✅/❌와 누락된 근거를 보여준다)

    python run_eval.py --detail
        실패 문항이 무엇을 대신 가져왔는지까지 출력

    python run_eval.py --out result.json
        결과 저장. 개선 전후 비교에 쓴다.

zsh 주의: 명령 뒤에 # 주석을 붙이지 말 것. zsh는 이를 인자로 넘겨 에러가 난다.

BM25 인덱스(72MB)와 Chroma를 한 번만 로드하고 40문항을 돌린다.
search.py를 40번 호출하면 로딩만 40번이라 몇 분씩 걸린다.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# search.py에서 검증된 로직을 그대로 재사용한다.
# 평가용으로 검색 로직을 따로 구현하면 "평가는 통과하는데 실제로는 다른 코드"가 된다.
from search import (  # noqa: E402
    RRF_K, POOL, build_bm25, embed_query, tokenize, matches,
)


def norm(s: str) -> str:
    """표기 흔들림 흡수: '900만원' / '900만 원' / '900,000' 을 같게 본다."""
    return re.sub(r"[\s,]", "", s)


def hybrid_search(query, col, bm25, meta, k, mode="hybrid",
                  fund=None, doc_type=None, ctype=None):
    """search.py main()의 검색 부분과 동일한 로직. 인덱스는 호출자가 들고 있는다."""
    ranks: dict[str, dict[str, int]] = {}

    if mode != "bm25":
        res = col.query(query_embeddings=[embed_query(query)], n_results=POOL,
                        include=["metadatas"])
        for i, (cid, md) in enumerate(zip(res["ids"][0], res["metadatas"][0])):
            if not matches(md, fund, doc_type, ctype):
                continue
            ranks.setdefault(cid, {})["vec"] = i + 1

    if mode != "vector":
        scores = bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:POOL]
        for rank, i in enumerate(order):
            if scores[i] <= 0:
                break
            m = meta[i]
            md = {kk: vv for kk, vv in m.items() if kk != "text"}
            md["doc_ids"] = "," + ",".join(md.get("doc_ids") or []) + ","
            if not matches(md, fund, doc_type, ctype):
                continue
            ranks.setdefault(m["chunk_id"], {})["bm25"] = rank + 1

    fused = sorted(ranks.items(),
                   key=lambda kv: -sum(1.0 / (RRF_K + r) for r in kv[1].values()))
    return [cid for cid, _ in fused[:k]]


def run_compare(questions, col, bm25, meta, text_of, a):
    """hybrid / vector / bm25 × k=5,10,20 을 한 번에 비교한다.

    핵심은 **질의 임베딩을 문항당 1회만** 한다는 것이다.
    모드를 바꿔가며 따로 돌리면 같은 질문을 9번씩 임베딩하게 되는데,
    벡터는 모드·k와 무관하게 같은 값이므로 낭비다. CLOVA 호출도 아끼고 훨씬 빠르다.
    """
    KS = [5, 10, 20]
    MODES = ["hybrid", "vector", "bm25"]
    targets = [q for q in questions
               if q["expected_chunks"] and not q.get("engine_only")]

    print(f"비교 모드: {len(targets)}문항 × 3모드 × k{KS}")
    print(f"질의 임베딩은 문항당 1회만 합니다 (총 {len(targets)}회)\n")

    # 문항별로 각 검색기의 전체 순위를 한 번씩만 구해둔다.
    per_q = []
    for i, qq in enumerate(targets, 1):
        vec_rank, bm_rank = {}, {}

        emb = embed_query(qq["question"])
        res = col.query(query_embeddings=[emb], n_results=POOL,
                        include=["metadatas"])
        for j, (cid, md) in enumerate(zip(res["ids"][0], res["metadatas"][0])):
            if matches(md, None, None, None):
                vec_rank[cid] = j + 1

        scores = bm25.get_scores(tokenize(qq["question"]))
        order = sorted(range(len(scores)), key=lambda x: -scores[x])[:POOL]
        for r, x in enumerate(order):
            if scores[x] <= 0:
                break
            bm_rank[meta[x]["chunk_id"]] = r + 1

        per_q.append((qq, vec_rank, bm_rank))
        print(f"  [{i:2d}/{len(targets)}] {qq['question_id']} 검색 완료", end="\r")
    print(" " * 60, end="\r")

    def fuse(vec_rank, bm_rank, mode, k):
        ranks = {}
        if mode != "bm25":
            for cid, r in vec_rank.items():
                ranks.setdefault(cid, {})["vec"] = r
        if mode != "vector":
            for cid, r in bm_rank.items():
                ranks.setdefault(cid, {})["bm25"] = r
        fused = sorted(ranks.items(),
                       key=lambda kv: -sum(1.0 / (RRF_K + r) for r in kv[1].values()))
        return [cid for cid, _ in fused[:k]]

    grid = {}
    for mode in MODES:
        for k in KS:
            ev = cid = 0
            for qq, vr, br in per_q:
                got = fuse(vr, br, mode, k)
                blob = norm(" ".join(text_of.get(c, "") for c in got))
                ev += all(norm(ap) in blob for ap in qq["answer_points"])
                cid += bool(set(got) & set(qq["expected_chunks"]))
            grid[(mode, k)] = (ev, cid)

    n = len(per_q)
    print("=" * 62)
    print(f"검색기 × k 비교 — {n}문항 기준")
    print("=" * 62)
    print("\n  ▶ 근거 충족률 (주 지표: LLM이 정답을 만들 수 있는가)\n")
    print(f"    {'':10}" + "".join(f"k={k:<8}" for k in KS))
    for mode in MODES:
        row = "".join(f"{grid[(mode,k)][0]/n:>6.1%}   " for k in KS)
        print(f"    {mode:<10}{row}")
    print("\n  청크 적중률 (보조: 지정한 바로 그 출처를 집었나)\n")
    print(f"    {'':10}" + "".join(f"k={k:<8}" for k in KS))
    for mode in MODES:
        row = "".join(f"{grid[(mode,k)][1]/n:>6.1%}   " for k in KS)
        print(f"    {mode:<10}{row}")

    # ── 자동 해석: 숫자만 보고 판단하지 않도록 결론을 붙인다
    h5, v5, b5 = (grid[(m, 5)][0] / n for m in MODES)
    h20 = grid[("hybrid", 20)][0] / n
    print("\n" + "-" * 62)
    print("  해석")
    if h5 >= max(v5, b5) + 0.02:
        print(f"    · 하이브리드({h5:.1%})가 벡터({v5:.1%})·BM25({b5:.1%})보다 높습니다.")
        print("      RRF 융합이 제 역할을 하고 있습니다.")
    elif abs(h5 - max(v5, b5)) < 0.02:
        weak = "벡터" if v5 < b5 else "BM25"
        print(f"    · 하이브리드({h5:.1%})가 단독 최고({max(v5,b5):.1%})와 거의 같습니다.")
        print(f"      {weak} 쪽이 기여를 거의 못 하고 있다는 뜻입니다. 확인이 필요합니다.")
    else:
        print(f"    · 하이브리드({h5:.1%})가 단독보다 낮습니다. RRF가 좋은 결과를")
        print("      밀어내고 있습니다. POOL/RRF_K 조정을 검토하세요.")

    # k를 올렸을 때의 이득만 보고 "후보에 없다"고 단정하면 안 된다.
    # 다른 검색기가 같은 k에서 더 찾아낸다면, 근거는 분명히 코퍼스에 있고
    # 융합 방식이 그걸 못 끌어올리는 것이다. 둘을 반드시 나눠 봐야 한다.
    best20 = max(grid[(m, 20)][0] / n for m in MODES)
    best20_mode = max(MODES, key=lambda m: grid[(m, 20)][0])
    gain = h20 - h5

    print(f"    · k를 5→20으로 올려도 하이브리드는 {h5:.1%} → {h20:.1%} ({gain:+.1%}p)."
          if gain < 0.10 else
          f"    · k를 5→20으로 올리면 {h5:.1%} → {h20:.1%} ({gain:+.1%}p).")

    if best20 > h20 + 0.02:
        print(f"      그런데 {best20_mode} 단독은 k=20에서 {best20:.1%}로 더 높습니다.")
        print("      → 근거는 코퍼스에 분명히 있습니다. 청킹·임베딩 문제가 아니라")
        print("        RRF가 깊이에서 슬롯을 나눠 쓰느라 못 끌어올리는 것입니다.")
        print("        (하이브리드는 상위에서 강하고 깊이에서 손해 보는 게 정상입니다)")
        print("      → 조치가 필요하면 POOL 확대나 가중 RRF를 검토하세요.")
        print("        다만 k=5에서 이미 하이브리드가 가장 높으므로, 운영은")
        print("        k=5로 두는 편이 컨텍스트도 짧고 정확도도 최선입니다.")
    elif gain >= 0.10:
        print("      근거가 후보 안에는 있는데 순위가 밀린 것입니다.")
        print("      → 리랭커(--rerank)로 상당 부분 회수됩니다. 청킹은 건드릴 필요 없습니다.")
    else:
        print("      어떤 검색기로도 k=20에서 더 못 찾습니다.")
        print("      → 이제야 '후보에 없다'고 말할 수 있습니다. 청킹·임베딩을 보세요.")


def main():
    ap = argparse.ArgumentParser(description="평가셋으로 검색 정확도 측정")
    ap.add_argument("--evalset", default="./evalset_v1.json")
    ap.add_argument("--db", default="./dataset/chroma")
    ap.add_argument("--collection", default="pension")
    ap.add_argument("--chunks", default="./dataset/chunks_final.jsonl")
    ap.add_argument("--bm25-cache", default="./dataset/bm25.pkl")
    ap.add_argument("--fees-db", default="./dataset/fund_fees.sqlite")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--vector-only", action="store_true")
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--detail", action="store_true", help="실패 문항 상세 출력")
    ap.add_argument("--compare", action="store_true",
                    help="hybrid/vector/bm25 × k=5,10,20 을 한 번에 비교. "
                         "질의 임베딩을 1회만 하므로 API 호출이 1/9로 줄어든다.")
    ap.add_argument("--out", help="결과 JSON 저장 경로")
    a = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    mode = "vector" if a.vector_only else "bm25" if a.bm25_only else "hybrid"

    payload = json.loads(Path(a.evalset).read_text(encoding="utf-8"))
    questions = payload["questions"]

    print(f"평가셋: {a.evalset} ({len(questions)}문항) | 모드: {mode} | k={a.k}")
    print("인덱스 로딩 중… (BM25 72MB, 처음엔 20~30초 걸립니다)")
    t0 = time.monotonic()

    import chromadb
    col = chromadb.PersistentClient(path=a.db).get_collection(a.collection)
    bm25, meta = build_bm25(Path(a.chunks), Path(a.bm25_cache))
    print(f"로딩 완료 ({time.monotonic() - t0:.1f}초)\n")

    # 청크 본문을 들고 있어야 근거 충족을 판정할 수 있다.
    text_of = {m["chunk_id"]: m["text"] for m in meta}

    if a.compare:
        run_compare(questions, col, bm25, meta, text_of, a)
        return

    results = []
    by_type = defaultdict(lambda: {"n": 0, "ev": 0, "cid": 0})
    by_diff = defaultdict(lambda: {"n": 0, "ev": 0})

    # ── 검색 문항 채점 ────────────────────────────────────────────────
    searchable = [q for q in questions
                  if q["expected_chunks"] and not q.get("engine_only")]
    for i, qq in enumerate(searchable, 1):
        got = hybrid_search(qq["question"], col, bm25, meta, a.k, mode)
        exp = set(qq["expected_chunks"])

        # 주 지표: 상위 k개를 합친 본문이 answer_points를 전부 담고 있나
        blob = norm(" ".join(text_of.get(c, "") for c in got))
        missing = [ap for ap in qq["answer_points"] if norm(ap) not in blob]
        evidence_ok = not missing

        # 보조 지표: 지정한 청크가 나왔나
        rank = next((j + 1 for j, c in enumerate(got) if c in exp), None)

        rec = {
            "question_id": qq["question_id"],
            "question": qq["question"],
            "query_type": qq["query_type"],
            "difficulty": qq["difficulty"],
            "expected": sorted(exp),
            "retrieved": got,
            "evidence_ok": evidence_ok,
            "missing_points": missing,
            "chunk_hit": rank is not None,
            "chunk_hit_all": exp.issubset(set(got)),
            "first_hit_rank": rank,
            "rr": 1.0 / rank if rank else 0.0,
        }
        results.append(rec)

        t = by_type[qq["query_type"]]
        t["n"] += 1
        t["ev"] += evidence_ok
        t["cid"] += rec["chunk_hit"]
        d = by_diff[qq["difficulty"]]
        d["n"] += 1
        d["ev"] += evidence_ok

        # ✅ 근거O 청크O / 🟢 근거O 청크X(다른 출처가 답을 담음) / 🟡 근거X 청크O / ❌ 둘 다
        mark = ("✅" if rec["chunk_hit"] else "🟢") if evidence_ok else \
               ("🟡" if rec["chunk_hit"] else "❌")
        pos = f"@{rank}" if rank else "  "
        miss = f"  누락:{','.join(missing)}" if missing else ""
        print(f"  [{i:2d}/{len(searchable)}] {mark}{pos:>4} {qq['question_id']} "
              f"{qq['question'][:40]}{miss}")

    # ── SQL 문항 채점 ────────────────────────────────────────────────
    sql_qs = [q for q in questions if q.get("expected_sql")]
    sql_ok = sql_fail = 0
    sql_detail = []
    fees = Path(a.fees_db)
    if sql_qs and fees.exists():
        print()
        con = sqlite3.connect(fees)
        cur = con.cursor()
        tables = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for qq in sql_qs:
            sql = qq["expected_sql"]
            tbl = "fund_fees" if "fund_fees" in sql else sql.split("FROM")[1].split()[0]
            if tbl not in tables:
                sql_detail.append((qq["question_id"], "SKIP", f"{tbl} 테이블 없음"))
                continue
            try:
                rows = cur.execute(sql).fetchall()
                if rows:
                    sql_ok += 1
                    sql_detail.append((qq["question_id"], "OK", f"{len(rows)}행"))
                else:
                    sql_fail += 1
                    sql_detail.append((qq["question_id"], "EMPTY", "0행 — 근거 없음"))
            except Exception as e:  # noqa: BLE001
                sql_fail += 1
                sql_detail.append((qq["question_id"], "ERROR", str(e)[:60]))
        con.close()
        print("SQL 문항 검증:")
        for qid, st, msg in sql_detail:
            icon = {"OK": "✅", "EMPTY": "❌", "ERROR": "❌", "SKIP": "⏭ "}[st]
            print(f"  {icon} {qid}  {msg}")

    # ── 요약 ────────────────────────────────────────────────────────
    n = len(results)
    ev_ok = sum(r["evidence_ok"] for r in results)
    cid_ok = sum(r["chunk_hit"] for r in results)
    mrr = sum(r["rr"] for r in results) / n if n else 0

    print("\n" + "=" * 70)
    print(f"검색 정확도 ({mode}, k={a.k}) — 검색 채점 대상 {n}문항")
    print("=" * 70)
    print(f"  ▶ 근거 충족률   {ev_ok}/{n}  =  {ev_ok/n:.1%}"
          f"   ← 주 지표. LLM이 정답을 만들 수 있는 상태인가")
    print(f"    청크 적중률   {cid_ok}/{n}  =  {cid_ok/n:.1%}"
          f"   ← 보조. 지정한 바로 그 출처를 집었나")
    print(f"    MRR          {mrr:.3f}"
          f"           ← 순위 품질")

    print("\n  유형별 (근거 충족 / 전체)")
    for t, v in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
        bar = "█" * round(v["ev"] / v["n"] * 20)
        print(f"    {t:<12} {v['ev']:>2}/{v['n']:<2} {v['ev']/v['n']:>6.0%} {bar}"
              f"   (청크 {v['cid']}/{v['n']})")

    print("\n  난이도별 (근거 충족 / 전체)")
    for d in ["하", "중", "상"]:
        if d in by_diff:
            v = by_diff[d]
            print(f"    {d}  {v['ev']:>2}/{v['n']:<2}  {v['ev']/v['n']:.0%}")

    if sql_qs:
        print(f"\n  SQL 문항: {sql_ok}개 성공 / {sql_fail}개 실패")

    misses = [r for r in results if not r["evidence_ok"]]
    if misses:
        print(f"\n  ❌ 근거를 못 찾은 문항 {len(misses)}개 — 여기가 개선 대상입니다")
        for r in misses:
            print(f"     {r['question_id']}  {r['question'][:46]}")
            print(f"        누락된 근거: {r['missing_points']}")
            if a.detail:
                print(f"        기대 청크: {r['expected']}")
                print(f"        실제 상위: {r['retrieved'][:5]}")

    only_alt = [r for r in results if r["evidence_ok"] and not r["chunk_hit"]]
    if only_alt:
        print(f"\n  🟢 지정 청크는 아니지만 다른 출처가 답을 담은 문항 {len(only_alt)}개")
        print("     (내용은 맞으므로 실패가 아님. 중복 문서가 많다는 뜻)")
        for r in only_alt:
            print(f"     {r['question_id']}  {r['question'][:46]}")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "mode": mode, "k": a.k,
            "evidence_rate": ev_ok / n if n else 0,
            "chunk_hit_rate": cid_ok / n if n else 0,
            "mrr": mrr,
            "by_type": {t: dict(v) for t, v in by_type.items()},
            "sql_ok": sql_ok, "sql_fail": sql_fail,
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n결과 저장: {a.out}")

    print("\n다음에 무엇을 볼 것인가")
    print("  · --vector-only / --bm25-only 를 각각 돌려 세 숫자를 비교하세요.")
    print("    하이브리드가 둘 다보다 높아야 RRF가 제 역할을 하는 겁니다.")
    print("    한쪽과 같다면 다른 쪽이 기여를 못 하고 있다는 뜻입니다.")
    print("  · -k 10 으로 올려보세요. 근거 충족률이 크게 오르면 '순위' 문제라")
    print("    리랭커(--rerank)로 해결됩니다. 안 오르면 애초에 후보에 없는")
    print("    것이라 청킹·임베딩을 손봐야 합니다. 이 구분이 중요합니다.")
    print("  · ❌ 문항의 '누락된 근거'를 보면 무엇이 안 잡히는지 바로 나옵니다.")


if __name__ == "__main__":
    main()
