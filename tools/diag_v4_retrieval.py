#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""공식 v4 30문항의 **검색만** 재본다. HCX 답변 생성은 하지 않는다.

검색은 임베딩+BM25라 온도가 없어 **결정적**이다. 그래서 이 지표는 실행마다
흔들리지 않고, 검색 쪽 수정의 효과를 노이즈 없이 그대로 보여준다.
질의 임베딩만 부르므로 토큰도 거의 안 든다.

    python diag_v4_retrieval.py

정답지의 sources(fund_code + page)가 상위 근거 안에 들어왔는지 센다.
"""
import json
import os
import sys
from pathlib import Path

GOLD_CANDIDATES = [
    Path("gold_private_v4.json"),
    Path.home() / "Downloads/mirae_blind_eval_v4_EVALUATOR_PRIVATE/gold_private_v4.json",
]


def find_gold():
    for p in GOLD_CANDIDATES:
        if p.exists():
            return p
    sys.exit("gold_private_v4.json을 찾지 못했습니다. --gold로 경로를 주세요.")


def main():
    gold_p = Path(sys.argv[sys.argv.index("--gold") + 1]) if "--gold" in sys.argv else find_gold()
    gold = json.loads(gold_p.read_text(encoding="utf-8"))
    print(f"정답지: {gold_p}")

    import agent
    R = agent.Retriever()

    tot_src = hit_src = 0
    rows = []
    for q in gold["questions"]:
        st = {"question": q["question"], "trace": []}
        agent.route(st)
        R(st)          # query_emb는 Retriever가 알아서 만든다
        ev = st.get("evidence") or []

        got = set()
        for e in ev:
            got.add((e.get("fund_code"), e.get("page")))

        want = [(s.get("fund_code"), s.get("page")) for s in q.get("sources", [])]
        found = [w for w in want if w in got]
        # 쪽이 달라도 같은 펀드 문서를 봤으면 절반은 맞은 것으로 따로 센다
        same_fund = [w for w in want
                     if w not in got and any(g[0] == w[0] for g in got)]

        tot_src += len(want)
        hit_src += len(found)
        rows.append({
            "qid": q["question_id"], "cat": q["category"],
            "want": len(want), "hit": len(found), "fund_only": len(same_fund),
            "split": next((t for t in st["trace"] if "나눠 검색" in t), ""),
            "n_ev": len(ev),
        })

    print(f"\n{'문항':12} {'카테고리':22} {'정답출처':>6} {'적중':>4} {'펀드만':>5}")
    for r in rows:
        mark = "" if r["hit"] == r["want"] else ("  ◀ 일부" if r["hit"] else "  ◀ 전무")
        print(f"{r['qid']:12} {r['cat']:22} {r['want']:>6} {r['hit']:>4} "
              f"{r['fund_only']:>5}{mark}")

    full = sum(1 for r in rows if r["hit"] == r["want"])
    none = sum(1 for r in rows if r["hit"] == 0)
    print(f"\n정답 출처(펀드+쪽) 재현율  {hit_src}/{tot_src} = {hit_src/tot_src*100:.1f}%")
    print(f"모든 출처를 찾은 문항      {full}/{len(rows)}")
    print(f"하나도 못 찾은 문항        {none}/{len(rows)}")

    split = [r for r in rows if r["split"]]
    if split:
        print(f"\n분할 검색이 걸린 문항 {len(split)}개")
        for r in split:
            print(f"  {r['qid']}  {r['split'][10:]}")

    print("\n※ 이 수치는 결정적이다. 수정 전후로 돌려 그대로 비교하면 된다.")


if __name__ == "__main__":
    main()
