#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""여러 번 실행한 결과를 한꺼번에 채점해 평균과 흔들림을 본다.

HCX는 temperature 0.2라 같은 코드로도 실행마다 답이 달라진다. 실측으로
한 문항이 v2에서 맞고 v3에서 틀린 경우가 여러 번 나왔다. 그래서 한 번
실행한 점수로 "좋아졌다/나빠졌다"를 말하면 노이즈를 성능으로 착각한다.

    python score_avg.py raw_a.json raw_b.json raw_c.json --gold gold_holdout_v1.json

문항별로 몇 번 맞았는지(안정/불안정)까지 같이 보여준다. 팀원과 비교하기
전에 **내 점수의 오차 범위**를 알아야 비교가 의미를 갖는다.
"""
import argparse
import statistics as st
from pathlib import Path

import score_official as S

KEYS = [
    ("answer_cov", "A. 답변 정확도", 0.50),
    ("retrieval_cov", "B. 검색 커버리지", 0.20),
    ("citation", "C. 출처", 0.15),
    ("grounding", "E. 근거 정합성", 0.15),
    ("composite", "── 종합점수", None),
]


def spread(vals):
    """평균과 폭. 표본이 2개뿐이어도 표준편차 때문에 죽지 않게 한다."""
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd, min(vals), max(vals)


def main():
    ap = argparse.ArgumentParser(description="여러 실행의 평균과 흔들림")
    ap.add_argument("raws", nargs="+")
    ap.add_argument("--gold", default=None)
    ap.add_argument("--show", type=int, default=12,
                    help="불안정한 문항을 몇 개까지 보여줄지")
    a = ap.parse_args()

    gold = S.find_gold(a.gold)
    print(f"정답지: {gold.name}")

    runs, per_q = [], {}
    for p in a.raws:
        rows = S.run(Path(p), gold)
        runs.append((Path(p).name, S.summarize(rows)))
        for r in rows:
            per_q.setdefault(r["question_id"], []).append(r)

    n = len(runs)
    print(f"\n{'='*70}")
    print(f"{n}회 실행 평균")
    print(f"{'='*70}")
    for name, _ in runs:
        print(f"  · {name}")
    print()

    for key, label, w in KEYS:
        vals = [s[key] for _, s in runs]
        m, sd, lo, hi = spread(vals)
        wt = f"가중 {w*100:.0f}%" if w else ""
        print(f"  {label:16} {m*100:5.1f}%  ± {sd*100:4.1f}p   "
              f"(최저 {lo*100:.1f} ~ 최고 {hi*100:.1f})  {wt}")

    print(f"\n  실행별 종합: "
          + " / ".join(f"{s['composite']*100:.1f}%" for _, s in runs))

    # ── 문항 안정성 ────────────────────────────────────────────────
    stable_ok, stable_bad, flaky = [], [], []
    for qid, rows in per_q.items():
        covs = [r["answer_cov"] for r in rows]
        if all(c >= 0.999 for c in covs):
            stable_ok.append(qid)
        elif all(c < 0.001 for c in covs):
            stable_bad.append(qid)
        else:
            flaky.append((qid, covs, rows[0]))

    print(f"\n{'='*70}")
    print("문항 안정성")
    print(f"{'='*70}")
    print(f"  항상 만점  {len(stable_ok):2d}개")
    print(f"  항상 0점   {len(stable_bad):2d}개  {stable_bad}")
    print(f"  실행마다 다름 {len(flaky):2d}개  ← 이 문항들의 1회 점수는 믿지 말 것")
    print()

    for qid, covs, r in sorted(flaky, key=lambda x: st.mean(x[1]))[:a.show]:
        rng = f"{min(covs)*100:.0f}~{max(covs)*100:.0f}%"
        print(f"  {qid}  ({r.get('domain')}/{r.get('difficulty')})  "
              f"평균 {st.mean(covs)*100:5.1f}%  폭 {rng}")
        print(f"    회차별: " + " → ".join(f"{c*100:.0f}%" for c in covs))
        q = (r.get("question") or "")[:72]
        print(f"    Q: {q}")

    print(f"\n{'='*70}")
    print("읽는 법")
    print(f"{'='*70}")
    print("  · ±폭이 팀원과의 점수 차보다 크면 그 차이는 아직 의미가 없다.")
    print("  · '항상 0점'은 진짜 결함이다. 여기부터 고칠 것.")
    print("  · '실행마다 다름'은 프롬프트·검색이 아니라 생성 변동일 가능성이 크다.")
    print()
    print("  ※ 이 도구는 **같은 코드로 여러 번 돌린 결과**를 넣어야 의미가 있다.")
    print("    코드가 다른 실행을 섞으면 코드 변경 효과가 '변동'으로 잘못 읽힌다.")


if __name__ == "__main__":
    main()
