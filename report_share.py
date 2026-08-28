#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""채점 결과를 **팀에 그대로 붙여넣을 수 있는 마크다운 한 장**으로 만든다.

score_official.py의 콘솔 출력은 터미널용이라 상위 15개만 보여주고 줄바꿈도
거칠다. 이 스크립트는 같은 채점 함수를 그대로 쓰되, 전 문항을 표로 정리하고
카테고리별 집계를 붙인다.

    python report_share.py --raw raw_holdout_v2.json --gold gold_holdout_v2.json
    python report_share.py --raw ... --gold ... --all        (전 문항 정답까지)
    python report_share.py --raw ... --gold ... --hide-gold  (정답 가림)
    python report_share.py --raw ... --gold ... --compare raw_이전.json

⚠ --hide-gold 를 빼면 **정답과 채점 키워드가 그대로 문서에 들어간다.**
   홀드아웃 평가셋을 팀원과 공유하면 그 평가셋은 더 이상 홀드아웃이 아니다.
   점수만 공유할 거면 --hide-gold 를 쓰는 쪽이 맞다.
"""
import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import score_official as S

BAR = "█"


def pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def one_line(s, n=200):
    s = " ".join((s or "").split())
    return (s[:n] + "…") if len(s) > n else s


def git_head():
    try:
        h = subprocess.run(["git", "log", "-1", "--format=%h %s"],
                           capture_output=True, text=True, timeout=10)
        return h.stdout.strip() or "(git 정보 없음)"
    except Exception:
        return "(git 정보 없음)"


def group(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r.get(key) or "미분류", []).append(r)
    return out


def bar(x, width=12):
    if x is None:
        return ""
    n = round(x * width)
    return BAR * n + "·" * (width - n)


def build(rows, gold_path, raw_path, hide_gold, cmp_rows=None,
          show_all=False):
    s = S.summarize(rows)
    L = []
    a = L.append

    a(f"# 평가 결과 — {Path(gold_path).stem}")
    a("")
    a(f"- 평가셋: `{Path(gold_path).name}` ({s['n']}문항)")
    a(f"- 실행 결과: `{Path(raw_path).name}`")
    a(f"- 코드 버전: `{git_head()}`")
    a(f"- 작성: {datetime.now():%Y-%m-%d %H:%M}")
    a("")

    a("## 종합")
    a("")
    a("| 지표 | 점수 | 가중 |")
    a("|---|---:|---:|")
    a(f"| A. 답변 정확도 | {pct(s['answer_cov'])} | 50% |")
    a(f"| B. 검색 커버리지 | {pct(s['retrieval_cov'])} | 20% |")
    a(f"| C. 출처 | {pct(s['citation'])} | 15% |")
    a(f"| &nbsp;&nbsp;· 원문 문서 | {pct(s['source_recall'])} | |")
    a(f"| &nbsp;&nbsp;· locator | {pct(s['locator_recall'])} | |")
    a(f"| E. 근거 정합성 | {pct(s['grounding'])} | 15% |")
    a(f"| **종합** | **{pct(s['composite'])}** | |")
    a("")
    if cmp_rows:
        cs = S.summarize(cmp_rows)
        d = (s["composite"] - cs["composite"]) * 100
        a(f"이전 실행 대비 **{pct(cs['composite'])} → {pct(s['composite'])} "
          f"({d:+.1f}p)**")
        a("")

    a("### 상태 분포")
    a("")
    names = {"AUTO_CORRECT": "자동 정답", "PARTIAL_REVIEW": "부분 정답",
             "SEMANTIC_REVIEW": "표현 불일치(사람 확인 필요)",
             "CONTRADICTION_REVIEW": "모순 표현 감지",
             "INFRA_FAIL": "실행 실패", "MISSING_FROM_RUN": "응답 없음"}
    for k, v in sorted(s["states"].items(), key=lambda x: -x[1]):
        a(f"- {names.get(k, k)} — {v}문항")
    a("")
    a("> 자동 채점은 오답을 **확정하지 않는다.** 정답인데 표현이 달라 문자열이 "
      "안 맞는 경우가 있어서, 만점이 아닌 문항은 사람이 한 번 본다.")
    a("")

    for key, title in (("category", "카테고리별"), ("difficulty", "난이도별")):
        a(f"## {title}")
        a("")
        a(f"| {title[:-1]} | 문항 | 답변 정확도 | |")
        a("|---|---:|---:|---|")
        for k, sub in sorted(group(rows, key).items(),
                             key=lambda x: -S._avg([r["answer_cov"] for r in x[1]])):
            v = S._avg([r["answer_cov"] for r in sub])
            a(f"| {k} | {len(sub)} | {pct(v)} | `{bar(v)}` |")
        a("")

    a("## 문항별")
    a("")
    a("| 문항 | 카테고리 | 난이도 | 정확도 | 상태 |")
    a("|---|---|---|---:|---|")
    for r in rows:
        mark = "✅" if r["answer_cov"] >= 0.999 else (
            "⚠️" if r["answer_cov"] > 0 else "❌")
        a(f"| {r['question_id']} | {r.get('category','')} | "
          f"{r.get('difficulty','')} | {mark} {pct(r['answer_cov'])} | "
          f"{names.get(r['state'], r['state'])} |")
    a("")

    bad = [r for r in rows if r["answer_cov"] < 0.999
           or r.get("contradiction_hits")]
    shown = rows if show_all else bad
    a(f"## {'전 문항 상세' if show_all else f'만점이 아닌 문항 {len(bad)}개'}")
    a("")
    if not shown:
        a("없음.")
    for r in shown:
        mark = "✅" if (r["answer_cov"] >= 0.999
                        and not r.get("contradiction_hits")) else (
            "⚠️" if r["answer_cov"] > 0 else "❌")
        a(f"### {mark} {r['question_id']} · {r.get('category','')} · "
          f"{pct(r['answer_cov'])}")
        a("")
        a(f"**질문**")
        a("")
        a(f"> {one_line(r.get('question'), 400)}")
        a("")
        # 정답을 답변보다 **먼저** 놓는다. 채점 결과를 읽을 때
        # "무엇이 맞는 답이었나"를 먼저 알아야 우리 답을 판단할 수 있다.
        if not hide_gold and r.get("gold_answer"):
            a(f"**정답**")
            a("")
            a(f"> {one_line(r.get('gold_answer'), 400)}")
            a("")
        a(f"**우리 답변**")
        a("")
        a(f"> {one_line(r.get('answer'), 500)}")
        a("")
        if r.get("missing"):
            if hide_gold:
                a(f"**누락** 필수 항목 {len(r['missing'])}개 "
                  f"(내용은 정답지 보호를 위해 생략)")
            else:
                for m in r["missing"]:
                    a(f"**누락**(가중치 {m.get('weight')}) "
                      f"— 다음 중 하나가 답변에 있어야 함: `{m.get('want')}`")
            a("")
        if r.get("contradiction_hits"):
            a(f"**⚠ 모순 표현 감지** `{r['contradiction_hits']}`")
            a("")

    a("---")
    a("")
    a("### 채점 방식")
    a("")
    a("- **A 답변 정확도** — 정답지가 요구한 항목이 답변에 있는지 (가중 평균)")
    a("- **B 검색 커버리지** — 그 항목이 검색된 근거 안에 있었는지")
    a("- **C 출처** — 정답 문서를 인용했는지(2/3) + 문서 안 위치까지 밝혔는지(1/3)")
    a("- **E 근거 정합성** — 답변에 쓴 사실이 근거에도 있는지 (지어내지 않았는지)")
    a("")
    a("B·C는 검색 단계에서 결정되고 온도가 없어 **실행마다 같은 값**이 나온다. "
      "실행 간 점수 차이는 사실상 전부 생성(A·E) 쪽 변동이다.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--compare", help="이전 실행 raw")
    ap.add_argument("--out", help="저장 경로 (기본: 공유_{평가셋}.md)")
    ap.add_argument("--hide-gold", action="store_true",
                    help="정답·채점 키워드를 가린다 (홀드아웃 보호)")
    ap.add_argument("--all", action="store_true",
                    help="맞힌 문항까지 전 문항의 질문·정답·답변을 싣는다")
    a = ap.parse_args()

    rows = S.run(Path(a.raw), Path(a.gold))
    cmp_rows = S.run(Path(a.compare), Path(a.gold)) if a.compare else None
    md = build(rows, a.gold, a.raw, a.hide_gold, cmp_rows, a.all)

    out = Path(a.out or f"공유_{Path(a.gold).stem}.md")
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n{'=' * 60}\n저장: {out}  ({len(md):,}자)")
    if not a.hide_gold:
        print("⚠ 이 문서에는 정답과 채점 키워드가 들어 있습니다.\n"
              "  팀원과 공유하면 이 평가셋은 홀드아웃 기능을 잃습니다.\n"
              "  점수만 공유하려면 --hide-gold 를 붙이세요.")


if __name__ == "__main__":
    main()
