#!/usr/bin/env python3
"""Mirae Integrated Blind v1 — 공식 정답지 기반 채점기.

FAIRNESS_AND_SCORING.md 에 정의된 규칙을 그대로 구현한다.

  자동 상태 (WRONG는 자동으로 확정하지 않는다 — 정책 명시 사항):
    AUTO_CORRECT          등록된 핵심 사실을 모두 확인
    PARTIAL_REVIEW        일부만 확인
    SEMANTIC_REVIEW       문자열로는 판정 불가 (0% 확인)
    CONTRADICTION_REVIEW  사전등록된 모순 표현 발견
    INFRA_FAIL            빈 답변·실행오류 — 사실 오답으로 집계하지 않음

  지표 (종합점수 가중치):
    A. Answer Correctness          50%   ← 사람 판정이 최종. 여기서는 자동 근사치.
    B. Retrieval Core Coverage     20%
    C. Citation                    15%   (원문 문서 2/3 + locator 1/3)
    E. Answer-Grounding Alignment  15%

사용법
------
    python score_official.py --raw raw_penguinnote_integrated_v2.json
    python score_official.py --raw new.json --compare old.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GOLD_NAME = "gold_private_integrated_v1.json"

# 정답지를 흔히 두는 위치들을 훑는다. --gold 로 직접 지정할 수도 있다.
_GOLD_CANDIDATES = [
    Path.cwd() / GOLD_NAME,
    Path.cwd() / "mirae_integrated_blind_v1_ANSWERS_ONLY" / GOLD_NAME,
    Path.home() / "Downloads" / "mirae_integrated_blind_v1_ANSWERS_ONLY" / GOLD_NAME,
    Path.home() / "Downloads" / GOLD_NAME,
    Path.home() / "Desktop" / "mirae_integrated_blind_v1_ANSWERS_ONLY" / GOLD_NAME,
    Path("/mnt/user-data/uploads/mirae_integrated_blind_v1_ANSWERS_ONLY") / GOLD_NAME,
]


def find_gold(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise SystemExit(f"정답지를 찾을 수 없습니다: {p}")
        return p
    for p in _GOLD_CANDIDATES:
        if p.exists():
            return p
    # 마지막 수단: 홈 아래를 한 번 훑는다
    for root in (Path.home() / "Downloads", Path.home() / "Desktop", Path.cwd()):
        if root.exists():
            for p in root.rglob(GOLD_NAME):
                return p
    raise SystemExit(
        f"{GOLD_NAME} 을 찾지 못했습니다.\n"
        f"  --gold /경로/{GOLD_NAME} 로 직접 지정해 주세요.")


# ──────────────────────────────────────────────────────────────────
# 문자열 정규화
# ──────────────────────────────────────────────────────────────────

def ns(s: str) -> str:
    """공백 전부 제거 + 소문자화.

    한국어는 띄어쓰기가 자유로워서('중도인출이 불가능' vs '중도인출이불가능')
    공백을 남겨두면 같은 뜻인데 못 찾는다. 정책상 '표현이 다르다'는 오답
    사유가 아니므로 최대한 관대하게 맞춘다.
    """
    return re.sub(r"\s+", "", s or "").lower()


# ──────────────────────────────────────────────────────────────────
# 숫자 매칭
# ──────────────────────────────────────────────────────────────────

def _num_variants(v: float) -> list[str]:
    """하나의 숫자를 사람이 쓸 법한 표기들로 펼친다.

    0.2 → '0.2', '0.20', '.2'
    900 → '900', '900만', '9,000,000'
    """
    out = set()
    out.add(f"{v:g}")
    if v == int(v):
        iv = int(v)
        out.add(str(iv))
        out.add(f"{iv:,}")
    else:
        for nd in (1, 2, 3, 4):
            s = f"{v:.{nd}f}"
            if float(s) == v:
                out.add(s)
                out.add(s + "0")      # 0.2 → 0.20 (정책: 0.20% = 0.2%)
    return [s for s in out if s]


def number_in(v: float, text: str) -> bool:
    """숫자가 텍스트에 있는지. 다른 숫자 안에 박힌 경우는 제외한다.

    이게 없으면 0.33이 '10.335' 안에서도 잡혀 오탐이 난다.
    앞뒤가 숫자거나 소수점이면 매치로 치지 않는다.
    """
    t = ns(text)
    for cand in _num_variants(v):
        c = ns(cand)
        for m in re.finditer(re.escape(c), t):
            before = t[m.start() - 1] if m.start() > 0 else ""
            after = t[m.end()] if m.end() < len(t) else ""
            # 앞이 숫자·소수점이면 더 큰 수의 일부다.
            # 쉼표도 마찬가지다 — '1,900만원'의 '900'을 900으로 세면 안 된다.
            # 단 후보 자체가 쉼표 표기(1,800)일 때는 정상이므로 제외한다.
            if before.isdigit() or before == ".":
                continue
            if before == "," and "," not in c:
                continue
            if after.isdigit():
                continue
            # 뒤가 쉼표+숫자면 '900,000' 같은 더 큰 수의 앞부분이다
            if after == "," and m.end() + 1 < len(t) and t[m.end() + 1].isdigit():
                continue
            if after == "." and m.end() + 1 < len(t) and t[m.end() + 1].isdigit():
                # '0.2' 가 '0.25' 의 앞부분으로 잡히는 것을 막는다
                continue
            return True
    return False


def text_in(alts: list[str], text: str) -> bool:
    t = ns(text)
    return any(ns(a) and ns(a) in t for a in alts)


def req_hit(req: dict, text: str) -> bool:
    if req.get("type") == "number":
        return any(number_in(v, text) for v in req.get("values", []))
    return text_in(req.get("alts", []), text)


# ──────────────────────────────────────────────────────────────────
# 인프라 실패 판정
# ──────────────────────────────────────────────────────────────────

FALLBACK_PREFIX = "답변 생성에 실패"


def is_infra_fail(r: dict) -> tuple[bool, str]:
    ans = (r.get("answer") or "").strip()
    if not ans:
        return True, "빈 답변"
    if r.get("error"):
        return True, f"실행 오류: {str(r['error'])[:60]}"
    st = r.get("_http_status")
    if st not in (None, 200):
        return True, f"HTTP {st}"
    if ans.startswith(FALLBACK_PREFIX):
        return True, "생성 실패 → 근거 요약 대체"
    return False, ""


# ──────────────────────────────────────────────────────────────────
# 채점
# ──────────────────────────────────────────────────────────────────

def score_one(gq: dict, r: dict | None) -> dict:
    qid = gq["question_id"]
    out = {"question_id": qid, "domain": gq.get("domain"),
           "category": gq.get("category"), "difficulty": gq.get("difficulty"),
           "question": gq.get("question"), "gold_answer": gq.get("gold_answer")}

    if r is None:
        out.update(state="MISSING_FROM_RUN", answer_cov=0.0,
                   retrieval_cov=0.0, source_recall=0.0,
                   locator_recall=0.0, grounding=0.0, missing=[])
        return out

    ans = r.get("answer") or ""
    ctx = r.get("retrieved_context") or ""
    srcs = r.get("sources")
    src_text = json.dumps(srcs, ensure_ascii=False) if srcs else ""
    cite_hay = f"{src_text}\n{ctx}\n{ans}"

    out["answer"] = ans
    out["elapsed_sec"] = r.get("elapsed_sec")

    infra, why = is_infra_fail(r)

    # ── A. 답변 정확도 (가중 커버리지) ─────────────────────────
    tot = hit = 0.0
    missing, matched = [], []
    # ── B. 검색 핵심 커버리지 (evidence=true 항목이 context에 있는지) ──
    e_tot = e_hit = 0.0
    # ── E. 근거 정합성 (답변에 쓴 **문서 기반 항목**이 context에도 있는지) ──
    g_tot = g_hit = 0.0

    for req in gq.get("required", []):
        w = float(req.get("weight", 1.0))
        tot += w
        in_ans = req_hit(req, ans)
        in_ctx = req_hit(req, ctx)
        if in_ans:
            hit += w
            matched.append(req)
            # E는 **문서에서 왔어야 할 항목**만 센다.
            #
            # evidence=False 항목은 정답지가 "이건 자료에서 인용할 말이 아니라
            # 에이전트가 스스로 해야 할 말"이라고 표시해 둔 것이다 — 거절,
            # 한계 고지, 안내 창구. 그런 문장이 retrieved_context에 있을 리가
            # 없다. 그런데 이걸 E에 넣으면, **한계를 제대로 고지할수록 E가
            # 떨어진다.** 실측(2026-08-30): 한계 고지를 코드로 고정해 H3-19·20이
            # 40%→100%가 되자 E가 79.9%→76.7%로 내려갔다. 잘한 것이 감점으로
            # 잡힌 것이라 지표 정의가 틀린 것이다.
            #
            # 참고로 팀 자체 blind 평가 팩(score_blind.py — 대회 주최측이 아니라
            # 우리가 만든 것)의 grounding은 정의가 아예 다르다. grounding=true
            # 그룹이 context에 있는지만 보고 답변은 보지 않는다(우리 B에 가깝다).
            # 이 수정은 우리 자체 지표(score_official.py)에만 해당한다.
            # 실제 대회 배점 공식은 공지되지 않았다 — 알고 있는 건 슬라이드의
            # 7개 평가지표 나열뿐이다. 자세한 경위는
            # claude/2026-08-30-정정-배점기준-출처.md 참조.
            if req.get("evidence"):
                g_tot += w
                if in_ctx:
                    g_hit += w
        else:
            missing.append(req)
        if req.get("evidence"):
            e_tot += w
            if in_ctx:
                e_hit += w

    out["answer_cov"] = (hit / tot) if tot else 0.0
    out["retrieval_cov"] = (e_hit / e_tot) if e_tot else None
    out["grounding"] = (g_hit / g_tot) if g_tot else None
    out["missing"] = [{"type": m["type"],
                       "want": m.get("values") or m.get("alts"),
                       "weight": m.get("weight")} for m in missing]
    out["n_required"] = len(gq.get("required", []))
    out["n_matched"] = len(matched)

    # ── C. 출처 / locator ────────────────────────────────────
    gsrc = gq.get("sources", [])
    s_hit = l_hit = 0
    # 정답지 자체에 locator가 없는 출처(71개 중 6개)는 분모에서 뺀다.
    # 밝힐 위치가 없는데 0점을 주면 locator 재현율이 부당하게 낮아진다.
    l_tot = 0
    src_detail = []
    for s in gsrc:
        names = [s.get("file", "")] + list(s.get("aliases", []))
        # 확장자를 뗀 이름도 후보에 넣는다 (doc38.docx → doc38)
        names += [Path(n).stem for n in names if n]
        found = any(ns(n) and ns(n) in ns(cite_hay) for n in names)
        loc = (s.get("locator") or "").strip()
        if loc:
            l_tot += 1
        loc_found = False
        if found and loc:
            mpage = re.search(r"p\.?\s*(\d+)", loc)
            if mpage:
                pg = mpage.group(1)
                loc_found = bool(
                    re.search(rf"(?<!\d){pg}\s*(?:쪽|페이지|p)", cite_hay)
                    or re.search(rf"p\.?\s*{pg}(?!\d)", cite_hay, re.I))
            else:
                key = re.sub(r"[§·\s]+", "", loc)
                loc_found = bool(key) and ns(key)[:6] in ns(cite_hay)
        s_hit += int(found)
        l_hit += int(loc_found)
        src_detail.append({"file": s.get("file"), "found": found,
                           "locator": loc, "locator_found": loc_found})

    out["source_recall"] = (s_hit / len(gsrc)) if gsrc else None
    out["locator_recall"] = (l_hit / l_tot) if l_tot else None
    out["sources_detail"] = src_detail

    # ── 상태 판정 ────────────────────────────────────────────
    contra = [p for p in (gq.get("contradiction_phrases") or [])
              if ns(p) and ns(p) in ns(ans)]
    out["contradiction_hits"] = contra

    if infra:
        out["state"] = "INFRA_FAIL"
        out["infra_reason"] = why
    elif contra:
        out["state"] = "CONTRADICTION_REVIEW"
    elif out["answer_cov"] >= 0.999:
        out["state"] = "AUTO_CORRECT"
    elif out["answer_cov"] > 0:
        out["state"] = "PARTIAL_REVIEW"
    else:
        out["state"] = "SEMANTIC_REVIEW"
    return out


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["state"] != "INFRA_FAIL"]
    a = _avg([r["answer_cov"] for r in rows])          # INFRA는 0점 반영
    b = _avg([r.get("retrieval_cov") for r in rows])
    c_src = _avg([r.get("source_recall") for r in rows])
    c_loc = _avg([r.get("locator_recall") for r in rows])
    e = _avg([r.get("grounding") for r in scored])
    citation = c_src * (2 / 3) + c_loc * (1 / 3)
    composite = a * 0.50 + b * 0.20 + citation * 0.15 + e * 0.15
    from collections import Counter
    return {"n": len(rows), "states": dict(Counter(r["state"] for r in rows)),
            "answer_cov": a, "retrieval_cov": b, "source_recall": c_src,
            "locator_recall": c_loc, "citation": citation, "grounding": e,
            "composite": composite}


def run(raw_path: Path, gold_path: Path) -> list[dict]:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if raw.get("question_set_sha256") != gold.get("question_set_sha256"):
        print(f"⚠ question_set_sha256 불일치 — 다른 질문셋입니다\n"
              f"   raw : {raw.get('question_set_sha256')}\n"
              f"   gold: {gold.get('question_set_sha256')}")
    by_id = {r["question_id"]: r for r in raw.get("results", [])}
    return [score_one(gq, by_id.get(gq["question_id"]))
            for gq in gold["questions"]]


def pct(x):
    return "  —  " if x is None else f"{x*100:5.1f}%"


def print_summary(s: dict, label: str):
    print(f"\n{'='*66}\n{label}\n{'='*66}")
    print(f"  문항 수                     {s['n']}")
    print(f"  A. 답변 정확도 (자동 근사)   {pct(s['answer_cov'])}   가중 50%")
    print(f"  B. 검색 핵심 커버리지        {pct(s['retrieval_cov'])}   가중 20%")
    print(f"  C. 출처(문서/locator)       {pct(s['citation'])}   가중 15%")
    print(f"       · 원문 문서 재현율      {pct(s['source_recall'])}")
    print(f"       · locator 재현율        {pct(s['locator_recall'])}")
    print(f"  E. 근거 정합성               {pct(s['grounding'])}   가중 15%")
    print(f"  ── 종합점수                  {pct(s['composite'])}")
    print(f"\n  상태 분포: {s['states']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--compare", help="이전 실행 결과와 비교")
    ap.add_argument("--out", help="문항별 상세를 JSON으로 저장")
    ap.add_argument("--show", type=int, default=15,
                    help="사람 검토가 필요한 문항을 몇 개 보여줄지")
    ap.add_argument("--gold", help=f"{GOLD_NAME} 경로 (미지정 시 자동 탐색)")
    a = ap.parse_args()

    gold_path = find_gold(a.gold)
    print(f"정답지: {gold_path}")

    rows = run(Path(a.raw), gold_path)
    s_new = summarize(rows)
    print_summary(s_new, f"채점 결과 — {Path(a.raw).name}")

    if a.compare:
        old = run(Path(a.compare), gold_path)
        s_old = summarize(old)
        print_summary(s_old, f"비교 대상 — {Path(a.compare).name}")

        obi = {r["question_id"]: r for r in old}
        print(f"\n{'='*66}\n문항별 변화\n{'='*66}")
        up, down, same = [], [], 0
        for r in rows:
            o = obi.get(r["question_id"])
            if not o:
                continue
            d = r["answer_cov"] - o["answer_cov"]
            if d > 0.01:
                up.append((d, r, o))
            elif d < -0.01:
                down.append((d, r, o))
            else:
                same += 1
        print(f"  좋아짐 {len(up)} / 나빠짐 {len(down)} / 동일 {same}\n")
        for tag, lst, rev in (("▲ 좋아짐", up, True), ("▼ 나빠짐", down, False)):
            if not lst:
                continue
            print(f"  {tag}")
            for d, r, o in sorted(lst, key=lambda x: -abs(x[0])):
                print(f"    {r['question_id']}  "
                      f"{o['answer_cov']*100:5.1f}% → {r['answer_cov']*100:5.1f}%"
                      f"  ({d*100:+.1f}p)  {o['state']} → {r['state']}")
            print()
        print(f"  종합점수 {s_old['composite']*100:.1f}% → "
              f"{s_new['composite']*100:.1f}% "
              f"({(s_new['composite']-s_old['composite'])*100:+.1f}p)")

    need = [r for r in rows if r["state"] not in ("AUTO_CORRECT",)]
    if a.show and need:
        print(f"\n{'='*66}\n사람 검토 필요 — 커버리지 낮은 순 {min(a.show, len(need))}개")
        print("(정책상 자동 채점은 WRONG을 확정하지 않습니다)")
        print("="*66)
        for r in sorted(need, key=lambda x: x["answer_cov"])[:a.show]:
            print(f"\n  [{r['state']}] {r['question_id']}  "
                  f"({r['domain']}/{r['difficulty']})  "
                  f"커버리지 {r['answer_cov']*100:.0f}%")
            print(f"    Q: {r['question'][:78]}")
            print(f"    정답: {(r.get('gold_answer') or '')[:78]}")
            if r.get("contradiction_hits"):
                print(f"    ⚠ 모순 표현: {r['contradiction_hits']}")
            if r.get("infra_reason"):
                print(f"    ⚠ {r['infra_reason']}")
            for m in r["missing"][:4]:
                print(f"    누락(w={m['weight']}): {m['want']}")

    if a.out:
        Path(a.out).write_text(
            json.dumps({"summary": s_new, "rows": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n상세 저장: {a.out}")


if __name__ == "__main__":
    main()
