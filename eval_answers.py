#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연금 Agent 공용 평가 도구 — 팀 누구나 자기 시스템으로 돌릴 수 있습니다.

왜 이게 따로 있나
-----------------
윤종의 `run_eval.py`는 Chroma + BM25 인덱스에 직접 붙어 있어서 다른 사람은 못 씁니다.
이 파일은 **파이썬 표준 라이브러리만** 씁니다. 설치할 것이 없고,
여러분 시스템이 벡터든 SQL이든 HCX 직접 호출이든 상관없이 같은 기준으로 채점합니다.

같은 평가셋으로 각자 돌려서 나온 숫자여야 서로 비교가 됩니다.

세 가지 사용법
-------------

【1】 결과 파일로 채점 — 가장 간단, 여기서 시작하세요

    여러분 시스템을 어떻게든 돌려서 아래 형식의 JSON을 만든 뒤 채점합니다.

        python eval_answers.py --results my_results.json

    my_results.json 형식 (평가 API 응답 규격과 동일):

        [
          {"question_id": "Q-001",
           "answer": "연금저축과 IRP를 합쳐 연 900만원까지…",
           "retrieved_context": "[doc6 1쪽] …근거 원문…"},
          {"question_id": "Q-002", "answer": "…"}
        ]

    `retrieved_context`는 없어도 됩니다(있으면 근거 제공률도 같이 잽니다).
    `questions_only.json`을 참고해 질문 목록을 가져다 쓰세요.

【2】 파이썬 함수 연결 — 시스템이 이미 파이썬이면 이게 편합니다

    아무 .py 파일에 함수 하나만 만들어 두세요.

        # my_agent.py
        def answer(question: str) -> dict:
            ...
            return {"answer": "...", "retrieved_context": "..."}

    그리고 실행합니다.

        python eval_answers.py --adapter my_agent:answer

    함수가 문자열만 반환해도 됩니다. 그럼 그게 answer로 취급됩니다.

【3】 배포한 API 엔드포인트 직접 호출 — 제출 직전 점검용

        python eval_answers.py --endpoint http://123.45.67.89/answer

    주최측이 실제로 보낼 요청과 똑같은 형태(GET, 헤더 없음, question_id·question
    쿼리 파라미터)로 호출합니다. 그래서 정확도뿐 아니라 아래도 같이 점검합니다.

      · 응답 5개 필드가 다 있는지, 전부 문자열인지
      · 300초 타임아웃 안에 들어오는지 (실측 시간 출력)
      · Content-Type이 application/json인지

    제출 전에 반드시 한 번 돌려보세요. 규격이 틀리면 정확도와 무관하게 0점입니다.

무엇을 재는가
------------
  정답 포함률 (주 지표) : answer 안에 정답 핵심어(answer_points)가 다 들어있나
  근거 제공률           : retrieved_context 안에 그 근거가 들어있나
                          → 답은 맞는데 근거가 없으면 LLM이 지어낸 것일 수 있습니다
  경고                  : 오답으로 알려진 표현이 답변에 있으면 사람이 보라고 표시

주의: 자동 채점은 1차 필터입니다. 문자열 매칭이라 표현이 다르면 놓칠 수 있습니다.
      최종 판단은 사람이 하세요. `--show` 로 답변 전문을 볼 수 있습니다.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

TIMEOUT_LIMIT = 300  # 주최측 규정


def load_env(path: str = ".env") -> int:
    """.env를 표준 라이브러리만으로 읽는다.

    python-dotenv를 쓰면 설치 의존성이 생긴다. 이 도구는 팀원 누구나
    설치 없이 돌릴 수 있어야 하므로 직접 파싱한다.
    이미 설정된 환경변수는 덮어쓰지 않는다.
    """
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def norm(s: str) -> str:
    """표기 흔들림 흡수. '900만원' / '900만 원' / '900,000' 을 같게 본다."""
    return re.sub(r"[\s,]", "", s or "")


def point_found(point: str, blob: str) -> bool:
    """정답 핵심어가 답변에 있는지 판정한다.

    '|'로 대안 표현을 나열할 수 있다. 하나라도 맞으면 통과.
        "근거법령이 다른|법령 차이|법령이 다르"

    같은 뜻을 다르게 쓴 것을 오답으로 잡던 문제가 실제로 있었다.
    답변은 "언제든지 변경 가능"인데 정답 문자열이 "언제든 변경"이라
    조사 하나 때문에 틀린 것으로 처리됐다.
    """
    return any(norm(alt) in blob for alt in point.split("|") if alt.strip())


# ──────────────────────────────────────────────────────────────────────
# 답변 수집 — 세 가지 경로
# ──────────────────────────────────────────────────────────────────────

def collect_from_file(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):                 # {"results": [...]} 형태도 허용
        raw = raw.get("results") or raw.get("questions") or []
    out = {}
    for r in raw:
        if not isinstance(r, dict) or "question_id" not in r:
            continue
        out[r["question_id"]] = r
    return out


def collect_from_adapter(spec: str, questions: list[dict]) -> dict[str, dict]:
    """spec = 'module:function'"""
    if ":" not in spec:
        sys.exit("--adapter 형식은 'module:function' 입니다. 예: my_agent:answer")
    mod_name, fn_name = spec.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    fn = getattr(importlib.import_module(mod_name), fn_name)

    out = {}
    for i, q in enumerate(questions, 1):
        print(f"  [{i:2d}/{len(questions)}] {q['question_id']} 호출 중…", end="\r")
        t0 = time.monotonic()
        try:
            r = fn(q["question"])
        except Exception as e:                              # noqa: BLE001
            r = {"answer": "", "error": f"{type(e).__name__}: {e}"}
        if isinstance(r, str):
            r = {"answer": r}
        r["elapsed"] = time.monotonic() - t0
        r["question_id"] = q["question_id"]
        out[q["question_id"]] = r
    print(" " * 70)
    return out


def collect_from_endpoint(url: str, questions: list[dict]) -> dict[str, dict]:
    """주최측이 보낼 요청과 동일하게 호출한다. 헤더를 붙이지 않는 것이 핵심."""
    out = {}
    for i, q in enumerate(questions, 1):
        params = urllib.parse.urlencode(
            {"question_id": q["question_id"], "question": q["question"]})
        full = f"{url}{'&' if '?' in url else '?'}{params}"
        print(f"  [{i:2d}/{len(questions)}] {q['question_id']} 호출 중…", end="\r")
        t0 = time.monotonic()
        rec: dict = {"question_id": q["question_id"]}
        try:
            with urllib.request.urlopen(full, timeout=TIMEOUT_LIMIT) as resp:
                body = resp.read().decode("utf-8")
                rec["_status"] = resp.status
                rec["_ctype"] = resp.headers.get("Content-Type", "")
            try:
                rec.update(json.loads(body))
            except json.JSONDecodeError:
                rec["error"] = f"JSON 파싱 실패: {body[:120]}"
        except Exception as e:                              # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed"] = time.monotonic() - t0
        out[q["question_id"]] = rec
    print(" " * 70)
    return out


# ──────────────────────────────────────────────────────────────────────
# 규격 점검 (엔드포인트 모드에서만 의미가 있다)
# ──────────────────────────────────────────────────────────────────────

REQUIRED = ["question_id", "question", "retrieved_context", "think_trace", "answer"]


def check_schema(results: dict[str, dict]) -> list[str]:
    problems = []
    slow = [(qid, r["elapsed"]) for qid, r in results.items()
            if r.get("elapsed", 0) > TIMEOUT_LIMIT * 0.8]
    for qid, r in sorted(results.items()):
        if r.get("error"):
            problems.append(f"{qid}: 호출 실패 — {r['error']}")
            continue
        if r.get("_status") not in (None, 200):
            problems.append(f"{qid}: HTTP {r['_status']} (200이어야 함)")
        ct = r.get("_ctype")
        if ct is not None and "application/json" not in ct:
            problems.append(f"{qid}: Content-Type이 '{ct}' — application/json이어야 함")
        missing = [f for f in REQUIRED if f not in r]
        if missing:
            problems.append(f"{qid}: 응답 필드 누락 {missing}")
        nonstr = [f for f in REQUIRED if f in r and not isinstance(r[f], str)]
        if nonstr:
            problems.append(f"{qid}: 문자열이 아닌 필드 {nonstr} — 전부 string이어야 함")
        if r.get("question_id") and r["question_id"] != qid:
            problems.append(f"{qid}: 응답의 question_id가 '{r['question_id']}'로 다름")
    for qid, sec in slow:
        problems.append(f"{qid}: {sec:.0f}초 소요 — 300초 한도에 위험하게 근접")
    return problems


# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="연금 Agent 공용 평가 도구 (표준 라이브러리만 사용)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--results", help="결과 JSON 파일 경로")
    src.add_argument("--adapter", help="'module:function' 형식의 파이썬 함수")
    src.add_argument("--endpoint", help="배포한 /answer 엔드포인트 URL")
    ap.add_argument("--evalset", default="./evalset_v1.json")
    ap.add_argument("--type", help="특정 유형만 (예: 세제)")
    ap.add_argument("--show", action="store_true", help="답변 전문 출력")
    ap.add_argument("--out", help="채점 결과 저장 경로")
    ap.add_argument("--name", default="", help="비교표에 쓸 시스템 이름")
    a = ap.parse_args()

    n_env = load_env()
    if n_env:
        print(f"(.env에서 {n_env}개 설정을 읽었습니다)")

    ep = Path(a.evalset)
    if not ep.exists():
        sys.exit(f"평가셋을 찾을 수 없습니다: {ep}\n"
                 f"evalset_v1.json을 이 폴더에 두거나 --evalset 로 경로를 주세요.")
    questions = json.loads(ep.read_text(encoding="utf-8"))["questions"]
    if a.type:
        questions = [q for q in questions if q["query_type"] == a.type]
    if not questions:
        sys.exit("해당하는 문항이 없습니다.")

    label = a.name or (a.endpoint or a.adapter or a.results)
    print(f"평가셋 {ep.name} · {len(questions)}문항 · 대상: {label}\n")

    if a.results:
        results = collect_from_file(Path(a.results))
    elif a.adapter:
        results = collect_from_adapter(a.adapter, questions)
    else:
        results = collect_from_endpoint(a.endpoint, questions)

    # ── 채점 ──────────────────────────────────────────────────────────
    rows = []
    hallu = []          # coverage=none 문항의 환각 점검 결과
    by_type = defaultdict(lambda: {"n": 0, "ans": 0, "ctx": 0})
    by_diff = defaultdict(lambda: {"n": 0, "ans": 0})
    n_missing = 0

    # 근거가 코퍼스에 없는 문항은 "모른다고 답하는가"가 정답이다.
    # answer_points가 비어 있으면 all([])==True라 자동 통과돼 버리므로 따로 다룬다.
    UNSURE = ["확인되지 않", "확인할 수 없", "찾지 못", "정보가 없", "자료에 없",
              "포함되어 있지 않", "알 수 없", "제공되지 않", "언급되어 있지 않",
              "나와 있지 않", "확인이 어렵"]

    for q in questions:
        r = results.get(q["question_id"])
        if r is None:
            n_missing += 1
            continue
        ans = norm(r.get("answer", ""))
        ctx = norm(r.get("retrieved_context", ""))

        if q.get("coverage") == "none":
            said_unsure = any(norm(u) in ans for u in UNSURE)
            hallu.append({"question_id": q["question_id"],
                          "question": q["question"],
                          "ok": said_unsure,
                          "answer": r.get("answer", "")})
            mark = "✅" if said_unsure else "🔴"
            print(f"  {mark} [환각점검] {q['question_id']}  {q['question'][:38]}")
            if not said_unsure:
                print(f"        근거가 없는데 단정적으로 답했습니다: "
                      f"{(r.get('answer') or '')[:90]}")
            continue

        miss_a = [p for p in q["answer_points"] if not point_found(p, ans)]
        miss_c = [p for p in q["answer_points"] if not point_found(p, ctx)]

        # 오답 마커는 답변 **앞부분**에서만 본다.
        # 프롬프트가 "핵심 답변을 먼저"라고 지시하므로 결론은 앞에 온다.
        # 뒤쪽 예외 설명까지 훑으면 오탐이 난다. 실제로 "원칙적으로 이전 불가능,
        # 단 퇴직충당금은 이전이 가능합니다"라는 정확한 답변이
        # '이전이 가능합니다' 때문에 오답으로 표시됐다.
        # 오답 마커는 답변 앞부분에서만 본다(프롬프트가 결론을 먼저 쓰게 지시).
        # 게다가 정답 핵심어를 **전부** 갖춘 답변이면 마커가 걸려도 무시한다.
        # "원칙적으로 이전 불가능. 단, 퇴직충당금은 이전이 가능합니다" 같은
        # 정확한 답변이 예외 절 때문에 오답으로 표시된 적이 있다.
        head = norm((r.get("answer", "") or "")[:250])
        warns = ([] if (not miss_a and q["answer_points"])
                 else [w for w in q.get("wrong_markers", []) if norm(w) in head])

        ok_a = not miss_a
        ok_c = bool(ctx) and not miss_c

        rows.append({
            "question_id": q["question_id"], "question": q["question"],
            "query_type": q["query_type"], "difficulty": q["difficulty"],
            "answer_ok": ok_a, "context_ok": ok_c,
            "missing_in_answer": miss_a, "warnings": warns,
            "elapsed": round(r.get("elapsed", 0), 1),
            "answer": r.get("answer", ""),
        })
        t = by_type[q["query_type"]]
        t["n"] += 1; t["ans"] += ok_a; t["ctx"] += ok_c
        d = by_diff[q["difficulty"]]
        d["n"] += 1; d["ans"] += ok_a

        mark = "✅" if ok_a else "❌"
        if warns:
            mark = "⚠️ "
        el = f" {r['elapsed']:5.1f}s" if r.get("elapsed") else ""
        line = f"  {mark}{el} {q['question_id']}  {q['question'][:40]}"
        if miss_a:
            line += f"\n        누락: {miss_a}"
        if warns:
            line += f"\n        ⚠️ 오답 의심 표현: {warns}  → 함정: {q['trap'][:60]}"
        print(line)
        if a.show:
            print(f"        답변: {(r.get('answer') or '(없음)')[:300]}\n")

    if not rows:
        sys.exit("\n채점할 결과가 없습니다. question_id가 평가셋과 맞는지 확인하세요.")

    # ── 실패를 점수로 위장하지 않는다 ──────────────────────────────
    # 시스템이 통째로 죽었는데 "정답률 15%"라고 출력하면 성능 문제로 오해한다.
    # 실제로 그 사고가 있었다(.env 미로드 → 전 문항 실패 → 15.4%로 표시).
    errs = [(qid, r.get("error")) for qid, r in results.items() if r.get("error")]
    empty = [qid for qid, r in results.items()
             if not r.get("error") and not (r.get("answer") or "").strip()]
    broken = len(errs) + len(empty)
    if broken >= max(3, len(results) * 0.3):
        print("\n" + "!" * 66)
        print(f"채점을 중단합니다 — {len(results)}개 중 {broken}개가 실패했습니다.")
        print("!" * 66)
        if errs:
            print(f"\n  오류 {len(errs)}건. 대표 메시지:")
            seen = set()
            for qid, e in errs:
                key = str(e)[:80]
                if key in seen:
                    continue
                seen.add(key)
                print(f"    [{qid}] {e}")
                if len(seen) >= 3:
                    break
        if empty:
            print(f"\n  답변이 빈 문항 {len(empty)}건: {empty[:8]}")
        print("\n  자주 있는 원인")
        print("    · .env를 못 읽음 → 실행 위치가 .env가 있는 폴더인지 확인")
        print("    · API 키·REQUEST_ID 누락 → 각 시스템의 selftest로 먼저 확인")
        print("    · 인덱스/DB 경로가 다름")
        print("\n  이 상태의 점수는 성능이 아니라 고장입니다. 고친 뒤 다시 돌리세요.")
        sys.exit(1)
    if broken:
        print(f"\n  ⚠️ {broken}개 문항이 실패했습니다(오류 {len(errs)} / 빈 답변 "
              f"{len(empty)}). 점수가 낮게 나올 수 있으니 원인을 확인하세요.")

    # ── 규격 점검 ────────────────────────────────────────────────────
    if a.endpoint:
        print("\n" + "=" * 66)
        print("평가 API 규격 점검")
        print("=" * 66)
        problems = check_schema(results)
        if problems:
            print("  ❌ 아래를 고치지 않으면 정확도와 무관하게 점수를 못 받습니다.\n")
            for p in problems[:20]:
                print(f"     · {p}")
            if len(problems) > 20:
                print(f"     … 외 {len(problems)-20}건")
        else:
            print("  ✅ 5개 필드 전부 문자열, HTTP 200, application/json, 시간 여유 OK")

    # ── 요약 ─────────────────────────────────────────────────────────
    n = len(rows)
    ans_ok = sum(r["answer_ok"] for r in rows)
    ctx_ok = sum(r["context_ok"] for r in rows)
    warned = [r for r in rows if r["warnings"]]

    print("\n" + "=" * 66)
    print(f"채점 결과 — {label}")
    print("=" * 66)
    print(f"  ▶ 정답 포함률   {ans_ok}/{n} = {ans_ok/n:.1%}   ← 주 지표")
    print(f"    근거 제공률   {ctx_ok}/{n} = {ctx_ok/n:.1%}   "
          f"← 답의 근거를 retrieved_context에 실었나")
    if ctx_ok < ans_ok:
        print(f"      ⚠️ 정답률보다 근거율이 {ans_ok-ctx_ok}건 낮습니다. "
              f"근거 없이 답한 것이 있다는 뜻이라 확인이 필요합니다.")
    if n_missing:
        print(f"    ⚠️ 결과가 없는 문항 {n_missing}개는 채점에서 빠졌습니다.")

    print("\n  유형별")
    for t, v in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
        bar = "█" * round(v["ans"] / v["n"] * 18)
        print(f"    {t:<12} {v['ans']:>2}/{v['n']:<2} {v['ans']/v['n']:>6.0%} {bar}")

    print("\n  난이도별")
    for d in ["하", "중", "상"]:
        if d in by_diff:
            v = by_diff[d]
            print(f"    {d}  {v['ans']:>2}/{v['n']:<2}  {v['ans']/v['n']:.0%}")

    if hallu:
        ok = sum(1 for h in hallu if h["ok"])
        print(f"\n  환각 점검 (근거가 코퍼스에 없는 {len(hallu)}문항)")
        print(f"    모른다고 답함  {ok}/{len(hallu)}"
              f"   ← 정답률보다 중요할 수 있습니다")
        bad = [h for h in hallu if not h["ok"]]
        if bad:
            print(f"    🔴 근거 없이 단정한 문항 {len(bad)}개 — 프롬프트 보강 필요")
            for h in bad:
                print(f"       {h['question_id']}  {h['question'][:44]}")

    times = [r["elapsed"] for r in rows if r["elapsed"]]
    if times:
        print(f"\n  응답시간  평균 {sum(times)/len(times):.1f}초 / "
              f"최대 {max(times):.1f}초  (한도 {TIMEOUT_LIMIT}초)")

    if warned:
        print(f"\n  ⚠️ 오답 의심 {len(warned)}건 — 사람이 직접 확인하세요")
        for r in warned:
            print(f"     {r['question_id']}  {r['warnings']}")

    fails = [r for r in rows if not r["answer_ok"]]
    if fails:
        print(f"\n  ❌ 틀린 문항 {len(fails)}개")
        for r in fails:
            print(f"     {r['question_id']} {r['question'][:42]}")
            print(f"        누락: {r['missing_in_answer']}")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "name": label,
            "answer_rate": ans_ok / n, "context_rate": ctx_ok / n,
            "n": n, "by_type": {k: dict(v) for k, v in by_type.items()},
            "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  결과 저장: {a.out}")

    print("\n  ※ 문자열 매칭 기반 1차 채점입니다. 표현이 달라 놓치는 경우가 있으니")
    print("    ❌로 찍힌 것은 --show 로 답변을 직접 확인해 보세요.")


if __name__ == "__main__":
    main()
