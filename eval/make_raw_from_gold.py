#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정답지(gold)에서 질문만 뽑아 에이전트를 돌리고 RAW 결과를 만든다.

정답은 절대 에이전트에 넘기지 않는다. gold 파일에서 question_id와 question만
읽고, 나머지(gold_answer·required·sources)는 채점기(score_official.py)만 본다.

    python make_raw_from_gold.py --gold gold_holdout_v1.json \
        --out raw_penguinnote_holdout_v1.json

중간에 끊겨도 --resume 을 붙이면 이미 답한 문항은 건너뛴다.

    python make_raw_from_gold.py --gold gold_holdout_v1.json \
        --out raw_penguinnote_holdout_v1.json --resume
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def git_meta() -> dict:
    def cmd(a):
        try:
            return subprocess.check_output(
                a, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""
    sha = cmd(["git", "rev-parse", "HEAD"])
    status = cmd(["git", "status", "--porcelain"])
    return {"git_commit": sha, "git_dirty": bool(status),
            "git_status": status[:5000]}


def load_adapter(spec: str):
    mod, fn = spec.split(":", 1)
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(mod), fn)


def normalize(r) -> dict:
    if isinstance(r, str):
        return {"answer": r}
    if isinstance(r, dict):
        return r
    return {"answer": str(r)}


def main():
    ap = argparse.ArgumentParser(
        description="gold에서 질문만 뽑아 RAW 실행 결과를 만든다 (채점 안 함)")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="agent:answer_for_eval")
    ap.add_argument("--name", default="penguinnote")
    ap.add_argument("--limit", type=int, default=0,
                    help="앞에서 N문항만 (0이면 전부)")
    ap.add_argument("--only", default="",
                    help="쉼표로 구분한 question_id만 실행")
    ap.add_argument("--resume", action="store_true",
                    help="--out에 이미 있는 문항은 건너뛴다")
    a = ap.parse_args()

    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    qs = [{"question_id": q["question_id"], "question": q["question"]}
          for q in gold["questions"]]

    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        qs = [q for q in qs if q["question_id"] in want]
    if a.limit:
        qs = qs[:a.limit]

    out_path = Path(a.out)
    done: dict[str, dict] = {}
    if a.resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        for r in prev.get("results", []):
            if (r.get("answer") or "").strip() and not r.get("error"):
                done[r["question_id"]] = r
        if done:
            print(f"이어서 실행: 이미 끝난 {len(done)}문항은 건너뜁니다\n")

    answer = load_adapter(a.adapter)

    # 토큰 집계가 있으면 이번 실행분만 세도록 초기화한다
    try:
        importlib.import_module(a.adapter.split(":", 1)[0]).reset_token_usage()
    except Exception:
        pass

    results, t_start = [], time.monotonic()
    for i, q in enumerate(qs, 1):
        qid = q["question_id"]
        if qid in done:
            results.append(done[qid])
            print(f"[{i:02d}/{len(qs)}] {qid}  (건너뜀)", flush=True)
            continue

        print(f"[{i:02d}/{len(qs)}] {qid}  {q['question'][:44]}", flush=True)
        t = time.monotonic()
        try:
            r = normalize(answer(q["question"]))
            err = str(r.get("error") or "")
        except Exception as e:
            r, err = {}, f"{type(e).__name__}: {e}"

        sec = round(time.monotonic() - t, 3)
        row = {
            "question_id": qid,
            "question": q["question"],
            "answer": str(r.get("answer") or ""),
            "retrieved_context": str(r.get("retrieved_context") or ""),
            "think_trace": str(r.get("think_trace") or r.get("trace") or ""),
            "sources": r.get("sources") if "sources" in r else "",
            "elapsed_sec": sec,
            "error": err,
        }
        results.append(row)

        mark = "✗ " + err[:60] if err else f"{len(row['answer'])}자"
        print(f"        {sec:6.1f}초  {mark}", flush=True)

        # 매 문항마다 저장한다. 중간에 끊겨도 지금까지가 남는다.
        out_path.write_text(json.dumps({
            "benchmark": gold.get("note", "")[:80] or "holdout",
            "version": gold.get("version", ""),
            "system_name": a.name,
            "question_set_sha256": gold.get("question_set_sha256"),
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            **git_meta(),
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    total = time.monotonic() - t_start
    ok = sum(1 for r in results if (r["answer"] or "").strip() and not r["error"])
    print(f"\n{'=' * 60}")
    print(f"RAW 저장: {out_path}")
    print(f"문항 {len(results)}개 | 응답 성공 {ok}개 | 총 {total / 60:.1f}분")
    try:
        u = importlib.import_module(a.adapter.split(":", 1)[0]).token_usage()
        print(f"HCX 호출 {u['calls']}회 | 토큰 입력 {u['prompt']:,} "
              f"+ 출력 {u['completion']:,} = {u['total']:,}")
    except Exception:
        pass
    print("\n채점:")
    print(f"  python score_official.py --raw {out_path.name} "
          f"--gold {Path(a.gold).name} --show 15")


if __name__ == "__main__":
    main()
