#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PenguinNote/eval_answers.py 결과 JSON + gold evalset을 병합해
질문 / 모델답변 / 정답 / answer_points / 1차채점 / 실패타깃을 한 파일로 만든다.

Usage:
  python merge_eval_result_with_gold_v4.py \
    --evalset evalset_v4_stress_36.json \
    --result result_penguinnote_v4_stress36.json \
    --out result_with_gold_v4_stress36.json \
    --csv result_with_gold_v4_stress36.csv
"""
import argparse, csv, json, re
from pathlib import Path

def norm(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "")
    return s

def point_hit(answer, point):
    # answer_points use "|" as accepted alternatives
    a = norm(answer)
    alts = [norm(z) for z in str(point).split("|") if z.strip()]
    return any(z in a for z in alts)

def extract_items(result):
    # Support common result shapes.
    if isinstance(result, list):
        return result
    for k in ("rows", "results", "items", "questions", "records", "outputs"):
        if isinstance(result.get(k), list):
            return result[k]
    raise ValueError("결과 JSON에서 문항 리스트를 찾지 못했습니다. keys=" + ",".join(result.keys()))

def get_qid(x):
    return x.get("question_id") or x.get("id") or x.get("qid")

def get_answer(x):
    for k in ("answer", "model_answer", "response", "output", "prediction", "text"):
        if x.get(k) is not None:
            return x.get(k)
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    gold = json.loads(Path(args.evalset).read_text(encoding="utf-8"))
    res = json.loads(Path(args.result).read_text(encoding="utf-8"))

    gold_by_id = {x["question_id"]: x for x in gold["questions"]}
    res_items = extract_items(res)
    res_by_id = {get_qid(x): x for x in res_items if get_qid(x)}

    merged = []
    for qid, g in gold_by_id.items():
        r = res_by_id.get(qid, {})
        ans = get_answer(r)
        pts = g.get("answer_points", [])
        hits = [point_hit(ans, p) for p in pts]
        auto_score = (sum(hits) / len(hits)) if hits else None
        merged.append({
            "question_id": qid,
            "domain": g.get("domain"),
            "query_type": g.get("query_type"),
            "difficulty": g.get("difficulty"),
            "question": g.get("question"),
            "model_answer": ans,
            "gold_answer": g.get("gold_answer"),
            "answer_points": pts,
            "answer_point_hits": hits,
            "answer_point_score": auto_score,
            "all_answer_points_hit": all(hits) if hits else None,
            "trap": g.get("trap"),
            "failure_target": g.get("failure_target"),
            "source_doc": g.get("source_doc"),
            "raw_result": r,
        })

    scores = [x["answer_point_score"] for x in merged if x["answer_point_score"] is not None]
    summary = {
        "n_questions": len(merged),
        "n_result_matches": sum(1 for x in merged if x["raw_result"]),
        "all_points_hit_count": sum(1 for x in merged if x["all_answer_points_hit"] is True),
        "all_points_hit_rate": (
            sum(1 for x in merged if x["all_answer_points_hit"] is True) / len(merged)
            if merged else None
        ),
        "mean_answer_point_score": (sum(scores) / len(scores)) if scores else None,
        "warning": "문자열 기반 1차 채점입니다. 의미상 정답/동의어는 사람 검토가 필요합니다."
    }

    out = {
        "benchmark_version": gold.get("version"),
        "summary": summary,
        "items": merged,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.csv:
        with Path(args.csv).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "question_id","domain","query_type","question","model_answer","gold_answer",
                "answer_point_score","all_answer_points_hit","failure_target","trap","source_doc"
            ])
            for x in merged:
                w.writerow([
                    x["question_id"],x["domain"],x["query_type"],x["question"],
                    x["model_answer"],x["gold_answer"],x["answer_point_score"],
                    x["all_answer_points_hit"],x["failure_target"],x["trap"],x["source_doc"]
                ])

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
