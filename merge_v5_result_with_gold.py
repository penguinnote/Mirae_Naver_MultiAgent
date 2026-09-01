import argparse, csv, json
from pathlib import Path

def load_results(obj):
    if isinstance(obj, dict) and "results" in obj: return obj["results"]
    if isinstance(obj, list): return obj
    raise ValueError("result JSON must be list or {'results': [...]}")

ap = argparse.ArgumentParser()
ap.add_argument("--evalset", default="evalset_v5_gold_quality_36.json")
ap.add_argument("--result", required=True)
ap.add_argument("--out", default="result_with_gold_v5.json")
ap.add_argument("--csv", dest="csv_out", default="result_with_gold_v5.csv")
args = ap.parse_args()

ev = json.loads(Path(args.evalset).read_text(encoding="utf-8"))
rr = load_results(json.loads(Path(args.result).read_text(encoding="utf-8")))
gold = {q["question_id"]: q for q in ev["questions"]}

rows = []
for r in rr:
    qid = r.get("question_id")
    g = gold.get(qid, {})
    rows.append({
        "question_id": qid,
        "domain": g.get("domain",""),
        "query_type": g.get("query_type",""),
        "question": g.get("question", r.get("question","")),
        "model_answer": r.get("answer", r.get("model_answer","")),
        "gold_answer": g.get("gold_answer",""),
        "gold_explanation": g.get("gold_explanation",""),
        "answer_points": g.get("answer_points",[]),
        "trap": g.get("trap",""),
        "failure_target": g.get("failure_target",""),
        "source_doc": g.get("source_doc",""),
        "elapsed_sec": r.get("elapsed_sec",""),
        "error": r.get("error",""),
    })

Path(args.out).write_text(json.dumps({"n":len(rows),"rows":rows}, ensure_ascii=False, indent=2), encoding="utf-8")
with open(args.csv_out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    for row in rows:
        row = dict(row)
        row["answer_points"] = json.dumps(row["answer_points"], ensure_ascii=False)
        w.writerow(row)
print("wrote", args.out, args.csv_out)
