import json, sys, hashlib
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "evalset_v5_gold_quality_36.json")
data = json.loads(p.read_text(encoding="utf-8"))
qs = data["questions"]
assert data["n_questions"] == len(qs), (data["n_questions"], len(qs))
ids = [q["question_id"] for q in qs]
assert len(ids) == len(set(ids)), "duplicate question_id"
required = {"question_id","question","gold_answer","answer_points","source_doc","trap","failure_target"}
for i,q in enumerate(qs,1):
    miss = required - q.keys()
    assert not miss, f"{i} missing {miss}"
    assert q["question"].strip() and q["gold_answer"].strip()
    assert isinstance(q["answer_points"], list) and q["answer_points"], q["question_id"]

blob = json.dumps(
    [{"question_id":q["question_id"],"question":q["question"]} for q in qs],
    ensure_ascii=False, sort_keys=True, separators=(",",":")
).encode("utf-8")
print("OK")
print("questions:", len(qs))
print("question_set_sha256:", hashlib.sha256(blob).hexdigest())
