#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""killing camp H-08 한 문항만 돌려 **SQL 원문을 포함한 전체 트레이스**를 찍는다.

spotcheck6은 트레이스를 한 줄로 잘라 보여줘서 SQL이 안 보인다. 0행의 원인을
추측하지 않고 실제 SQL을 보고 판단하기 위한 스크립트다. HCX 호출은 2~3회.

    python diag_h08.py
    python diag_h08.py H-11        (다른 문항을 보고 싶으면)
"""
import json
import sqlite3
import sys
from pathlib import Path

import agent

qid = sys.argv[1] if len(sys.argv) > 1 else "H-08"
gold = {q["question_id"]: q for q in json.loads(
    Path("killing_camp_v1.json").read_text(encoding="utf-8"))["questions"]}
q = gold[qid]

print(f"[{qid}] {q['question']}\n")
r = agent.run(question=q["question"], question_id=f"diag-{qid}", use_cache=False)

print("=" * 74)
print("트레이스 (SQL 포함)")
print("=" * 74)
print(r.get("think_trace") or "(비어 있음)")

print("\n" + "=" * 74)
print("답변")
print("=" * 74)
print(r.get("answer"))

print("\n" + "=" * 74)
print("정답지가 요구하는 것")
print("=" * 74)
for req in q.get("required", []):
    print(f"  w={req.get('weight', 1)}  {req.get('alts') or req.get('values')}")

print("\n" + "=" * 74)
print("이 펀드의 DB 실제 행")
print("=" * 74)
conn = sqlite3.connect(agent.FUND_FEES_DB)
conn.row_factory = sqlite3.Row
for row in conn.execute(
        "SELECT class_code, channel, fee_total FROM fund_fees "
        "WHERE REPLACE(fund_name,' ','') LIKE '%NH-Amundi하나로단기채%'"):
    print("  class=%-8s channel=%-10s fee=%s"
          % (row["class_code"], row["channel"], row["fee_total"]))
