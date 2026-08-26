"""v3에서 남은 문제 5개만 확인한다 (전체 재실행 전 사전 점검)."""
import re
import sys

sys.path.insert(0, ".")
from agent import run

CASES = [
    ("IB1-79E75141",
     "자영업자가 IRP 가입이 가능한지 먼저 알려주세요. 그리고 "
     "미래에셋프리미엄크레딧알파의 C-P가 퇴직연금용 클래스인지도 확인해 주세요.",
     ["연금저축", "오프라인"], "v3 48% — C-P를 '퇴직연금용'이라 답함(오답)"),
    ("IB1-00A43216",
     "미래에셋고배당포커스연금저축증권전환형자투자신탁1호는 어떤 상품유형이고 "
     "투자위험등급은 어떻게 분류되어 있나요?",
     ["주식형", "3등급"], "v3 67% — '주식형' 표현 누락"),
    ("IB1-BF6DB35D",
     "하나파워e단기채증권자투자신탁은 투자설명서에서 어떤 상품유형과 "
     "위험등급으로 분류되어 있나요?",
     ["채권형", "6등급"], "v3 67% — '채권형' 표현 누락"),
    ("IB1-99BB1EF3",
     "삼성클래식연금증권전환형투자신탁 제1호는 투자설명서상 어떤 상품유형이고 "
     "위험등급은 몇 등급인가요?",
     ["채권형", "5등급"], "v3 67% — v2에선 100%였다가 표현 때문에 하락"),
    ("IB1-88B9A8BC",
     "하나IT코리아증권자투자신탁 제1호의 상품유형과 위험등급은 어떻게 "
     "표시되어 있나요?",
     ["주식형", "2등급"], "v3 67% — '주식형' 표현 누락"),
]

PAGE_RE = re.compile(r"\d+\s*쪽|p\.\s*\d+|슬라이드\s*\d+|§")

for qid, q, want, before in CASES:
    r = run(q, qid, use_cache=False)
    ans = r.get("answer") or ""
    hit = [w for w in want if w.replace(" ", "") in ans.replace(" ", "")]
    print("=" * 78)
    print(f"{qid}   이전: {before}")
    print(f"  기대 {want} → 포함 {hit}  ({len(hit)}/{len(want)})"
          f"   출처표기: {'✅' if PAGE_RE.search(ans) else '❌'}")
    for line in (r.get("think_trace") or "").split("\n"):
        if "fee_sql" in line or "need_sql" in line or "복합" in line:
            print("  " + line[:130])
    print("-" * 78)
    print("  " + ans.replace("\n", "\n  ")[:520])
    print()
