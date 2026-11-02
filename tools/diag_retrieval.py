"""검색만 진단한다. HCX는 부르지 않는다 (임베딩 API만 사용).

묻는 것: 정답이 든 문서가 후보 안에 있기는 한가? 몇 위인가?

  · 상위 8개 안 → 검색은 정상. 문제는 다른 데 있다.
  · 9~60위     → 후보엔 있는데 순위가 낮다. 리랭커가 답이 될 수 있다.
  · 후보 밖    → 검색어와 문서 표현이 어긋난다. 순위 조정으로는 못 고친다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import TOP_K, get_retriever, route          # noqa: E402
from search import POOL, embed_query                   # noqa: E402

# (문항, 질문, 정답이 든 문서)
CASES = [
    ("IB1-10461FA3",
     "퇴직연금 실물이전 전에 사전조회는 어디에서 할 수 있는지 알려주세요. "
     "또 미래에셋프리미엄크레딧알파 A-e의 총보수도 함께 확인해 주세요.",
     ["doc35"]),
    ("IB1-8998795D",
     "타사로 퇴직연금 실물이전을 하기 전에 이전 가능 상품을 사전조회하려면 "
     "어디에서 신청하며, 홈페이지에서도 신청할 수 있나요? "
     "어느 금융기관에서 신청해야 하나요?",
     ["doc35"]),
    ("IB1-3C309B9C",
     "원리금보장상품이 처음 만기된 뒤 아무 운용지시를 하지 않았다면 디폴트옵션 "
     "자동매수까지 어떤 대기 절차가 있고, 같은 상품의 두 번째 만기에는 어떻게 되나요?",
     ["doc30"]),
    ("IB1-95267889",
     "퇴직연금 MP 구독 서비스의 위험등급이 '높은 위험(2등급)'으로 "
     "산정되는 이유는 무엇인가요?",
     ["doc54"]),
    ("IB1-226C57EC",
     "한국투자골드플랜연금증권전환형투자신탁1호(국공채)는 무엇을 주된 "
     "투자대상으로 하며 위험등급은 어떻게 분류되나요?",
     ["KR5113420012"]),
    ("IB1-C0CF9396",
     "75세에 일반 연금 형태로 받는 사람의 연금소득세율을 알려주고, "
     "하나파워e단기채 A-E와 미래에셋차세대Fun인덱스 Ae 중 총보수가 낮은 "
     "클래스도 골라주세요.",
     ["doc38"]),
    ("IB1-79E75141",
     "자영업자가 IRP 가입이 가능한지 먼저 알려주세요. 그리고 "
     "미래에셋프리미엄크레딧알파의 C-P가 퇴직연금용 클래스인지도 확인해 주세요.",
     ["doc14"]),
]


def owner(cid: str) -> str:
    """chunk_id에서 문서 식별자를 뽑는다. doc35_p1_0000 → doc35"""
    return cid.split("_")[0]


R = get_retriever()
print(f"POOL={POOL}, TOP_K={TOP_K}\n")

summary = []
for qid, q, wanted in CASES:
    st = {"question": q, "trace": []}
    route(st)
    r = st["route"]
    e = embed_query(q)

    variants = [("전체", None)]
    if r.get("hybrid"):
        variants += [("제도만", "연금문서"), ("상품만", "투자설명서")]
    elif r.get("doc_type"):
        variants = [(f"{r['doc_type']}", r["doc_type"]), ("전체", None)]

    print("=" * 78)
    print(f"{qid}  (route: doc_type={r.get('doc_type') or '전체'}"
          f"{', 복합' if r.get('hybrid') else ''})")
    print(f"  정답 문서: {wanted}")

    best = None
    for label, dt in variants:
        fused = R._fuse(q, e, r.get("fund_code"), dt)
        ranks = [(i + 1, cid) for i, (cid, _) in enumerate(fused)
                 if owner(cid) in wanted]
        top = [owner(cid) for cid, _ in fused[:8]]
        if ranks:
            pos = ranks[0][0]
            best = pos if best is None else min(best, pos)
            verdict = ("상위 8개 안 ✅" if pos <= 8
                       else f"후보엔 있으나 {pos}위 ⚠️")
        else:
            verdict = f"후보 {len(fused)}개 안에 없음 ❌"
        print(f"    [{label:6}] 후보 {len(fused):3}개 | 정답문서 최고순위: "
              f"{ranks[0][0] if ranks else '-':>4}  {verdict}")
        print(f"              상위8 → {top}")

    # 실제 파이프라인이 최종적으로 무엇을 넘기는지 (분할·병합까지 반영)
    st2 = {"question": q, "trace": [], "route": r, "query_emb": e}
    R(st2)
    final = [e2["chunk_id"] for e2 in st2["evidence"]]
    got = any(owner(c) in wanted for c in final)
    print(f"    [최종]   {'✅ 정답문서 포함' if got else '❌ 정답문서 없음'}")
    print(f"              → {[owner(c) for c in final]}")
    for t in st2["trace"]:
        if "복합" in t or "폴백" in t:
            print(f"              · {t[:110]}")
    summary.append((qid, best, got))

print()
print("=" * 78)
print("요약")
print("=" * 78)
ok = 0
for qid, best, got in summary:
    if best is None:
        tag = "후보 밖 — 검색어/표현 불일치"
    elif best <= 8:
        tag = "단일 검색 상위 8개 안"
    else:
        tag = f"단일 검색 {best}위"
    ok += bool(got)
    print(f"  {'✅' if got else '❌'} {qid}  {tag}"
          f"  |  최종 근거에 정답문서 {'포함' if got else '없음'}")
print(f"\n  최종적으로 정답 문서를 넘긴 문항: {ok}/{len(summary)}")
