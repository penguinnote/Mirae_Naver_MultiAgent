from __future__ import annotations
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
Search · PY
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연금 Agent — 하이브리드 검색
=============================
 
벡터 검색(Chroma) + 키워드 검색(BM25)을 RRF로 합친다.
 
왜 하이브리드인가
----------------
연금 도메인 질의에는 두 종류가 섞여 있다.
 
  "중도인출하면 세금 얼마나 떼?"        → 의미 검색이 강하다
  "종류C-P2 판매보수 알려줘"            → 키워드 검색이 강하다
 
'종류C-P2', 'KR5157450090', '미래에셋퇴직플랜단기증권자투자신탁1호' 같은
고유명사·코드는 임베딩 공간에서 서로 가깝게 뭉쳐버려서 벡터 검색만으로는
정확한 하나를 못 집는다. BM25는 반대로 정확히 그걸 집는다.
둘을 합치면 어느 쪽 질의든 놓치지 않는다.
 
사용법
------
    pip install chromadb rank-bm25 requests python-dotenv
 
    python search.py "퇴직연금 중도인출 사유가 뭐야?"
    python search.py "판매보수 얼마야" --fund KR5157450090
    python search.py "위험등급" --type table -k 10
"""


RRF_K = 60          # Reciprocal Rank Fusion 상수. 통상 60을 쓴다.
POOL = 60           # 각 검색기에서 가져올 후보 수

RERANK_URL = "https://clovastudio.stream.ntruss.com/v1/api-tools/reranker"
RERANK_IN = 15      # 리랭커에 넘길 후보 수 (RRF 상위 N개)
RERANK_DOC_CHARS = 1200   # 문서당 잘라 보낼 길이
# 리랭커도 LLM이라 입력이 길면 느려지고 한도에 걸린다.
# 15 × 1200 = 18,000자 ≈ 9천 토큰 수준으로 맞춘다.


# ──────────────────────────────────────────────────────────────────────────
# 리랭커 (CLOVA Studio)
#
# ⚠️ 이 API는 일반적인 리랭커와 다를 수 있다.
#    네이버 쿡북 설명상 "리랭킹 + 답변 생성"을 겸하는 프롬프트 LLM이고,
#    응답 필드도 result.result 하나뿐이다.
#    → 실제 응답을 보고 파서를 맞춰야 하므로 --rerank-raw 를 먼저 돌릴 것.
# ──────────────────────────────────────────────────────────────────────────

def call_reranker(query: str, items: list[tuple[str, str]], raw: bool = False):
    """items = [(chunk_id, text), ...]

    공식 규격상 documents는 문자열 배열이 아니라
    [{"id": "...", "doc": "..."}] 형태의 객체 배열이다.
    id를 chunk_id로 주면 응답에서 어떤 청크가 선택됐는지 바로 알 수 있다.
    """
    import requests
    key = os.environ.get("CLOVA_API_KEY", "").strip()
    if not key:
        raise SystemExit("CLOVA_API_KEY가 .env에 없습니다.")
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}
    rid = os.environ.get("CLOVA_RERANK_REQUEST_ID", "").strip()
    if rid:
        headers["X-NCP-CLOVASTUDIO-REQUEST-ID"] = rid

    body = {
        "documents": [{"id": cid, "doc": txt[:RERANK_DOC_CHARS]} for cid, txt in items],
        "query": query,
        "maxTokens": int(os.environ.get("CLOVA_RERANK_MAXTOKENS", "1024")),
    }
    url = os.environ.get("CLOVA_RERANK_URL", RERANK_URL).strip()
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    js = r.json()
    return js if raw else parse_rerank(js, [cid for cid, _ in items])


def parse_rerank(js: dict, ids: list[str]) -> list[str] | None:
    """응답에서 '선택된 청크 id 목록'을 순서대로 뽑아낸다.

    응답 형태가 확정되지 않아 알려진 후보를 차례로 시도한다.
    끝내 못 찾으면 None을 반환하고 호출부가 리랭킹을 건너뛴다.
    """
    idset = set(ids)

    # ① 객체 배열로 오는 경우: [{"id": "...", "score": 0.9}, ...]
    #    CLOVA Studio 리랭커는 실제로는 "citedDocuments"에 담아 보낸다.
    #    (리랭킹+답변생성을 겸하는 도구라, 생성한 답변에 실제로 인용한
    #     문서만 순서대로 들어있다 — 15개 전부가 아니라 부분집합일 수 있다.
    #     호출부는 이 경우도 감안해 나머지를 RRF 순서로 뒤에 붙인다.)
    for path in (("result", "citedDocuments"), ("result", "documents"),
                 ("result", "results"), ("results",), ("data",), ("documents",)):
        cur = js
        try:
            for k in path:
                cur = cur[k]
        except (KeyError, TypeError):
            continue
        if isinstance(cur, list) and cur and isinstance(cur[0], dict):
            got = [d.get("id") for d in cur if d.get("id") in idset]
            if got:
                return got
            # id 없이 index로만 오는 변형
            idxs = [d.get("index", d.get("idx")) for d in cur]
            if all(isinstance(i, int) for i in idxs):
                return [ids[i] for i in idxs if 0 <= i < len(ids)]

    # ② 문자열 응답 안에 id가 섞여 나오는 경우
    txt = None
    for path in (("result", "result"), ("result", "message", "content"), ("result", "text")):
        cur = js
        try:
            for k in path:
                cur = cur[k]
            if isinstance(cur, str):
                txt = cur
                break
        except (KeyError, TypeError):
            continue
    if txt:
        seen, out = set(), []
        for cid in ids:                      # 등장 순서대로 훑는다
            pos = txt.find(cid)
            if pos >= 0 and cid not in seen:
                seen.add(cid)
                out.append((pos, cid))
        if out:
            return [cid for _, cid in sorted(out)]
    return None


# ──────────────────────────────────────────────────────────────────────────
# 한국어 토크나이저 — 형태소 분석기 없이 쓸 수 있는 실용 버전
# ──────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """단어 토큰 + 한글 2-gram.

    한국어는 조사가 붙어 '보수는/보수를/보수가'가 전부 다른 토큰이 된다.
    2-gram을 함께 넣으면 조사 차이를 넘어 매칭된다.
    형태소 분석기(kiwipiepy)를 쓰면 더 좋지만 설치 없이 이 정도면 충분히 쓸 만하다.
    """
    text = text.lower()
    words = re.findall(r"[a-z0-9]+|[가-힣]+", text)
    toks: list[str] = []
    for w in words:
        toks.append(w)
        if re.match(r"^[가-힣]+$", w) and len(w) > 2:
            toks += [w[i:i + 2] for i in range(len(w) - 1)]
    return toks


# ──────────────────────────────────────────────────────────────────────────

def build_bm25(chunks_path: Path, cache_path: Path):
    from rank_bm25 import BM25Okapi
    if cache_path.exists() and cache_path.stat().st_mtime >= chunks_path.stat().st_mtime:
        with cache_path.open("rb") as f:
            return pickle.load(f)
    print("BM25 인덱스 생성 중… (최초 1회, 이후 캐시 사용)", file=sys.stderr)
    rows = [json.loads(l)
            for l in chunks_path.open(encoding="utf-8") if l.strip()]
    bm25 = BM25Okapi([tokenize(r["text"]) for r in rows])
    meta = [{k: v for k, v in r.items() if k != "text"} | {"text": r["text"]}
            for r in rows]
    obj = (bm25, meta)
    with cache_path.open("wb") as f:
        pickle.dump(obj, f)
    return obj


def embed_query(text: str) -> list[float]:
    sys.path.insert(0, str(Path(__file__).parent))
    from embed_and_index import ClovaEmbedder, DummyEmbedder
    if os.environ.get("CLOVA_API_KEY", "").strip():
        return ClovaEmbedder().embed(text)
    print("⚠️ CLOVA_API_KEY가 없어 dummy 임베더를 씁니다 (벡터 결과 무의미)", file=sys.stderr)
    return DummyEmbedder().embed(text)


def matches(md: dict, fund: str | None, doc_type: str | None, ctype: str | None) -> bool:
    if ctype and md.get("content_type") != ctype:
        return False
    if doc_type and md.get("doc_type") != doc_type:
        return False
    if fund:
        # 개별 청크는 fund_code로, 공유 청크는 doc_ids로 판정한다.
        ids = md.get("doc_ids") or ""
        if md.get("fund_code") != fund and f",{fund}," not in ids:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="하이브리드 검색 (벡터 + BM25)")
    ap.add_argument("query")
    ap.add_argument("--db", default="./dataset/chroma")
    ap.add_argument("--collection", default="pension")
    ap.add_argument("--chunks", default="./dataset/chunks_final.jsonl")
    ap.add_argument("--bm25-cache", default="./dataset/bm25.pkl")
    ap.add_argument("-k", type=int, default=5, help="최종 결과 수")
    ap.add_argument("--fund", help="펀드표준코드로 한정 (예: KR5157450090)")
    ap.add_argument("--doc-type", dest="doc_type", help="투자설명서 | 연금문서")
    ap.add_argument("--type", dest="ctype", choices=["text", "table"])
    ap.add_argument("--vector-only", action="store_true")
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--rerank", action="store_true",
                    help="RRF 결과를 CLOVA 리랭커로 재정렬")
    ap.add_argument("--rerank-raw", action="store_true",
                    help="리랭커 응답 원문을 그대로 출력하고 종료 (규격 확인용)")
    ap.add_argument("--json", action="store_true", help="JSON으로 출력")
    a = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ranks: dict[str, dict[str, int]] = {}
    store: dict[str, dict] = {}

    # ── 벡터 검색
    if not a.bm25_only:
        import chromadb
        col = chromadb.PersistentClient(path=a.db).get_collection(a.collection)
        res = col.query(query_embeddings=[embed_query(a.query)], n_results=POOL,
                        include=["documents", "metadatas", "distances"])
        for i, (cid, doc, md) in enumerate(zip(res["ids"][0], res["documents"][0],
                                               res["metadatas"][0])):
            if not matches(md, a.fund, a.doc_type, a.ctype):
                continue
            store[cid] = {"text": doc, **md}
            ranks.setdefault(cid, {})["vec"] = i + 1

    # ── BM25 검색
    if not a.vector_only:
        bm25, meta = build_bm25(Path(a.chunks), Path(a.bm25_cache))
        scores = bm25.get_scores(tokenize(a.query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:POOL]
        for rank, i in enumerate(order):
            if scores[i] <= 0:
                break
            m = meta[i]
            md = {k: v for k, v in m.items() if k != "text"}
            md["doc_ids"] = "," + ",".join(md.get("doc_ids") or []) + ","
            if not matches(md, a.fund, a.doc_type, a.ctype):
                continue
            cid = m["chunk_id"]
            store.setdefault(cid, {"text": m["text"], **md})
            ranks.setdefault(cid, {})["bm25"] = rank + 1

    # ── RRF 융합: 두 순위의 역수를 더한다. 점수 스케일이 달라도 안전하다.
    fused_all = sorted(
        ranks.items(),
        key=lambda kv: -sum(1.0 / (RRF_K + r) for r in kv[1].values()))

    # ── 리랭킹 (선택)
    if a.rerank or a.rerank_raw:
        cand = fused_all[:RERANK_IN]
        items = [(cid, store[cid]["text"]) for cid, _ in cand]

        if a.rerank_raw:
            js = call_reranker(a.query, items, raw=True)
            total = sum(len(t[:RERANK_DOC_CHARS]) for _, t in items)
            print(f"후보 {len(items)}개 / 총 {total:,}자를 리랭커로 전송했습니다.\n")
            print("=== 응답 원문 ===")
            print(json.dumps(js, ensure_ascii=False, indent=2)[:3000])
            print("\n=== 파서 결과 ===")
            got = parse_rerank(js, [c for c, _ in items])
            print(f"  선택된 순서: {got}" if got else
                  "  ⚠️ id를 못 찾았습니다. 위 원문을 보고 parse_rerank()에 경로를 추가하세요.")
            return

        try:
            got = call_reranker(a.query, items)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 리랭커 호출 실패, RRF 순서를 그대로 씁니다: {e}", file=sys.stderr)
            got = None

        if got:
            order = {cid: i for i, cid in enumerate(got)}
            cand.sort(key=lambda kv: order.get(kv[0], 10_000))
            fused_all = cand + fused_all[RERANK_IN:]
        else:
            print("⚠️ 리랭커 응답을 해석하지 못해 RRF 순서를 유지합니다.", file=sys.stderr)

    fused = fused_all[:a.k]

    if a.json:
        out = [{"chunk_id": cid, "rrf_sources": src, **store[cid]}
               for cid, src in fused]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if not fused:
        print("검색 결과 없음.")
        return

    print(f'질의: "{a.query}"'
          + (f"  [펀드 {a.fund}]" if a.fund else "")
          + (f"  [{a.ctype}]" if a.ctype else ""))
    print("=" * 78)
    for n, (cid, src) in enumerate(fused, 1):
        d = store[cid]
        src_s = "+".join(f"{k}#{v}" for k, v in sorted(src.items()))
        title = d.get("fund_name") or f"[{d.get('n_docs')}개 문서 공통]"
        print(f"\n{n}. {title}")
        print(f"   {d.get('section') or '(섹션 없음)'} · {d.get('page')}쪽"
              f" · {d.get('content_type')}"
              + (f" · 기준일 {d['base_date']}" if d.get("base_date") else "")
              + f"   ({src_s})")
        body = d["text"]
        print("   " + (body[:400] + "…" if len(body)
              > 400 else body).replace("\n", "\n   "))
    print()


if __name__ == "__main__":
    main()
