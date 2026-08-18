#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연금 Agent — 임베딩 직전 정리 단계
==================================

build_dataset.py가 만든 chunks.jsonl을 받아 **벡터 DB에 넣을 최종본**을 만든다.

왜 이 단계가 필요한가
--------------------
투자설명서 100개는 자본시장법이 요구하는 **동일한 법정 문구**를 공유한다.
"수익자총회", "손해배상책임", "투자신탁 해지" 같은 조항은 펀드가 달라도 글자까지 같다.
그대로 임베딩하면 두 가지가 망가진다.

  1. 비용    — 같은 문장을 수십 번 임베딩한다.
  2. 검색품질 — "수익자총회가 뭐야?"에 완전히 똑같은 청크 25개가 상위를 채운다.
                정작 필요한 다른 정보가 밀려난다. 이게 더 치명적이다.

그래서 완전히 동일한 청크는 **하나만 남기고**, 어느 문서들이 공유하는지를
`doc_ids`에 기록한다. 검색할 때 `scope == "공통"`이거나 `doc_ids`에 해당 펀드가
들어있으면 잡히므로 정보 손실은 없다.

⚠️ 유사 중복은 절대 병합하지 않는다
    숫자만 다른 두 청크는 "같은 서식의 다른 수수료율"일 수 있다.
    0.72%와 1.30%를 하나로 합치면 그 순간 데이터셋이 거짓말을 시작한다.
    **완전 일치(exact match)만 병합한다.**

사용법
------
    python finalize_dataset.py --in ./dataset/chunks.jsonl --out ./dataset/chunks_final.jsonl
    python finalize_dataset.py --in ./dataset/chunks.jsonl --exclude ncp_data_test "복사본.pdf"
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

# 2개 이상의 문서가 공유하면 '공통'이다.
#
# 임의의 임계값(예: 3개 이상)을 두면 안 된다. 4개 펀드가 우연히 같은 매매회전율
# 표를 갖는 경우처럼, 공유 문서 수가 적은 청크도 특정 펀드의 것이 아니다.
# 공유되는 순간 "어느 펀드 것"이라고 말할 수 없으므로, 개별 식별자는 지우고
# doc_ids로만 추적한다.
COMMON_THRESHOLD = 2


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def find_duplicate_docs(rows: list[dict]) -> dict[str, str]:
    """같은 doc_id가 여러 경로에 있으면 정식 경로 하나만 남긴다.

    예: 'ncp_data_test/R2_KR518101002M 복사본.pdf'와
        '투자설명서 복사본/KR518101002M/R2_KR518101002M.pdf'는 같은 문서다.
    경로가 깊은 쪽(= 펀드코드 폴더 아래 정리된 쪽)을 정식으로 본다.
    """
    by_doc: dict[str, set[str]] = collections.defaultdict(set)
    for r in rows:
        by_doc[r["doc_id"]].add(r["source_path"])
    canonical = {}
    for doc_id, paths in by_doc.items():
        if len(paths) > 1:
            best = max(paths, key=lambda p: (p.count("/"), -len(p)))
            canonical[doc_id] = best
    return canonical


def main() -> None:
    ap = argparse.ArgumentParser(description="chunks.jsonl → 임베딩용 최종본")
    ap.add_argument("--in", dest="inp", default="./dataset/chunks.jsonl")
    ap.add_argument("--out", dest="out", default="./dataset/chunks_final.jsonl")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="source_path에 이 문자열이 들어가면 제외")
    ap.add_argument("--keep-duplicate-docs", action="store_true",
                    help="같은 문서의 사본을 지우지 않는다")
    a = ap.parse_args()

    inp, out = Path(a.inp), Path(a.out)
    rows = load(inp)
    n0 = len(rows)
    print(f"입력: {n0:,} 청크")

    # ── 1) 경로 기반 제외
    if a.exclude:
        pats = [nfc(p) for p in a.exclude]
        rows = [r for r in rows if not any(p in nfc(r["source_path"]) for p in pats)]
        print(f"  경로 제외:        -{n0-len(rows):,}")

    # ── 2) 같은 문서의 사본 제거
    if not a.keep_duplicate_docs:
        canon = find_duplicate_docs(rows)
        if canon:
            before = len(rows)
            rows = [r for r in rows
                    if r["doc_id"] not in canon or r["source_path"] == canon[r["doc_id"]]]
            print(f"  중복 문서 사본:   -{before-len(rows):,}  ({len(canon)}개 문서)")
            for d, p in list(canon.items())[:5]:
                print(f"      {d} → {p}")

    # ── 3) 완전 일치 청크 병합
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[hashlib.md5(r["text"].encode("utf-8")).hexdigest()].append(r)

    final = []
    for h, grp in groups.items():
        base = dict(grp[0])
        doc_ids = sorted({g["doc_id"] for g in grp})
        pages = sorted({(g["doc_id"], g["page"]) for g in grp})

        base["doc_ids"] = doc_ids
        base["n_docs"] = len(doc_ids)
        base["scope"] = "공통" if len(doc_ids) >= COMMON_THRESHOLD else "개별"
        base["text_hash"] = h[:16]

        if base["scope"] == "공통":
            # 특정 펀드의 것이 아니므로 개별 식별자를 지운다.
            # 어느 펀드에 딸린 조항인지는 doc_ids로 여전히 알 수 있다.
            base["fund_code"] = None
            base["fund_name"] = None
            base["base_date"] = None
            base["chunk_id"] = f"COMMON_{h[:12]}"
            base["source_path"] = f"[{len(doc_ids)}개 문서 공통]"
            base["page"] = pages[0][1]
        final.append(base)

    print(f"  중복 청크 병합:   -{len(rows)-len(final):,}")

    # ── 4) 저장
    final.sort(key=lambda r: (r["scope"] != "개별", r.get("doc_id") or "", r["page"]))
    with out.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 5) 리포트
    sc = collections.Counter(r["scope"] for r in final)
    ct = collections.Counter(r["content_type"] for r in final)
    chars = sum(r["n_chars"] for r in final)
    chars0 = sum(r["n_chars"] for r in rows)

    print(f"\n출력: {len(final):,} 청크  ({len(final)/n0*100:.0f}% 유지)")
    print(f"  개별 {sc.get('개별',0):,} / 공통 {sc.get('공통',0):,}")
    print(f"  본문 {ct.get('text',0):,} / 표 {ct.get('table',0):,}")
    print(f"  총 글자수 {chars:,}  (정리 전 {chars0:,} · {(1-chars/chars0)*100:.0f}% 절감)")
    print(f"  → 임베딩 토큰 약 {chars//2:,} 추정 (한국어 대략 2자/토큰)")
    print(f"\n  {out}")

    top = sorted((r for r in final if r["scope"] == "공통"),
                 key=lambda r: -r["n_docs"])[:5]
    if top:
        print("\n[가장 많이 공유된 법정문구]")
        for r in top:
            snippet = re.sub(r"\s+", " ", r["text"])[:60]
            print(f"  {r['n_docs']:>3}개 문서 · {snippet}…")


if __name__ == "__main__":
    main()
