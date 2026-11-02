"""
3단계: Chroma 제도 벡터 교체 + 청크 파일 교체 + BM25 재생성.

사용법:
    venv/bin/python swap_inst_chunks_v3.py              # dry-run (기본, 안전)
    venv/bin/python swap_inst_chunks_v3.py --apply       # 실제 변경
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset"
BACKUP = Path.home() / "dev" / "backup_20260831"

NEW_JSONL = DATASET / "chunks_final.NEW.jsonl"
NEW_VECTORS = DATASET / "inst_vectors.NEW.jsonl"
LIVE_JSONL = DATASET / "chunks_final.jsonl"
OLD_JSONL = DATASET / "chunks_final.OLD.jsonl"
BM25_PKL = DATASET / "bm25.pkl"
BM25_STALE = BACKUP / "bm25.stale.pkl"
CHROMA_DIR = str(DATASET / "chroma")

BATCH_SIZE = 200

_fail = 0


def _chk(label, actual, expected):
    global _fail
    ok = actual == expected
    mark = "✅" if ok else "❌"
    print(f"  {mark} {label}: {actual}  (기대: {expected})")
    if not ok:
        _fail += 1
    return ok


# ── Phase 0: 사전 점검 ───────────────────────────────
def phase0(col):
    global _fail
    _fail = 0
    print("\n" + "=" * 60)
    print("Phase 0 — 사전 점검")
    print("=" * 60)

    # NEW jsonl 줄 수 + doc_type 분포
    dt_counts = defaultdict(int)
    new_chunk_ids = set()
    new_inst_ids = set()
    line_count = 0
    with open(NEW_JSONL, encoding="utf-8") as f:
        for line in f:
            line_count += 1
            d = json.loads(line)
            dt_counts[d["doc_type"]] += 1
            new_chunk_ids.add(d["chunk_id"])
            if d["doc_type"] == "연금문서":
                new_inst_ids.add(d["chunk_id"])

    _chk("chunks_final.NEW.jsonl 줄 수", line_count, 15196)
    _chk("투자설명서", dt_counts.get("투자설명서", 0), 14273)
    _chk("연금문서", dt_counts.get("연금문서", 0), 760)
    _chk("기타", dt_counts.get("기타", 0), 163)

    # 벡터 파일
    vec_ids = set()
    vec_dims = set()
    vec_count = 0
    with open(NEW_VECTORS, encoding="utf-8") as f:
        for line in f:
            vec_count += 1
            d = json.loads(line)
            vec_ids.add(d["chunk_id"])
            vec_dims.add(len(d["embedding"]))

    _chk("inst_vectors.NEW.jsonl 줄 수", vec_count, 760)
    _chk("벡터 차원", vec_dims, {1024})
    _chk("벡터 chunk_id ↔ NEW 연금문서 chunk_id", vec_ids == new_inst_ids, True)

    # Chroma 현황
    _chk("Chroma 컬렉션 전체 개수", col.count(), 14813)
    old_ids = col.get(where={"doc_type": "연금문서"}, include=[])["ids"]
    _chk("Chroma 연금문서 id 개수", len(old_ids), 377)

    # 백업 존재
    backup_exists = (BACKUP / "chroma" / "chroma.sqlite3").exists()
    _chk("백업 chroma.sqlite3 존재", backup_exists, True)

    if _fail > 0:
        print(f"\n❌ 사전 점검 실패 {_fail}건 — 중단합니다.")
        sys.exit(1)

    print("\n✅ 사전 점검 전 항목 통과.")
    return old_ids


# ── Phase 1: 옛 제도 벡터 삭제 ───────────────────────
def phase1(col, old_ids):
    print("\n" + "=" * 60)
    print("Phase 1 — Chroma에서 옛 제도 벡터 377개 삭제")
    print("=" * 60)
    col.delete(ids=old_ids)
    after = col.count()
    print(f"  삭제 후 count: {after}")
    if after != 14436:
        print(f"  ❌ 기대값 14,436과 다릅니다. 중단합니다.")
        sys.exit(1)
    print("  ✅ 삭제 완료.")


# ── Phase 2: 신규 제도 벡터 추가 ─────────────────────
def phase2(col):
    print("\n" + "=" * 60)
    print("Phase 2 — 신규 제도 벡터 760개 추가")
    print("=" * 60)

    # 신규 청크 메타 로드 (연금문서만)
    inst_chunks = {}
    with open(NEW_JSONL, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["doc_type"] == "연금문서":
                inst_chunks[d["chunk_id"]] = d

    # 벡터 로드
    vectors = {}
    with open(NEW_VECTORS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            vectors[d["chunk_id"]] = d["embedding"]

    # 정렬해서 순서 일관성 확보
    chunk_ids = sorted(inst_chunks.keys())
    print(f"  대상: {len(chunk_ids)}개")

    # 배치 추가
    for i in range(0, len(chunk_ids), BATCH_SIZE):
        batch_ids = chunk_ids[i:i + BATCH_SIZE]
        batch_embs = [vectors[cid] for cid in batch_ids]
        batch_docs = [inst_chunks[cid]["text"] for cid in batch_ids]
        batch_metas = []
        for cid in batch_ids:
            c = inst_chunks[cid]
            meta = {
                "doc_id":       c["doc_id"],
                "doc_type":     "연금문서",
                "content_type": c["content_type"],
                "fund_code":    "",
                "fund_name":    "",
                "base_date":    "",
                "section":      c["section"],
                "source_path":  c["source_path"],
                "doc_ids":      f',{c["doc_id"]},',
                "page":         int(c["page"]) if c["page"] is not None else 0,
                "n_chars":      c["n_chars"],
                "n_docs":       1,
                "scope":        "개별",
                "is_ocr":       False,
            }
            batch_metas.append(meta)
        col.add(
            ids=batch_ids,
            embeddings=batch_embs,
            documents=batch_docs,
            metadatas=batch_metas,
        )
        print(f"    배치 {i // BATCH_SIZE + 1}: {len(batch_ids)}개 추가")

    # 검증
    total = col.count()
    inst_result = col.get(where={"doc_type": "연금문서"}, include=[])["ids"]
    print(f"\n  추가 후 count: {total}")
    print(f"  연금문서 id 수: {len(inst_result)}")
    if total != 15196:
        print(f"  ❌ 전체 count 기대 15,196 ≠ {total}. 중단합니다.")
        sys.exit(1)
    if len(inst_result) != 760:
        print(f"  ❌ 연금문서 기대 760 ≠ {len(inst_result)}. 중단합니다.")
        sys.exit(1)
    print("  ✅ 추가 완료.")


# ── Phase 3: 청크 파일 교체 ──────────────────────────
def phase3():
    print("\n" + "=" * 60)
    print("Phase 3 — 청크 파일 교체")
    print("=" * 60)
    # chunks_final.jsonl → chunks_final.OLD.jsonl
    shutil.move(str(LIVE_JSONL), str(OLD_JSONL))
    print(f"  {LIVE_JSONL.name} → {OLD_JSONL.name}")
    # chunks_final.NEW.jsonl → chunks_final.jsonl
    shutil.move(str(NEW_JSONL), str(LIVE_JSONL))
    print(f"  {NEW_JSONL.name} → {LIVE_JSONL.name}")
    print("  ✅ 교체 완료.")


# ── Phase 4: BM25 재생성 ─────────────────────────────
def phase4():
    print("\n" + "=" * 60)
    print("Phase 4 — BM25 캐시 재생성")
    print("=" * 60)
    # 기존 bm25.pkl → 백업
    if BM25_PKL.exists():
        shutil.move(str(BM25_PKL), str(BM25_STALE))
        print(f"  {BM25_PKL.name} → {BM25_STALE}")
    else:
        print("  ⚠️ bm25.pkl이 없음 (이미 옮겨졌거나 처음)")

    # 재생성
    sys.path.insert(0, str(BASE_DIR))
    from search import build_bm25
    bm25, meta = build_bm25(LIVE_JSONL, BM25_PKL)
    print(f"  BM25 meta 길이: {len(meta)}")
    if len(meta) != 15196:
        print(f"  ❌ 기대 15,196 ≠ {len(meta)}. 중단합니다.")
        sys.exit(1)
    print("  ✅ BM25 재생성 완료.")
    return meta


# ── 최종 검증 ─────────────────────────────────────────
def final_verify(col):
    print("\n" + "=" * 60)
    print("최종 검증")
    print("=" * 60)
    global _fail
    _fail = 0

    _chk("col.count()", col.count(), 15196)

    inst = col.get(where={"doc_type": "연금문서"}, include=[])["ids"]
    _chk("Chroma 연금문서 개수", len(inst), 760)
    inv = col.get(where={"doc_type": "투자설명서"}, include=[])["ids"]
    _chk("Chroma 투자설명서 개수", len(inv), 14273)
    etc = col.get(where={"doc_type": "기타"}, include=[])["ids"]
    _chk("Chroma 기타 개수", len(etc), 163)

    # jsonl 줄 수
    with open(LIVE_JSONL, encoding="utf-8") as f:
        jsonl_count = sum(1 for _ in f)
    _chk("chunks_final.jsonl 줄 수", jsonl_count, 15196)

    # BM25 meta
    import pickle
    with open(BM25_PKL, "rb") as f:
        _, meta = pickle.load(f)
    _chk("BM25 meta 길이", len(meta), 15196)

    # Chroma id ↔ jsonl chunk_id 일치
    all_chroma_ids = set(col.get(include=[])["ids"])
    all_jsonl_ids = set()
    with open(LIVE_JSONL, encoding="utf-8") as f:
        for line in f:
            all_jsonl_ids.add(json.loads(line)["chunk_id"])

    only_chroma = all_chroma_ids - all_jsonl_ids
    only_jsonl = all_jsonl_ids - all_chroma_ids
    print(f"  Chroma에만 있는 id: {len(only_chroma)}")
    print(f"  jsonl에만 있는 id: {len(only_jsonl)}")
    _chk("Chroma ↔ jsonl 완전 일치", (len(only_chroma), len(only_jsonl)), (0, 0))

    # 신규 청크 1개 메타데이터 덤프
    sample_id = sorted(inst)[0]
    sample = col.get(ids=[sample_id], include=["metadatas"])
    print(f"\n── 신규 청크 메타데이터 샘플: {sample_id} ──")
    print(json.dumps(sample["metadatas"][0], ensure_ascii=False, indent=2))
    print(f"  키 수: {len(sample['metadatas'][0])}")

    if _fail > 0:
        print(f"\n❌ 최종 검증 실패 {_fail}건!")
        sys.exit(1)

    print("\n✅ 최종 검증 전 항목 통과.")


# ── 메인 ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="3단계: Chroma 교체 + 청크 교체 + BM25 재생성")
    ap.add_argument("--apply", action="store_true",
                    help="실제로 변경한다 (기본은 dry-run)")
    args = ap.parse_args()

    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection("pension")

    # Phase 0: 사전 점검 (항상 실행)
    old_ids = phase0(col)

    if not args.apply:
        print("\n" + "=" * 60)
        print("🔒 dry-run 모드입니다. 아무것도 변경하지 않았습니다.")
        print("   실제 적용하려면:  venv/bin/python swap_inst_chunks_v3.py --apply")
        print("=" * 60)
        return

    # Phase 1~3은 반드시 연속으로 끝낸다
    phase1(col, old_ids)
    phase2(col)
    phase3()
    phase4()

    # 최종 검증 — Chroma 클라이언트를 다시 열어 디스크 반영 확인
    client2 = chromadb.PersistentClient(path=CHROMA_DIR)
    col2 = client2.get_collection("pension")
    final_verify(col2)

    print("\n" + "=" * 60)
    print("✅ 3단계 완료.")
    print("   4단계(평가셋 재실행)는 별도 지시를 기다립니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
