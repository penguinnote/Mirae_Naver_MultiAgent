"""
팀원이 재청킹한 제도(연금문서) 데이터 760청크를
우리 스키마(chunks_final.jsonl 형식)로 변환한다.

입력
  ~/Downloads/db/rag_chunks_v3.json      — 신규 760청크
  ~/Downloads/db/rag_embeddings_v3.json  — 같은 760개 + embedding(1024차원)
  dataset/chunks_final.jsonl             — 현재 라이브 청크 14,813줄 (읽기만)

출력
  dataset/chunks_final.NEW.jsonl   — 투자설명서 14,273 + 기타 163 + 신규 연금문서 760
  dataset/inst_vectors.NEW.jsonl   — 760줄 {"chunk_id": <새 id>, "embedding": [...]}
  dataset/inst_id_map.NEW.json     — {"rag3_000001": "doc32_p2_0000", ...}
"""

import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET   = BASE_DIR / "dataset"
DOWNLOADS = Path.home() / "Downloads" / "db"

SRC_CHUNKS = DOWNLOADS / "rag_chunks_v3.json"
SRC_EMBEDS = DOWNLOADS / "rag_embeddings_v3.json"
LIVE_JSONL = DATASET / "chunks_final.jsonl"

OUT_JSONL   = DATASET / "chunks_final.NEW.jsonl"
OUT_VECTORS = DATASET / "inst_vectors.NEW.jsonl"
OUT_ID_MAP  = DATASET / "inst_id_map.NEW.json"


def load_new_chunks():
    with open(SRC_CHUNKS, encoding="utf-8") as f:
        return json.load(f)


def load_new_embeddings():
    with open(SRC_EMBEDS, encoding="utf-8") as f:
        data = json.load(f)
    return {x["chunk_id"]: x["embedding"] for x in data}


# ── 3-2: chunk_id 재발급 ──────────────────────────────
def assign_new_ids(chunks):
    """(document, page_start) 그룹별로 원본 순서를 유지하며 새 id를 붙인다."""
    groups = defaultdict(list)
    for i, x in enumerate(chunks):
        page = x["page_start"] if x["page_start"] is not None else 0
        key = (x["document"], page)
        groups[key].append(i)

    id_map = {}          # old_id → new_id
    new_ids = [None] * len(chunks)

    for (doc, page), indices in groups.items():
        prefix = f"{doc}_p{page}"
        for seq, idx in enumerate(indices):
            new_id = f"{prefix}_{seq:04d}"
            old_id = chunks[idx]["chunk_id"]
            id_map[old_id] = new_id
            new_ids[idx] = new_id

    return new_ids, id_map


# ── doc_title 정리 ───────────────────────────────────
def clean_doc_title(t):
    """document_title을 3~30자 문서 제목으로 정리한다."""
    t = " ".join((t or "").split())
    flat = re.sub(r"\s", "", t)
    head = flat[:4]
    p = flat.find(head, 4) if len(head) >= 4 else -1
    if 6 <= p <= 30:
        cnt, out = 0, []
        for ch in t:
            if not ch.isspace():
                cnt += 1
            if cnt > p:
                break
            out.append(ch)
        t = "".join(out).strip()
    if len(t) > 30:
        cut = t.rfind(" ", 0, 30)
        t = t[:cut if cut >= 10 else 30]
    return t.strip(" -·,")


# ── section 채우기 (5단계 개정) ──────────────────────
def pick_section(x):
    """heading_path와 table_title만 쓴다. title·document_title은 문장형이라 누출 원인."""
    hp = x.get("heading_path")
    if hp and isinstance(hp, list) and any(hp):
        return " > ".join(hp)
    if x.get("table_title"):
        return x["table_title"]
    return None


# ── 필드 매핑 ────────────────────────────────────────
def convert_chunk(x, new_id, doc_titles):
    return {
        "chunk_id":     new_id,
        "doc_id":       x["document"],
        "doc_type":     "연금문서",
        "doc_title":    doc_titles[x["document"]],
        "fund_code":    None,
        "fund_name":    None,
        "kofia_code":   None,
        "base_date":    None,
        "page":         x["page_start"],           # int 또는 None
        "section":      pick_section(x),
        "content_type": "table" if x["content_type"] == "table" else "text",
        "is_ocr":       False,
        "text":         x["text"],
        "n_chars":      len(x["text"]),
        "source_path":  f'0.docs_renamed 복사본/{x["file_name"]}',
        "doc_ids":      [x["document"]],
        "n_docs":       1,
        "scope":        "개별",
        "text_hash":    hashlib.md5(x["text"].encode("utf-8")).hexdigest()[:16],
    }


# ── 검증 ──────────────────────────────────────────────
def verify(kept_lines, new_records, new_ids, id_map, embed_map):
    ok = True

    # ── 1) 줄 수
    total = len(kept_lines) + len(new_records)
    _chk("chunks_final.NEW.jsonl 총 줄 수", total, 15196)

    # ── 2) doc_type 분포
    dt_counts = defaultdict(int)
    for line in kept_lines:
        d = json.loads(line)
        dt_counts[d["doc_type"]] += 1
    for r in new_records:
        dt_counts[r["doc_type"]] += 1
    _chk("투자설명서", dt_counts.get("투자설명서", 0), 14273)
    _chk("연금문서",   dt_counts.get("연금문서", 0),   760)
    _chk("기타",       dt_counts.get("기타", 0),       163)

    # ── 3) chunk_id 중복
    all_ids = set()
    for line in kept_lines:
        all_ids.add(json.loads(line)["chunk_id"])
    dup_new = 0
    for r in new_records:
        if r["chunk_id"] in all_ids:
            dup_new += 1
        all_ids.add(r["chunk_id"])
    new_id_set = set(r["chunk_id"] for r in new_records)
    internal_dup = len(new_records) - len(new_id_set)
    _chk("chunk_id 내부 중복", internal_dup, 0)
    _chk("신규 id ↔ 기존 잔존 id 충돌", dup_new, 0)

    # ── 4) section 채움률 (heading_path + table_title만 → 121/760)
    filled_sec = sum(1 for r in new_records if r.get("section"))
    _chk("section 채워진 신규 청크", filled_sec, 121)
    # 문장형 잔재 검사 — heading_path 원본의 구두점(1건)은 허용
    dash_sec = sum(1 for r in new_records if r.get("section") and " - " in r["section"])
    _chk("section에 ' - '가 든 것 (heading_path 원본 구두점 1건 허용)", dash_sec <= 1, True)

    # ── 4b) doc_title 검증
    has_dt = sum(1 for r in new_records if r.get("doc_title"))
    _chk("doc_title 채워진 연금문서 청크", has_dt, 760)
    dt_uniq = set(r["doc_title"] for r in new_records if r.get("doc_title"))
    # 53개: "중도인출" 6문서(doc22,46~50)의 원본 제목이 동일해서 고유값은 53
    _chk("doc_title 고유값 개수", len(dt_uniq), 53)
    dt_lens = [len(r["doc_title"]) for r in new_records if r.get("doc_title")]
    _chk("doc_title 최소 길이 ≥ 3", min(dt_lens) >= 3, True)
    _chk("doc_title 최대 길이 ≤ 30", max(dt_lens) <= 30, True)

    # ── 4c) chunk_id 집합 ↔ 라이브 파일 chunk_id 집합 일치 (Chroma 보호)
    new_all_ids = set()
    for line in kept_lines:
        new_all_ids.add(json.loads(line)["chunk_id"])
    for r in new_records:
        new_all_ids.add(r["chunk_id"])
    live_ids = set()
    with open(LIVE_JSONL, encoding="utf-8") as f:
        for line in f:
            live_ids.add(json.loads(line)["chunk_id"])
    only_new = new_all_ids - live_ids
    only_live = live_ids - new_all_ids
    _chk("chunk_id 집합 일치 (NEW에만)", len(only_new), 0)
    _chk("chunk_id 집합 일치 (LIVE에만)", len(only_live), 0)

    # ── 5) page None
    none_page = sum(1 for r in new_records if r["page"] is None)
    _chk("page가 None인 신규 청크", none_page, 315)

    # ── 6) 이웃 검증
    new_by_id = {r["chunk_id"]: r for r in new_records}
    neighbor_total = 0
    neighbor_ok = 0
    for r in new_records:
        cid = r["chunk_id"]
        # prefix_{n} 형태에서 n 추출
        parts = cid.rsplit("_", 1)
        prefix, n = parts[0], int(parts[1])
        for delta in (-1, 1):
            nb_id = f"{prefix}_{n + delta:04d}"
            if nb_id in new_by_id:
                neighbor_total += 1
                if new_by_id[nb_id]["doc_id"] == r["doc_id"]:
                    neighbor_ok += 1
    pct = (neighbor_ok / neighbor_total * 100) if neighbor_total else 0
    _chk(f"이웃 검증 ({neighbor_ok}/{neighbor_total})", pct, 100.0)

    # ── 7) 벡터 파일
    _chk("inst_vectors.NEW.jsonl 줄 수", len(embed_map), 760)
    dims = set(len(v) for v in embed_map.values())
    _chk("임베딩 차원 종류", dims, {1024})

    vec_ids = set(id_map.values())
    new_ids_set = set(new_ids)
    _chk("벡터 chunk_id ↔ 신규 청크 id 완전 일치", vec_ids == new_ids_set, True)

    return ok


_fail_count = 0


def _chk(label, actual, expected):
    global _fail_count
    match = actual == expected
    mark = "✅" if match else "❌"
    print(f"  {mark} {label}: {actual}  (기대: {expected})")
    if not match:
        _fail_count += 1


# ── 샘플 출력 ─────────────────────────────────────────
def print_samples(new_records, raw_chunks):
    raw_by_id = {x["chunk_id"]: x for x in raw_chunks}

    # doc29 FAQ 하나
    for r in new_records:
        if r["doc_id"] == "doc29":
            old = [oid for oid, nid in id_map_global.items() if nid == r["chunk_id"]]
            src = raw_by_id.get(old[0]) if old else None
            if src and src.get("question"):
                _print_sample("doc29 FAQ", r)
                break

    # doc55 하나
    for r in new_records:
        if r["doc_id"] == "doc55":
            _print_sample("doc55", r)
            break

    # table 청크 하나
    for r in new_records:
        if r["content_type"] == "table":
            _print_sample("table 청크", r)
            break


def _print_sample(label, r):
    print(f"\n── 샘플: {label} ──")
    shown = {k: v for k, v in r.items() if k != "text"}
    shown["text"] = r["text"][:200] + ("…" if len(r["text"]) > 200 else "")
    print(json.dumps(shown, ensure_ascii=False, indent=2))


# ── 메인 ──────────────────────────────────────────────
id_map_global = {}


def main():
    global id_map_global

    print("=" * 60)
    print("build_inst_chunks_v3.py — 제도 청크 어댑터 (2단계)")
    print("=" * 60)

    # 1) 신규 데이터 로드
    print("\n[1] 신규 데이터 로드")
    raw_chunks = load_new_chunks()
    embed_map_raw = load_new_embeddings()
    print(f"  청크: {len(raw_chunks)}, 임베딩: {len(embed_map_raw)}")

    # 2) chunk_id 재발급
    print("\n[2] chunk_id 재발급")
    new_ids, id_map = assign_new_ids(raw_chunks)
    id_map_global = id_map
    print(f"  재발급 완료: {len(id_map)}개")
    print(f"  예시: {list(id_map.items())[:3]}")

    # 3) 문서 제목 정리
    print("\n[3] 문서 제목 정리")
    doc_titles = {}
    for x in raw_chunks:
        doc = x["document"]
        if doc not in doc_titles:
            doc_titles[doc] = clean_doc_title(x.get("document_title", ""))
    print(f"  문서 {len(doc_titles)}개 제목 추출")
    for doc in sorted(doc_titles):
        print(f"    {doc}: {doc_titles[doc]}")

    # 4) 필드 매핑
    print("\n[4] 필드 매핑 → 신규 레코드 생성")
    new_records = []
    for i, x in enumerate(raw_chunks):
        new_records.append(convert_chunk(x, new_ids[i], doc_titles))
    print(f"  생성: {len(new_records)}개")

    # 5) 기존 청크에서 비-연금문서만 유지
    print("\n[5] 기존 청크 필터링 (연금문서 제거)")
    kept_lines = []
    dropped = 0
    with open(LIVE_JSONL, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["doc_type"] == "연금문서":
                dropped += 1
            else:
                kept_lines.append(line.rstrip("\n"))
    print(f"  유지: {len(kept_lines)}, 제거: {dropped}")

    # 6) 임베딩 → 새 id로 매핑
    print("\n[6] 임베딩 id 변환")
    embed_new = {}
    for old_id, emb in embed_map_raw.items():
        embed_new[id_map[old_id]] = emb
    print(f"  변환: {len(embed_new)}개")

    # 7) 검증
    print("\n[7] 검증")
    verify(kept_lines, new_records, new_ids, id_map, embed_new)

    if _fail_count > 0:
        print(f"\n❌ 검증 실패 {_fail_count}건 — 파일을 쓰지 않습니다.")
        sys.exit(1)

    # 8) 파일 쓰기
    print("\n[8] 파일 쓰기")

    # chunks_final.NEW.jsonl
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sz = OUT_JSONL.stat().st_size
    print(f"  {OUT_JSONL}  ({sz:,} bytes)")

    # inst_vectors.NEW.jsonl
    with open(OUT_VECTORS, "w", encoding="utf-8") as f:
        for r in new_records:
            row = {"chunk_id": r["chunk_id"], "embedding": embed_new[r["chunk_id"]]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sz = OUT_VECTORS.stat().st_size
    print(f"  {OUT_VECTORS}  ({sz:,} bytes)")

    # inst_id_map.NEW.json
    with open(OUT_ID_MAP, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    sz = OUT_ID_MAP.stat().st_size
    print(f"  {OUT_ID_MAP}  ({sz:,} bytes)")

    # 9) 샘플 출력
    print("\n[9] 샘플 청크")
    print_samples(new_records, raw_chunks)

    # 10) 라이브 파일 무결성 확인
    print("\n[10] 라이브 파일 무결성")
    live_mtime = os.path.getmtime(LIVE_JSONL)
    print(f"  chunks_final.jsonl mtime: {live_mtime}")
    print("  (이 값이 스크립트 실행 전과 같은지 확인)")

    print("\n" + "=" * 60)
    print("✅ 완료. 3단계(Chroma 교체 + BM25 재생성)는 별도 지시를 기다립니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
