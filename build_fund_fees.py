#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연금 Agent — fund_fees SQL 테이블 구축
=======================================

chunks_final.jsonl 안의 표 청크(content_type == "table")에서
"클래스 종류 / 총보수" 계열의 수수료·보수 표를 찾아 정형 데이터로 뽑아
SQLite 테이블(fund_fees)로 만든다.

왜 필요한가
-----------
벡터 검색은 "0.5% 이하", "가장 싼 5개" 같은 정확한 비교·정렬을 못 한다.
같은 표 청크를 두 번째로 가공해서(벡터화와는 별개로) SQL로 조회 가능하게 만든다.

설계
----
1. 표 하나가 여러 마크다운 줄로 흩어져 있고, 헤더가 1~3줄에 걸쳐 나뉜다.
   컬럼 위치가 펀드마다 다르므로 **헤더 텍스트로 컬럼을 찾는다**(고정 인덱스 금지).
2. "클래스" 헤더 셀 주변 몇 줄 안에 "총보수" 컬럼이 없으면 수수료 표가 아니라고 보고 건너뛴다
   (환매/전환수수료 표, 설정·환매 좌수 표, 비용예시(원화) 표 등 다른 표들이 있음).
3. 완전히 다른 레이아웃(표가 통째로 뭉개져 셀 구분이 안 되는 문서)은 --skipped-csv로 따로
   내보내고 건너뛴다. 잘못된 값을 넣느니 빼는 게 낫다.

사용법
------
    python build_fund_fees.py --in ./dataset/chunks_final.jsonl --db ./dataset/fund_fees.sqlite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# 클래스 코드 → 계좌유형/가입경로 매핑 (멀티에이전트-설계안.md 근거)
# 코드로 못 알아내면 라벨 텍스트(연금저축/퇴직연금 등)로 폴백한다.
# ──────────────────────────────────────────────────────────────────────────

CODE_ACCOUNT_HINTS = {
    "P": "연금저축", "PE": "연금저축", "P2": "퇴직연금", "P2E": "퇴직연금",
}

HEADER_KEYS = {
    "class": ["클래스종류", "클래스 종류"],
    "front_load": ["판매수수료", "판매 수수료"],
    "fee_total": ["총보수"],           # 아래에서 동종유형/비용 포함 여부로 구분
    "distribution": ["판매보수"],
    "peer_avg": ["동종유형"],
    "total_cost_ratio": ["비용"],       # "총보수ㆍ비용" / "총보수·비용" 등, fee_total과 겹치므로 후처리로 구분
}


def cells(line: str) -> list[str]:
    """마크다운 표 한 줄을 셀 리스트로. 맨 앞/뒤 빈 칸(파이프 경계)은 버린다."""
    parts = [c.strip() for c in line.split("|")]
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def is_separator(line: str) -> bool:
    return bool(re.match(r"^\|?[\s:|-]+\|?$", line)) and "-" in line


def norm_header(s: str) -> str:
    return re.sub(r"\s+", "", s)


NUM_RE = re.compile(r"^-?\d[\d,]*\.?\d*\s*%?$")


def is_numeric_cell(s: str) -> bool:
    return bool(s) and bool(NUM_RE.match(s.strip()))


def find_fee_tables(text: str):
    """청크 텍스트 안에서 '클래스+총보수' 표를 전부 찾아 (헤더 컬럼맵, 데이터행들) 리스트로.

    PDF에서 표를 뽑을 때 셀 안의 줄바꿈이 별도의 표 행으로 쪼개지는 경우가 흔하다
    (예: "클래스"와 "종류"가 다른 줄에, "수수료선취-"와 "오프라인(A)"이 다른 줄에).
    그래서 라벨 텍스트 패턴이 아니라 **숫자 셀의 존재 여부**로 헤더/데이터 경계와
    "라벨만 있고 숫자는 없는 이어붙임 행"을 구분한다 — 훨씬 견고하다.

    헤더 컬럼 텍스트는 위→아래로 내려가며 비어있지 않은 값으로 덮어써서
    가장 안쪽(구체적인) 라벨을 쓴다. 바깥쪽 그룹 제목이 안쪽 컬럼명을
    오염시키는 걸 막기 위함이다.
    """
    lines = text.split("\n")
    tables = []
    i = 0
    while i < len(lines):
        if is_separator(lines[i]):
            i += 1
            continue
        anchor_cells = cells(lines[i])
        # "클래스"만 있어도 앵커로 인정한다 — "종류"가 다음 줄로 밀려나는 경우가 있다.
        class_col = next((k for k, v in enumerate(anchor_cells) if "클래스" in norm_header(v)), None)
        if class_col is None:
            i += 1
            continue

        ncols = len(anchor_cells)
        header_rows = [anchor_cells]
        j = i + 1
        while j < len(lines) and len(header_rows) < 8:
            if is_separator(lines[j]):
                j += 1
                continue
            c = cells(lines[j])
            if len(c) != ncols:
                break
            if sum(1 for x in c if is_numeric_cell(x)) >= 2:
                break  # 숫자가 2개 이상 나오면 헤더가 끝나고 데이터가 시작된 것
            header_rows.append(c)
            j += 1

        merged = ["" for _ in range(ncols)]
        for row in header_rows:
            for k, v in enumerate(row):
                if v:
                    merged[k] = v  # 안쪽(나중) 값으로 덮어씀 — 이어붙이지 않음
        merged_n = [norm_header(m) for m in merged]

        fee_total_col = next(
            (k for k, m in enumerate(merged_n)
             if k != class_col and "총보수" in m and "동종유형" not in m and "비용" not in m), None)
        if fee_total_col is None:
            # 총보수 컬럼이 없는 표(환매/전환수수료, 좌수 변동 등) — 우리가 원하는 표가 아님
            i = j
            continue

        front_load_col = next(
            (k for k, m in enumerate(merged_n) if "판매수수료" in m), None)
        distribution_col = next(
            (k for k, m in enumerate(merged_n) if "판매보수" in m), None)
        peer_avg_col = next(
            (k for k, m in enumerate(merged_n) if "동종유형" in m), None)
        total_cost_col = next(
            (k for k, m in enumerate(merged_n)
             if "총보수" in m and "비용" in m and k != fee_total_col), None)

        colmap = dict(class_col=class_col, front_load_col=front_load_col,
                      fee_total_col=fee_total_col, distribution_col=distribution_col,
                      peer_avg_col=peer_avg_col, total_cost_col=total_cost_col)

        # ── 데이터 행: 숫자가 있으면 새 행, 라벨만 있고 숫자가 없으면 직전 행 라벨에 이어붙인다
        data_rows: list[list[str]] = []
        k = j
        blank_streak = 0
        while k < len(lines):
            if is_separator(lines[k]):
                k += 1
                continue
            c = cells(lines[k])
            if len(c) != ncols:
                break
            label = c[class_col] if class_col < len(c) else ""
            n_numeric = sum(1 for x in c if is_numeric_cell(x))
            prev_label = data_rows[-1][class_col] if data_rows else ""
            prev_complete = bool(re.search(r"[)）]\s*$", prev_label))
            if n_numeric >= 1:
                data_rows.append(list(c))
                blank_streak = 0
            elif label and data_rows and len(label) <= 20 and not prev_complete:
                # 짧은 텍스트만, 그리고 직전 라벨이 아직 괄호로 안 닫혔을 때만 이어붙인다.
                # 이미 "(C-Re)"처럼 닫힌 라벨 뒤에 오는 "지급시기" 같은 다음 표 제목이
                # 잘못 붙는 걸 막는다.
                data_rows[-1][class_col] = (data_rows[-1][class_col] + label).strip()
                blank_streak += 1
                if blank_streak > 2:
                    break
            else:
                break
            k += 1

        if data_rows:
            tables.append((colmap, data_rows))
        i = k if k > i else i + 1

    return tables


CLASS_LABEL_RE = re.compile(r"[(（]([A-Za-z0-9가-힣\-]+)[)）]\s*$")
# "(C-P2(퇴직연금))"처럼 괄호 안에 괄호가 또 있는 경우, 맨 앞(가장 짧은) 코드를 우선 잡는다.
CLASS_LABEL_NESTED_RE = re.compile(r"[(（]([A-Za-z0-9\-]+)[(（]")


def looks_like_class_label(label: str) -> bool:
    if not label:
        return False
    if label.startswith("*") or label.startswith("※") or label.startswith("주"):
        return False
    if len(label) > 40:
        return False
    return bool(CLASS_LABEL_RE.search(re.sub(r"\s+", "", label)))


def parse_class_code(label: str) -> str | None:
    clean = re.sub(r"\s+", "", label)
    m = CLASS_LABEL_NESTED_RE.search(clean)
    if m:
        return m.group(1)
    m = CLASS_LABEL_RE.search(clean)
    return m.group(1) if m else None


def parse_num(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s in ("", "-", "–", "없음", "실비", "해당없음"):
        return None
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


def infer_account_type(label: str, code: str | None) -> str | None:
    label_n = re.sub(r"\s+", "", label)
    if "퇴직연금" in label_n:
        return "퇴직연금"
    if "연금저축" in label_n or "개인연금" in label_n:
        return "연금저축"
    if code:
        c = code.upper().replace("-", "")
        for k, v in CODE_ACCOUNT_HINTS.items():
            if c.endswith(k):
                return v
    return None


def infer_channel(label: str) -> str | None:
    label_n = re.sub(r"\s+", "", label)
    if "온라인슈퍼" in label_n:
        return "온라인슈퍼"
    if "온라인" in label_n:
        return "온라인"
    if "오프라인" in label_n:
        return "오프라인"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="표 청크에서 fund_fees SQL 테이블 구축")
    ap.add_argument("--in", dest="inp", default="./dataset/chunks_final.jsonl")
    ap.add_argument("--db", default="./dataset/fund_fees.sqlite")
    ap.add_argument("--csv", default="./dataset/fund_fees_preview.csv",
                    help="눈으로 확인하기 쉽게 CSV로도 저장")
    ap.add_argument("--skipped-csv", default="./dataset/fund_fees_skipped.csv",
                    help="'클래스+총보수'는 매칭됐지만 데이터 행을 못 뽑은 청크 목록")
    a = ap.parse_args()

    rows_out = []
    skipped = []
    seen_funds_with_rows = set()
    candidate_chunks = 0

    with open(a.inp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("content_type") != "table":
                continue
            text = r["text"]
            if "클래스" not in text or "총보수" not in text:
                continue
            candidate_chunks += 1
            tables = find_fee_tables(text)
            if not tables:
                skipped.append((r.get("chunk_id"), r.get("fund_code"), r.get("fund_name")))
                continue

            for colmap, data_rows in tables:
                for c in data_rows:
                    label = c[colmap["class_col"]]
                    code = parse_class_code(label)

                    def get(col_key):
                        idx = colmap[col_key]
                        if idx is None or idx >= len(c):
                            return None
                        return c[idx]

                    row = dict(
                        fund_code=r.get("fund_code"),
                        fund_name=r.get("fund_name"),
                        base_date=r.get("base_date"),
                        class_label=label,
                        class_code=code,
                        account_type=infer_account_type(label, code),
                        channel=infer_channel(label),
                        front_load_text=get("front_load_col"),
                        fee_total=parse_num(get("fee_total_col")),
                        fee_distribution=parse_num(get("distribution_col")),
                        fee_peer_avg=parse_num(get("peer_avg_col")),
                        fee_total_cost=parse_num(get("total_cost_col")),
                        chunk_id=r.get("chunk_id"),
                        page=r.get("page"),
                        source_path=r.get("source_path"),
                    )
                    # 총보수는 항상 %(0~10 사이)다. 표 열이 잘못 잡혀 "1,000만원 투자시
                    # 비용"(천원 단위, 수십~수백) 같은 다른 컬럼 값이 들어온 경우를 걸러낸다.
                    if row["fee_total"] is None or not (0 <= row["fee_total"] <= 10):
                        continue
                    rows_out.append(row)
                    if row["fund_code"]:
                        seen_funds_with_rows.add(row["fund_code"])

    # ── 중복 제거
    # 같은 표가 "요약정보"와 본문에 두 번 인쇄된 문서가 있어 (fund_code, class_code)가
    # 겹칠 수 있다. fee_total까지 완전히 같으면 같은 사실의 중복 인용이므로 하나만 남긴다.
    # fee_total이 다르면 데이터가 상충하는 것이므로 지우지 않고 둘 다 남기고 알린다.
    dedup_seen = set()
    conflicts = []
    deduped = []
    by_key = {}
    for row in rows_out:
        key = (row["fund_code"], row["class_code"])
        by_key.setdefault(key, []).append(row["fee_total"])
    for row in rows_out:
        full_key = (row["fund_code"], row["class_code"], row["fee_total"])
        if full_key in dedup_seen:
            continue
        dedup_seen.add(full_key)
        deduped.append(row)
    for key, totals in by_key.items():
        if key[1] and len(set(totals)) > 1:   # class_code가 없는(None) 키는 서로 다른 클래스가
            conflicts.append((key, sorted(set(totals))))  # 우연히 묶인 것이라 진짜 충돌이 아님
    rows_out = deduped

    # ── SQLite 적재
    db_path = Path(a.db)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    con.execute("""
        CREATE TABLE fund_fees (
            fund_code TEXT, fund_name TEXT, base_date TEXT,
            class_label TEXT, class_code TEXT,
            account_type TEXT, channel TEXT,
            front_load_text TEXT,
            fee_total REAL, fee_distribution REAL,
            fee_peer_avg REAL, fee_total_cost REAL,
            chunk_id TEXT, page INTEGER, source_path TEXT
        )
    """)
    con.executemany(
        f"INSERT INTO fund_fees VALUES ({','.join('?' * 15)})",
        [tuple(row.values()) for row in rows_out])
    con.execute("CREATE INDEX idx_fund_code ON fund_fees(fund_code)")
    con.execute("CREATE INDEX idx_account_type ON fund_fees(account_type)")
    con.commit()
    con.close()

    # ── CSV (눈으로 확인용)
    if rows_out:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    with open(a.skipped_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "fund_code", "fund_name"])
        w.writerows(skipped)

    # ── 요약
    total_funds_candidate = len({r.get("fund_code") for r in
                                  [json.loads(l) for l in open(a.inp, encoding="utf-8")]
                                  if r.get("content_type") == "table"
                                  and "클래스" in r["text"] and "총보수" in r["text"]})
    print(f"후보 청크 {candidate_chunks:,}개 중 표 인식 실패 {len(skipped):,}개")
    print(f"수수료 행 {len(rows_out):,}건 추출 → {a.db}")
    print(f"펀드 {len(seen_funds_with_rows):,}/{total_funds_candidate:,}개에서 데이터 확보")
    if conflicts:
        print(f"\n⚠️ 같은 (펀드,클래스)인데 총보수 값이 다른 경우 {len(conflicts)}건 — 확인 필요:")
        for key, totals in conflicts[:10]:
            print(f"   {key}: {totals}")
    print(f"미리보기: {a.csv}")
    print(f"건너뛴 목록: {a.skipped_csv}")


if __name__ == "__main__":
    main()
