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


def _row_label(c: list[str], class_col: int) -> str:
    """행의 실제 클래스 라벨 텍스트를 찾는다.

    헤더에서 잡은 class_col 자리에 값이 있으면 그대로 쓴다. 없으면(같은 표
    안에서도 행마다 들여쓰기가 달라 라벨이 한 칸 옆으로 밀리는 경우가 흔하다)
    첫 숫자 셀 앞에 있는 첫 비어있지 않은 셀을 라벨로 본다 — 이 표들은
    항상 '라벨 다음에 숫자'라는 순서를 지킨다.
    """
    if class_col < len(c) and c[class_col]:
        return c[class_col]
    first_num = next((idx for idx, x in enumerate(c) if is_numeric_cell(x)), len(c))
    for x in c[:first_num]:
        if x:
            return x
    return ""


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
        # 2부 13절 "보수 및 비용" 내역표는 헤더가 "클래스"가 아니라 "종류"다
        # (실측: KR5127420034 p.25 — C-P 0.471 등 연금 클래스가 이 표에만 있다).
        # "종류"는 흔한 단어라 셀 전체가 정확히 "종류"일 때만 앵커로 본다.
        # 이 표의 행 라벨은 괄호 없는 코드("C-P")라 bare_codes로 표시해 둔다.
        bare_codes = False
        if class_col is None:
            class_col = next((k for k, v in enumerate(anchor_cells)
                              if norm_header(v) == "종류"), None)
            bare_codes = class_col is not None
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
        seen = ["" for _ in range(ncols)]  # "동종유형" 판별 전용 — 절대 덮어쓰지 않고 이어붙인다
        for row in header_rows:
            for k, v in enumerate(row):
                if v:
                    merged[k] = v  # 안쪽(나중) 값으로 덮어씀 — 이어붙이지 않음
                    seen[k] = seen[k] + v
        merged_n = [norm_header(m) for m in merged]
        seen_n = [norm_header(m) for m in seen]

        # "동종유형"은 여러 표 행에 걸쳐 "동종"/"유형"으로 쪼개지는 경우가 있는데,
        # merged_n은 마지막 조각만 남기므로("총보수"만 남고 "동종유형"이 사라짐)
        # 이 필터만은 절대 안 잃어버리는 seen_n(이어붙인 값)으로 판별한다.
        # (실측: KR5122420005 p37 표 — "동종유형총보수" 컬럼이 "총보수"로 오인식돼
        # fee_total_col으로 잘못 잡혔고, 같은 펀드의 총보수가 표마다 0.30/0.34로
        # 어긋나는 원인이었다.)
        def _pick_fee_total_col(text_list):
            # "판매수수료 총 보수 판매보수"처럼 여러 컬럼 제목이 한 셀에 뭉개진
            # 열은 제외한다. 실측(KR5194450018 p.5): 이 열이 fee_total로 잡혀
            # 판매수수료율("납입금액의 1% 이내"→1.0)이 총보수로 실렸다 —
            # 진짜 총보수 열(다음 헤더 줄의 "총 보수")은 그 옆에 따로 있다.
            return next(
                (k for k, m in enumerate(text_list)
                 if k != class_col and "총보수" in m and "동종유형" not in seen_n[k]
                 and "판매수수료" not in seen_n[k] and "비용" not in m),
                None)

        fee_total_col = _pick_fee_total_col(merged_n)
        if fee_total_col is None:
            # merged_n은 "총"과 "보수"가 서로 다른 헤더 행에 걸쳐 있으면 마지막 조각
            # ("보수"만)만 남겨 "총보수" 매칭 자체가 실패하는 경우가 있다. 이때만
            # 안 잃어버리는 seen_n으로 다시 찾는다 — merged_n으로 이미 찾은 표는
            # 손대지 않으므로 기존에 맞던 표에는 영향이 없다.
            # (실측: KR5111420047·KR5111450067 — 실제 "동종유형" 평균값이 이 펀드
            # 자신의 총보수로 잘못 기록되고 있었다. 데이터가 아예 없는 것보다
            # 틀린 숫자가 있는 쪽이 더 나쁘다.)
            fee_total_col = _pick_fee_total_col(seen_n)
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
                      peer_avg_col=peer_avg_col, total_cost_col=total_cost_col,
                      bare_codes=bare_codes)

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
            n_numeric = sum(1 for x in c if is_numeric_cell(x))
            row_label = _row_label(c, class_col)
            prev_label = data_rows[-1][class_col] if data_rows else ""
            # '종류' 내역표의 라벨은 한 셀짜리 코드라 항상 완결이다. 이어붙이면
            # 표 아래 '지급'/'시기' 줄이 마지막 클래스 코드에 달라붙는다
            # (실측: 'S-퇴직'이 'S-퇴직지급시기'가 됨).
            prev_complete = bare_codes or bool(re.search(r"[)）]\s*$", prev_label))
            if n_numeric >= 1:
                # 들여쓰기가 행마다 달라 라벨이 헤더가 잡은 class_col이 아니라
                # 한 칸 옆에 찍히는 경우가 있다(같은 표 안에서도 행마다 다름).
                # _row_label이 실제 위치를 찾아주므로, class_col 자리를 그 값으로
                # 맞춰 넣어야 이후 로직(이어붙이기·최종 추출)이 정상 동작한다.
                if class_col < len(c) and row_label:
                    c = list(c)
                    c[class_col] = row_label
                data_rows.append(list(c))
                blank_streak = 0
            elif row_label and data_rows and len(row_label) <= 20 and not prev_complete:
                # 짧은 텍스트만, 그리고 직전 라벨이 아직 괄호로 안 닫혔을 때만 이어붙인다.
                # 이미 "(C-Re)"처럼 닫힌 라벨 뒤에 오는 "지급시기" 같은 다음 표 제목이
                # 잘못 붙는 걸 막는다.
                data_rows[-1][class_col] = (data_rows[-1][class_col] + row_label).strip()
                blank_streak += 1
                if blank_streak > 4:
                    break
            elif row_label and len(row_label) <= 20:
                # 라벨은 있지만(직전 라벨이 이미 닫혔거나 아직 데이터가 없어서) 이어붙일
                # 곳이 없는 짧은 잡음 줄 — "투자비용" 같은 절 제목이 표 중간에 한 줄
                # 끼어드는 경우다. 예전엔 여기서 표가 끝난 것으로 보고 break해서 뒤에
                # 남은 진짜 데이터 행(예: 온라인 클래스들)이 통째로 사라졌다.
                # 건너뛰고 계속 스캔한다.
                blank_streak += 1
                if blank_streak > 4:
                    break
            else:
                # 라벨도 없거나(완전 공백 행) 각주처럼 긴 문단이 시작된 것 — 진짜 표 끝.
                break
            k += 1

        if data_rows:
            tables.append((colmap, data_rows))
        i = k if k > i else i + 1

    return tables


# ── 요약정보(제1부 앞머리) 표가 텍스트로 풀린 레이아웃 ─────────────────────
#
# PDF에 따라 4~5쪽 요약표가 마크다운 표가 아니라 평문으로 뽑힌다. 행 하나가
# 세 줄로 쪼개진다: 라벨 위쪽 조각 / 숫자 줄 / 라벨 아래쪽 조각(클래스 코드 포함).
#
#     수수료선취- 납입금액의
#     0.383 0.203 0.340 0.383 69 110 153 245 516
#     오프라인(A) 0.30%이내
#
# 숫자 줄의 열 순서는 확정돼 있다(2026-09-01, 우리나라초단기채권 p.5로 검증):
#     [총보수] [판매보수] [동종유형 총보수] [총보수·비용(합성)] | 1년 2년 3년 5년 10년
# 앞 4개가 보수율(%), 뒤 5개는 1,000만원 투자시 기간별 총비용(천원)이라 버린다.
# 뒤 5개를 보수율로 실으면 안 된다 — 그게 배포 서버 환각의 원인이었다.
#
# 인식 규칙: 줄 끝에서부터 숫자·대시 토큰 run이 9개 이상이고, 마지막 5개는
# 정수(비용, 소수점 없음), 그 앞 4개는 0~10 사이 소수(요율) 또는 '-'여야 한다.
# 투자실적추이(숫자 5개)나 포트폴리오 표는 이 구조가 아니라서 걸리지 않는다.
# 클래스 코드가 괄호로 확인된 행만 싣는다 — 코드 없는 숫자 줄은 오인식 위험이
# 커서 버린다.

# 요율에 %를 붙이는 문서가 있다(실측: 베어링 고배당 "1.434% 0.700% …" —
# % 때문에 fee 줄로 인식되지 않아 펀드가 통째로 빠졌다). parse_num이 %를 벗긴다.
_NUMTOK_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?%?$")
_SUMMARY_CODE_RE = re.compile(r"[(（]([A-Za-z][A-Za-z0-9]{0,3}(?:-[A-Za-z0-9가-힣]{1,6})?)[)）]")
_LABEL_HINTS = ("수수료", "미징구", "선취", "후취", "오프라인", "온라인", "퇴직", "연금")
_NOTE_RE = re.compile(r"^[(（]?주\d|^※")


def _numeric_tail(tokens: list[str]) -> list[str]:
    run = []
    for t in reversed(tokens):
        if t == "-" or _NUMTOK_RE.match(t):
            run.append(t)
        else:
            break
    return list(reversed(run))


def _is_cost_tok(t: str) -> bool:
    return t == "-" or (bool(_NUMTOK_RE.match(t)) and "." not in t)


def _is_rate_tok(t: str) -> bool:
    if t == "-":
        return True
    if "." not in t:
        return False
    v = parse_num(t)
    return v is not None and 0 <= v <= 10


def find_summary_fee_rows(text: str) -> list[dict]:
    """요약표 텍스트 레이아웃에서 (라벨, 코드, 요율 4종) 행들을 뽑는다."""
    lines = [ln.strip() for ln in text.split("\n")]
    rows = []
    pending: list[str] = []   # 다음 숫자 줄의 위쪽 라벨 조각
    i = 0
    while i < len(lines):
        ln = lines[i]
        toks = ln.split()
        tail = _numeric_tail(toks)
        is_fee_line = (len(tail) >= 9
                       and all(_is_cost_tok(t) for t in tail[-5:])
                       and all(_is_rate_tok(t) for t in tail[-9:-5]))
        if not is_fee_line:
            # 라벨 조각 후보만 pending에 쌓는다. 아무 줄이나 쌓으면 헤더 잡음이
            # 라벨에 섞여 채널·계좌유형 추정을 오염시킨다.
            if (toks and len(ln) <= 40 and not _NOTE_RE.match(ln)
                    and (any(h in ln for h in _LABEL_HINTS) or _SUMMARY_CODE_RE.search(ln))):
                pending.append(ln)
                if len(pending) > 3:
                    pending.pop(0)
            else:
                pending = []
            i += 1
            continue

        rates = tail[-9:-5]
        extras = tail[:-9]                       # 요율 앞의 '-' 등 — 판매수수료 자리
        head = toks[:len(toks) - len(tail)]      # 숫자 줄 안의 라벨 조각
        top = pending
        pending = []

        # 아래쪽 라벨 조각: 코드가 나올 때까지 최대 2줄. 다음 숫자 줄은 먹지 않는다.
        #
        # 코드는 반드시 숫자 줄 자신(head) 또는 아래쪽 조각에서만 찾는다.
        # 실제 레이아웃에서 "(A-E)" 같은 코드는 항상 숫자 줄과 같은 줄이거나
        # 그 아래 줄에 온다. 위쪽 조각(pending)에서 찾으면, 청크 경계가 표
        # 중간을 지날 때 앞 행의 아래쪽 조각이 chunk 첫머리에 남아 **앞 행의
        # 코드가 다음 행에 붙는다** (실측: KR5111420047 p5_0009 오버랩 청크에서
        # A-E가 C-E의 0.1303을 가져갔다). 채널·계좌유형 추정도 같은 이유로
        # own(head+bottom)을 먼저 보고, 거기 없을 때만 위쪽 조각까지 본다.
        own_parts = list(head)
        j = i + 1
        consumed = 0
        while (_SUMMARY_CODE_RE.search(re.sub(r"\s+", "", "".join(own_parts)) or "") is None
               and j < len(lines) and consumed < 2):
            nxt = lines[j]
            ntoks = nxt.split()
            if not ntoks or len(nxt) > 40 or _NOTE_RE.match(nxt):
                break
            if len(_numeric_tail(ntoks)) >= 9:
                break
            own_parts.append(nxt)
            j += 1
            consumed += 1

        own = " ".join(own_parts).strip()
        label = " ".join(top + own_parts).strip()
        m = _SUMMARY_CODE_RE.search(re.sub(r"\s+", "", own))
        if m:
            front = None
            if "없음" in toks or any("없음" in x for x in own_parts):
                front = "없음"
            elif any(t == "-" for t in extras):
                front = "-"
            rows.append(dict(
                label=label, code=m.group(1), front_load_text=front,
                own_label=own,
                fee_total=parse_num(rates[0]),
                fee_distribution=parse_num(rates[1]),
                fee_peer_avg=parse_num(rates[2]),
                fee_total_cost=parse_num(rates[3]),
            ))
        i = j if j > i + 1 else i + 1
    return rows


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


CLASS_LABEL_PREFIX_RE = re.compile(r"^([A-Za-z0-9\-]{1,8})[(（]")
# "A(수수료선취-오프라인)"처럼 코드가 앞에, 설명이 괄호 안에 있는 라벨도 있다
# (실측: 신영밸류고배당·신영마라톤). 끝괄호 정규식은 이 경우 설명 문구를
# 통째로 코드로 잘못 잡는다("수수료선취-오프라인"이 class_code가 됨).


def parse_class_code(label: str) -> str | None:
    clean = re.sub(r"\s+", "", label)
    m = CLASS_LABEL_NESTED_RE.search(clean)
    if m:
        return m.group(1)
    m = CLASS_LABEL_RE.search(clean)
    if m and not re.search(r"[가-힣]{2,}", m.group(1)):
        # 잡힌 문자열이 진짜 코드처럼 짧고 한글 설명이 아닐 때만 신뢰한다.
        return m.group(1)
    m2 = CLASS_LABEL_PREFIX_RE.search(clean)
    if m2:
        return m2.group(1)
    return None


def parse_num(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s in ("", "-", "–", "없음", "실비", "해당없음"):
        return None
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


def infer_account_type(label: str, code: str | None) -> str | None:
    # 판매수수료 칸의 '없음'과 그룹 제목 '투자비용'이 줄나눔 탓에 라벨
    # 한가운데 끼어 '퇴 없음 직연금'·'퇴 투자비용 직연금'처럼 단어를 쪼개는
    # 행이 실측 2건 있다(C-e). 의미 없는 토큰이니 지우고 나서 본다.
    label_n = re.sub(r"\s+", "", label).replace("없음", "").replace("투자비용", "")
    if "퇴직연금" in label_n:
        return "퇴직연금"
    if "연금저축" in label_n or "개인연금" in label_n:
        return "연금저축"
    # 'S-퇴직'·'C-퇴직e'처럼 코드가 곧 라벨인 행(2부 내역표)은 '퇴직연금'
    # 완전일치가 안 잡힌다. 보수표 라벨에서 '퇴직'은 퇴직연금뿐이라(실측)
    # 부분일치로 보완한다 — 홀드아웃 H3-11이 이 4행 때문에 50%에 막혔다.
    if "퇴직" in label_n:
        return "퇴직연금"
    if code:
        c = code.upper().replace("-", "")
        for k, v in CODE_ACCOUNT_HINTS.items():
            if c.endswith(k):
                return v
    return None


def infer_channel(label: str, code: str | None = None) -> str | None:
    label_n = re.sub(r"\s+", "", label)
    if "온라인슈퍼" in label_n:
        return "온라인슈퍼"
    if "온라인" in label_n:
        return "온라인"
    if "오프라인" in label_n:
        return "오프라인"
    # 2부 내역표('종류' 앵커)는 라벨이 맨 코드라 채널 단어가 없다. 이때는
    # 투자설명서 자체에 명시된 클래스 명명 관례로 보충한다(예: 종류S-P
    # "수수료미징구-온라인슈퍼-개인연금", 종류C-Pe "수수료미징구-온라인-개인연금",
    # 종류C-P "수수료미징구-오프라인-개인연금").
    #   S로 시작   → 온라인슈퍼
    #   e로 끝남   → 온라인
    #   연금 클래스(P·P2로 끝남)의 무표기 → 오프라인
    # 연금 클래스가 아닌 무표기 코드(C1·W·F 등)는 채널 개념이 달라 건드리지 않는다.
    if code:
        c = code.upper().replace("-", "")
        if c.startswith("S"):
            return "온라인슈퍼"
        if c.endswith("E"):
            return "온라인"
        if c.endswith("P") or c.endswith("P2"):
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
    restored_identity = 0     # doc_ids로 신원을 복원한 행 수
    summary_rows = 0          # 요약표 텍스트 레이아웃에서 나온 행 수
    wide_shared = set()       # 4개 이상 문서가 공유하는데 행이 나온 청크 — 눈으로 확인용

    # finalize가 공통 청크의 fund_code·fund_name을 지우므로(설계 의도),
    # 코드→이름 사전은 추적 중인 manifest.csv에서 가져온다. 이게 없으면
    # 쌍둥이 문서 펀드 16종의 보수 행이 전부 NULL 신원으로 남는다.
    manifest = {}
    mpath = Path(a.inp).parent / "manifest.csv"
    if mpath.exists():
        with open(mpath, encoding="utf-8-sig") as mf:
            for mr in csv.DictReader(mf):
                if mr.get("fund_code"):
                    manifest[mr["fund_code"]] = (mr.get("fund_name") or None,
                                                 mr.get("base_date") or None)

    with open(a.inp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r["text"]
            # "총 보수"처럼 셀 안에서 공백이 낀 표기가 흔하다(실측: KR5127420034
            # p.25 내역표가 이 이유로 후보에서 탈락했다). 공백을 지우고 본다 —
            # 후보가 늘기만 하는 방향이라 기존에 잡히던 청크는 그대로 잡힌다.
            tn = re.sub(r"\s+", "", text)
            if "총보수" not in tn:
                continue
            # '클래스'가 없는 청크도 요약표 데이터 행(수수료선취-/수수료미징구-)이나
            # '종류' 앵커 내역표일 수 있다(청크 경계에서 헤더가 앞 청크로 잘림).
            # 다만 무작정 열면 후보가 폭증하니 보수표 고유 표지로만 넓힌다.
            if not any(k in tn for k in ("클래스", "수수료선취", "수수료미징구", "지급비율")):
                continue
            candidate_chunks += 1

            # (라벨, 코드, 요율)만 먼저 뽑고, 신원(fund_code)은 아래에서 한 번에 붙인다.
            parsed = []
            if r.get("content_type") == "table":
                for colmap, data_rows in find_fee_tables(text):
                    for c in data_rows:
                        label = c[colmap["class_col"]]
                        code = parse_class_code(label)
                        if code is None and colmap.get("bare_codes"):
                            # '종류' 내역표의 라벨은 괄호 없는 코드 그 자체다("C-P").
                            lab = re.sub(r"\s+", "", label)
                            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,3}(?:-[A-Za-z0-9가-힣]{1,6})?", lab):
                                code = lab
                        # 코드 없는 행도 버리지 않는다 — "운용전환일부터 해지일까지"
                        # 같은 기간 분할 행이 실데이터다(실측: KR5147430065 0.140%).
                        # '지급'/'시기' 각주 줄은 fee_total이 숫자가 아니라서
                        # 아래 0~10 필터가 알아서 거른다.
                        def get(col_key):
                            idx = colmap[col_key]
                            if idx is None or idx >= len(c):
                                return None
                            return c[idx]
                        parsed.append(dict(
                            label=label, code=code,
                            front_load_text=get("front_load_col"),
                            fee_total=parse_num(get("fee_total_col")),
                            fee_distribution=parse_num(get("distribution_col")),
                            fee_peer_avg=parse_num(get("peer_avg_col")),
                            fee_total_cost=parse_num(get("total_cost_col")),
                        ))
            n_table_rows = len(parsed)
            for sr in find_summary_fee_rows(text):
                # 채널·계좌유형은 own(자기 행 조각)에서 먼저 찾는다 — 위쪽 조각은
                # 오버랩 청크에서 앞 행의 것일 수 있다(find_summary_fee_rows 주석 참조).
                own = sr.pop("own_label", "") or sr["label"]
                sr["_ch"] = infer_channel(own, sr["code"]) or infer_channel(sr["label"], sr["code"])
                sr["_at"] = infer_account_type(own, sr["code"]) or infer_account_type(sr["label"], sr["code"])
                parsed.append(sr)
            summary_rows += len(parsed) - n_table_rows

            if not parsed:
                skipped.append((r.get("chunk_id"), r.get("fund_code"), r.get("fund_name")))
                continue

            # ── 신원 복원: 공통 청크(fund_code=None)는 doc_ids의 코드마다 행을 만든다.
            # 같은 표가 두 문서에 실렸다는 뜻이고 실제로 두 펀드의 보수가 동일하다.
            if r.get("fund_code"):
                idents = [(r.get("fund_code"), r.get("fund_name"), r.get("base_date"))]
            else:
                kr = [d for d in (r.get("doc_ids") or [])
                      if isinstance(d, str) and d.startswith("KR")]
                idents = [(d,
                           manifest.get(d, (None, None))[0],
                           r.get("base_date") or manifest.get(d, (None, None))[1])
                          for d in kr]
                if len(kr) >= 4:
                    wide_shared.add((r.get("chunk_id"), len(kr)))
                if not idents:
                    skipped.append((r.get("chunk_id"), None, None))
                    continue

            for pr in parsed:
                # 총보수는 항상 %(0~10 사이)다. 표 열이 잘못 잡혀 "1,000만원 투자시
                # 비용"(천원 단위, 수십~수백) 같은 다른 컬럼 값이 들어온 경우를 걸러낸다.
                if pr["fee_total"] is None or not (0 <= pr["fee_total"] <= 10):
                    continue
                for fc, fn, bd in idents:
                    rows_out.append(dict(
                        fund_code=fc, fund_name=fn, base_date=bd,
                        class_label=pr["label"], class_code=pr["code"],
                        account_type=pr.get("_at", infer_account_type(pr["label"], pr["code"])) if "_at" in pr else infer_account_type(pr["label"], pr["code"]),
                        channel=pr.get("_ch") if "_ch" in pr else infer_channel(pr["label"], pr["code"]),
                        front_load_text=pr["front_load_text"],
                        fee_total=pr["fee_total"],
                        fee_distribution=pr["fee_distribution"],
                        fee_peer_avg=pr["fee_peer_avg"],
                        fee_total_cost=pr["fee_total_cost"],
                        chunk_id=r.get("chunk_id"),
                        page=r.get("page"),
                        source_path=r.get("source_path"),
                    ))
                    if fc:
                        seen_funds_with_rows.add(fc)
                    if fc and not r.get("fund_code"):
                        restored_identity += 1

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
    total_funds_candidate = len(manifest) if manifest else 0
    print(f"후보 청크 {candidate_chunks:,}개 중 표 인식 실패 {len(skipped):,}개")
    print(f"수수료 행 {len(rows_out):,}건 추출 → {a.db}")
    print(f"펀드 {len(seen_funds_with_rows):,}/{total_funds_candidate:,}개에서 데이터 확보 (manifest 기준)")
    print(f"  · 요약표 텍스트 레이아웃에서 {summary_rows:,}행, 공통 청크 신원 복원 {restored_identity:,}행")
    if wide_shared:
        print(f"⚠️ 4개 이상 문서가 공유하는 청크에서 행이 나옴 — 법정 공통문구가 아닌지 확인 필요:")
        for cid, n in sorted(wide_shared):
            print(f"   {cid} (문서 {n}개)")
    if conflicts:
        print(f"\n⚠️ 같은 (펀드,클래스)인데 총보수 값이 다른 경우 {len(conflicts)}건 — 확인 필요:")
        for key, totals in conflicts[:10]:
            print(f"   {key}: {totals}")
    print(f"미리보기: {a.csv}")
    print(f"건너뛴 목록: {a.skipped_csv}")


if __name__ == "__main__":
    main()
