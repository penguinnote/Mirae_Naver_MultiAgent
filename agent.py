#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연금 Agent v0 — 검색 + 답변 생성

지금까지는 검색까지만 있었다. 이 파일이 그 위에 **답변 생성**을 얹는다.
평가셋 기준선(하이브리드 k=5, 근거 충족률 88.9%)이 이 에이전트의 천장이다.
여기서 나오는 정답률이 88.9%보다 낮으면 검색이 아니라 프롬프트 문제다.

구조
----
    route  →  retrieve  →  [fee_sql]  →  compose  →  to_response
   (판단)     (하이브리드)   (Text2SQL)    (HCX-007)    (평가 API 형식)

    fee_sql은 route가 need_sql을 설정할 때만 실행된다.
    수치 비교·정렬·집계 질문이 아니면 건너뛴다.

노드를 함수로 나눴다. v1에서 LangGraph로 옮길 때 그대로 노드가 되도록
입출력을 state 딕셔너리 하나로 통일해 뒀다. 지금 LangGraph를 쓰지 않는 이유는
노드가 3개뿐이라 얻는 것보다 디버깅 비용이 크기 때문이다.

사용법
------
    python agent.py "퇴직연금 중도인출 사유가 뭐야?"
    python agent.py "총보수 싼 연금저축 펀드 알려줘" --show-evidence
    python agent.py --selftest

    # 평가셋으로 전체 채점 (공용 채점기 사용)
    python eval_answers.py --adapter agent:answer_for_eval --name "윤종 v0"

환경변수 (.env)
--------------
    CLOVA_API_KEY            공통
    CLOVA_CHAT_REQUEST_ID    HCX-007 서비스 앱 ID  ← 임베딩·리랭커와 다른 값
    CLOVA_CHAT_URL           비워두면 기본값 사용
    CLOVA_CHAT_PROFILE       body 파라미터 조합. 비워두면 자동 탐색한다.
                             --selftest가 찾아준 값을 넣어두면 탐색을 건너뛴다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# .env를 **임포트 시점에** 읽는다.
# main() 안에서만 읽으면 `--adapter agent:answer_for_eval` 처럼
# 모듈로 임포트될 때 설정이 통째로 비어버린다. 실제로 그 버그를 겪었다.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from search import (  # noqa: E402
    RRF_K, POOL, build_bm25, embed_query, tokenize, matches,
)

CHAT_URL_DEFAULT = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007"

TOP_K = 5            # 평가 결과상 k=5가 최적 (k=20으로 늘려도 +2.8%p, 컨텍스트는 4배)
EVIDENCE_CHARS = 1500  # 청크 하나당 프롬프트에 넣을 최대 길이
DEADLINE = 240       # 주최측 타임아웃 300초 중 60초를 안전 마진으로 남긴다


# ══════════════════════════════════════════════════════════════════════
# HCX-007 클라이언트
#
# ⚠️ 리랭커에서 겪은 교훈: 응답 스키마를 문서만 보고 단정하면 안 된다.
#    실제로 와보니 문서에 없던 필드에 답이 들어 있었다.
#    그래서 여러 경로를 순서대로 훑는 방어적 파서를 쓴다.
# ══════════════════════════════════════════════════════════════════════

# HCX-007은 추론(reasoning) 계열이라 v3 파라미터 규격이 이전 모델과 다르다.
# maxTokens를 그대로 보내면 40001 "Invalid parameter: maxTokens"가 떨어진다.
# 문서를 추측하는 대신 **통하는 조합을 실제로 찾아내는** 방식을 쓴다.
# (리랭커 때 문서만 보고 스키마를 단정했다가 틀린 경험이 있다)
#
# 한 번 찾아내면 _BODY_PROFILE에 고정되어 이후 호출은 곧바로 그 조합을 쓴다.

BODY_PROFILES = [
    # ① 추론 모델 표준: 토큰 상한 이름이 다르고 thinking을 끌 수 있다
    ("maxCompletionTokens+thinking",
     lambda mt, temp: {"maxCompletionTokens": mt,
                       "thinking": {"effort": "none"}}),
    # ② thinking 없이 이름만 바뀐 경우
    ("maxCompletionTokens",
     lambda mt, temp: {"maxCompletionTokens": mt}),
    # ③ 샘플링 파라미터까지 붙는 구형 규격
    ("maxTokens+sampling",
     lambda mt, temp: {"maxTokens": mt, "temperature": temp,
                       "topP": 0.8, "repeatPenalty": 5.0}),
    # ④ 최소 구성 — 다 안 되면 이거라도
    ("minimal", lambda mt, temp: {}),
]

_BODY_PROFILE: str | None = os.environ.get("CLOVA_CHAT_PROFILE") or None


def _chat_config() -> tuple[str, str, str]:
    key = os.environ.get("CLOVA_API_KEY", "").strip().strip('"').strip("'")
    rid = os.environ.get("CLOVA_CHAT_REQUEST_ID", "").strip().strip('"').strip("'")
    url = (os.environ.get("CLOVA_CHAT_URL", "").strip().strip('"').strip("'")
           or CHAT_URL_DEFAULT)
    if not key:
        raise RuntimeError("CLOVA_API_KEY가 없습니다 (.env 확인)")
    if not rid:
        raise RuntimeError("CLOVA_CHAT_REQUEST_ID가 없습니다. "
                           "임베딩·리랭커와 다른 값입니다 — 콘솔 > 서비스 앱 > "
                           "HCX-007 행 '코드 보기'에서 확인하세요.")
    return key, rid, url


def _post(messages, extra, max_tokens, temperature, timeout=120):
    import requests
    key, rid, url = _chat_config()
    headers = {
        "Authorization": f"Bearer {key}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": rid,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {"messages": messages, **extra}
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    return r


def call_hcx(messages: list[dict], max_tokens: int = 1200,
             temperature: float = 0.2, raw: bool = False, verbose: bool = False):
    """통하는 body 조합을 찾아서 호출한다. 찾은 조합은 전역에 고정한다."""
    global _BODY_PROFILE

    order = BODY_PROFILES
    if _BODY_PROFILE:
        order = ([p for p in BODY_PROFILES if p[0] == _BODY_PROFILE]
                 + [p for p in BODY_PROFILES if p[0] != _BODY_PROFILE])

    last = ""
    for name, build in order:
        extra = build(max_tokens, temperature)
        r = _post(messages, extra, max_tokens, temperature)
        if verbose:
            head = json.dumps(extra, ensure_ascii=False)[:70]
            print(f"    [{name:26}] {head:72} → HTTP {r.status_code}")
        if r.status_code == 200:
            if _BODY_PROFILE != name:
                _BODY_PROFILE = name
                if verbose:
                    print(f"    ✅ '{name}' 조합으로 고정합니다.")
            js = r.json()
            return js if raw else parse_hcx(js)
        last = f"HTTP {r.status_code}: {r.text[:250]}"
        # 파라미터 문제가 아니면(인증·한도 등) 다른 조합을 시도해도 소용없다
        if r.status_code != 400 or "arameter" not in r.text:
            break

    raise RuntimeError(
        f"HCX 호출 실패 — 마지막 응답: {last}\n"
        f"시도한 조합: {[n for n, _ in order]}")


def parse_hcx(js: dict) -> str:
    """응답에서 본문 텍스트를 뽑는다. 경로를 여러 개 시도한다."""
    paths = [
        ("result", "message", "content"),
        ("result", "content"),
        ("result", "text"),
        ("message", "content"),
    ]
    for p in paths:
        cur = js
        for kk in p:
            if not isinstance(cur, dict) or kk not in cur:
                cur = None
                break
            cur = cur[kk]
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
        # HCX-007은 추론 모델이라 content가 리스트로 올 수 있다
        if isinstance(cur, list):
            joined = "\n".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in cur)
            if joined.strip():
                return joined.strip()

    raise RuntimeError(
        "HCX 응답에서 본문을 못 찾았습니다. 아래 원문을 보고 parse_hcx()에 "
        f"경로를 추가하세요:\n{json.dumps(js, ensure_ascii=False)[:800]}")


# ══════════════════════════════════════════════════════════════════════
# ① route — 질문을 보고 어디를 뒤질지 정한다 (Disambiguator 축소판)
#
# LLM을 쓰지 않는다. 규칙으로 충분하고, 호출 1회를 아끼면 그만큼
# 데드라인에 여유가 생긴다. 애매하면 필터를 걸지 않는 쪽으로 판단한다
# (잘못 좁히면 정답을 아예 못 보지만, 안 좁히면 순위만 밀릴 뿐이다).
# ══════════════════════════════════════════════════════════════════════

INST_HINTS = [
    "퇴직연금", "irp", "dc", "db", "연금저축", "중도인출", "세액공제", "압류",
    "부담금", "가입자교육", "규약", "실물이전", "계약이전", "디폴트옵션",
    "종합과세", "연금수령", "과세재원", "지연이자", "중간정산", "임원",
]
PROD_HINTS = [
    "펀드", "투자신탁", "총보수", "판매보수", "판매수수료", "위험등급",
    "환매", "기준가", "클래스", "종류c", "설정일", "운용사",
]
FUND_CODE_RE = re.compile(r"\bKR\d{10}[A-Z0-9]?\b", re.I)

# ── SQL 라우팅 신호 ─────────────────────────────────────────────────
# 벡터 검색으로 못 푸는 수치 비교·정렬·집계 질문을 감지한다.
# "총보수 0.5% 이하 펀드 목록" → need_sql=True
# 판단은 규칙으로 한다. LLM 호출 1회를 아끼면 데드라인에 여유가 생긴다.
SQL_COMPARE = ["이하", "이상", "미만", "초과", "넘는", "안 넘는", "안넘는",
               "보다 싼", "보다 낮", "보다 높", "보다 비싼"]
SQL_SORT = ["가장 싼", "가장 비싼", "가장 낮은", "가장 높은", "가장 저렴",
            "싼 순", "낮은 순", "높은 순", "비싼 순", "저렴한 순"]
SQL_AGG = ["평균", "몇 개", "몇개", "총 몇", "얼마나 저렴", "얼마나 비싼",
           "얼마나 차이"]
SQL_FIELD = ["총보수", "수수료", "보수", "판매보수", "판매수수료"]

FUND_FEES_DB = "./dataset/fund_fees.sqlite"


def _needs_sql(question: str) -> bool:
    """수치 필터·정렬·집계가 필요한 질문인지 규칙으로 판별한다."""
    low = question.lower()
    has_field = any(f in low for f in SQL_FIELD)
    if not has_field:
        return False
    has_compare = any(c in low for c in SQL_COMPARE)
    has_sort = any(s in low for s in SQL_SORT)
    has_agg = any(a in low for a in SQL_AGG)
    return has_compare or has_sort or has_agg


def route(state: dict) -> dict:
    q = state["question"]
    low = q.lower()

    inst = sum(1 for h in INST_HINTS if h in low)
    prod = sum(1 for h in PROD_HINTS if h in low)

    doc_type = None
    if inst >= 2 and prod == 0:
        doc_type = "연금문서"        # 제도 질문 → 472청크로 좁힌다 (실측 +5.5%p)
    elif prod >= 2 and inst == 0:
        doc_type = "투자설명서"

    need_sql = _needs_sql(q)

    m = FUND_CODE_RE.search(q)
    state["route"] = {
        "doc_type": doc_type,
        "fund_code": m.group(0).upper() if m else None,
        "need_sql": need_sql,
        "reason": f"제도 키워드 {inst}개 / 상품 키워드 {prod}개"
                  + (", SQL 필요" if need_sql else ""),
    }
    state["trace"].append(
        f"route: {state['route']['reason']} → "
        f"doc_type={doc_type or '전체'}"
        + (f", fund={state['route']['fund_code']}" if m else "")
        + (", need_sql=True" if need_sql else ""))
    return state


# ══════════════════════════════════════════════════════════════════════
# ② retrieve — 하이브리드 검색 (벡터 + BM25 + RRF)
# ══════════════════════════════════════════════════════════════════════

class Retriever:
    """인덱스를 한 번만 로드해서 재사용한다. 요청마다 로드하면 20초씩 날아간다."""

    def __init__(self, db="./dataset/chroma", collection="pension",
                 chunks="./dataset/chunks_final.jsonl",
                 bm25_cache="./dataset/bm25.pkl"):
        # search.py의 embed_query()는 키가 없으면 경고만 찍고 더미 벡터를 쓴다.
        # 그 상태로 진행하면 차원이 안 맞아 Chroma에서 터지는데, 그 예외가
        # 상위에서 삼켜지면 "성능이 낮다"로 잘못 읽힌다. 여기서 먼저 끊는다.
        if not os.environ.get("CLOVA_API_KEY", "").strip():
            raise RuntimeError(
                "CLOVA_API_KEY가 비어 있습니다. .env를 못 읽었을 가능성이 큽니다.\n"
                "  · data_test 폴더에서 실행했는지 확인하세요 (.env가 여기 있습니다)\n"
                "  · python-dotenv가 설치돼 있는지: pip install python-dotenv")
        import chromadb
        self.col = chromadb.PersistentClient(path=db).get_collection(collection)
        self.bm25, self.meta = build_bm25(Path(chunks), Path(bm25_cache))
        self.text_of = {m["chunk_id"]: m["text"] for m in self.meta}
        self.md_of = {m["chunk_id"]: m for m in self.meta}

    def __call__(self, state: dict) -> dict:
        q = state["question"]
        r = state.get("route") or {}
        doc_type, fund = r.get("doc_type"), r.get("fund_code")

        # 폴백으로 재진입할 때 같은 질의를 다시 임베딩하지 않는다.
        # CLOVA 호출 1회 = 데드라인에서 그만큼 손해다.
        if "query_emb" not in state:
            state["query_emb"] = embed_query(q)

        ranks: dict[str, dict[str, int]] = {}
        res = self.col.query(query_embeddings=[state["query_emb"]], n_results=POOL,
                             include=["metadatas"])
        for i, (cid, md) in enumerate(zip(res["ids"][0], res["metadatas"][0])):
            if matches(md, fund, doc_type, None):
                ranks.setdefault(cid, {})["vec"] = i + 1

        scores = self.bm25.get_scores(tokenize(q))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:POOL]
        for rank, i in enumerate(order):
            if scores[i] <= 0:
                break
            m = self.meta[i]
            md = {k: v for k, v in m.items() if k != "text"}
            md["doc_ids"] = "," + ",".join(md.get("doc_ids") or []) + ","
            if matches(md, fund, doc_type, None):
                ranks.setdefault(m["chunk_id"], {})["bm25"] = rank + 1

        fused = sorted(ranks.items(),
                       key=lambda kv: -sum(1.0 / (RRF_K + x) for x in kv[1].values()))

        # 필터를 걸었는데 결과가 빈약하면 필터 없이 다시 (잘못 좁힌 경우 대비)
        if len(fused) < 3 and (doc_type or fund):
            state["trace"].append("retrieve: 필터 결과가 빈약해 전체 검색으로 폴백")
            state["route"] = {**r, "doc_type": None, "fund_code": None}
            return self(state)   # doc_type·fund가 None이 되므로 재귀는 1회로 끝난다

        ev = []
        for cid, src in fused[:TOP_K]:
            m = self.md_of[cid]
            ev.append({
                "chunk_id": cid,
                "text": self.text_of[cid][:EVIDENCE_CHARS],
                "doc_id": m.get("doc_id"),
                "doc_type": m.get("doc_type"),
                "fund_name": m.get("fund_name"),
                "fund_code": m.get("fund_code"),
                "page": m.get("page"),
                "base_date": m.get("base_date"),
                "source_path": m.get("source_path"),
                "found_by": "+".join(src.keys()),
            })
        state["evidence"] = ev
        state["trace"].append(
            f"retrieve: 하이브리드 검색(벡터+BM25, RRF) → 상위 {len(ev)}개 "
            f"[{', '.join(e['chunk_id'] for e in ev)}]")
        return state


# ══════════════════════════════════════════════════════════════════════
# ②-b  fee_sql — 수치 비교·정렬·집계를 SQL로 처리
#
# 벡터 검색은 "총보수가 가장 싼 펀드"를 원리상 못 푼다. 유사도로는
# 0.15%와 0.5%의 크기를 비교할 수 없기 때문이다.
# fund_fees.sqlite에 363행이 들어있고, HCX-007이 자연어→SQL을 생성한다.
#
# 안전장치:
#   · SELECT만 허용 (INSERT/UPDATE/DELETE/DROP 등 차단)
#   · 세미콜론 여러 문장 차단
#   · LIMIT 없으면 자동 추가
#   · 실행 실패 → 에러 피드백 포함해 1회 재시도 → 그래도 실패하면 검색 폴백
# ══════════════════════════════════════════════════════════════════════

TEXT2SQL_PROMPT = """당신은 SQLite SQL 전문가입니다. 사용자 질문을 SQL 쿼리 하나로 변환하십시오.

테이블: fund_fees (연금 펀드 보수·수수료 데이터, 363행)

CREATE TABLE fund_fees (
    fund_code TEXT,        -- 펀드 코드 (예: KR510902511M)
    fund_name TEXT,        -- 펀드명 (예: 미래에셋장기성장포커스증권자투자신탁1호(주식))
    base_date TEXT,        -- 기준일 (예: 2025-01-17)
    class_label TEXT,      -- 클래스명 (예: 수수료미징구-오프라인-개인연금(C-P))
    class_code TEXT,       -- 클래스 코드 (예: C-P, C-P2, S-P, A-e)
    account_type TEXT,     -- 계좌 유형: '연금저축' 또는 '퇴직연금' (빈 문자열 = 일반)
    channel TEXT,          -- 판매 채널: '온라인', '오프라인', '온라인슈퍼'
    front_load_text TEXT,  -- 선취수수료 설명
    fee_total REAL,        -- 총보수(%) ← 연간 총보수 비율
    fee_distribution REAL, -- 판매보수(%)
    fee_peer_avg REAL,     -- 유사펀드 평균 총보수(%)
    fee_total_cost REAL,   -- 총비용(%)
    chunk_id TEXT,         -- 원문 청크 ID
    page INTEGER,          -- 투자설명서 페이지
    source_path TEXT       -- 원문 경로
);

■ 실제 값 예시 (이 값만 사용하십시오):
  account_type: '연금저축', '퇴직연금', '' (빈 문자열은 일반 계좌)
  channel: '온라인', '오프라인', '온라인슈퍼'

■ 유형별 필터 패턴:
  채권형 펀드: fund_name LIKE '%채권%'
  주식형 펀드: fund_name LIKE '%주식%'
  특정 펀드 검색: fund_name LIKE '%키워드%'
  연금저축 전용: account_type = '연금저축'
  퇴직연금 전용: account_type = '퇴직연금'

■ 규칙:
  1. SELECT 문 하나만 출력하십시오 (설명·주석 없이 SQL만)
  2. ORDER BY로 정렬하십시오 (비교·순위 질문일 때)
  3. 적절한 LIMIT을 포함하십시오 (목록이면 20, 최소/최대면 5)
  4. 같은 펀드가 여러 클래스로 나오면 가장 낮은 보수의 클래스를 기준으로 하십시오
  5. 집계(평균, 개수 등)가 필요하면 GROUP BY와 AVG/COUNT를 사용하십시오"""


_DANGEROUS_KW = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM"
    r"|REPLACE|MERGE|TRUNCATE|GRANT|REVOKE|EXEC)\b",
    re.IGNORECASE,
)


def _extract_sql(text: str) -> str:
    """HCX 응답에서 SQL 쿼리를 추출한다.

    HCX-007은 마크다운 코드블록으로 감싸거나 설명을 덧붙일 수 있다.
    """
    text = text.strip()

    # ① 마크다운 코드블록 안의 SQL
    m = re.search(r"```(?:sql)?\s*\n(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # ② SELECT로 시작하는 부분 추출
    m = re.search(r"(SELECT\s+.+)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # ③ 그대로 반환 (validation에서 걸러짐)
    return text


def _validate_sql(sql: str) -> str:
    """SQL 안전 검사. 통과하면 정리된 SQL, 실패하면 ValueError."""
    sql = sql.strip().rstrip(";").strip()

    if not sql.upper().lstrip().startswith("SELECT"):
        raise ValueError(f"SELECT만 허용됩니다: {sql[:60]}")

    if _DANGEROUS_KW.search(sql):
        kw = _DANGEROUS_KW.search(sql).group(1)
        raise ValueError(f"금지된 키워드: {kw}")

    # 세미콜론으로 여러 문장 넣는 것 차단
    if ";" in sql:
        raise ValueError("여러 문장은 허용되지 않습니다")

    # LIMIT이 없으면 자동 추가
    if "LIMIT" not in sql.upper():
        sql += " LIMIT 50"

    return sql


def fee_sql(state: dict) -> dict:
    """자연어 → SQL → 실행. 실패하면 1회 재시도, 그래도 실패하면 검색으로 폴백."""
    import sqlite3

    q = state["question"]
    raw_sql = ""

    messages = [
        {"role": "system", "content": TEXT2SQL_PROMPT},
        {"role": "user", "content": q},
    ]

    for attempt in range(2):
        try:
            t0 = time.monotonic()
            raw_sql = call_hcx(messages, max_tokens=500, temperature=0.0)
            sql = _validate_sql(_extract_sql(raw_sql))

            conn = sqlite3.connect(FUND_FEES_DB)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.close()

            state["sql_query"] = sql
            state["sql_rows"] = rows
            state["trace"].append(
                f"fee_sql: SQL 실행 성공 ({len(rows)}행, "
                f"{time.monotonic()-t0:.1f}초)\n  {sql}")
            return state

        except Exception as e:                               # noqa: BLE001
            if attempt == 0:
                # 에러 메시지를 피드백으로 포함해 1회 재시도
                messages = [
                    {"role": "system", "content": TEXT2SQL_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": raw_sql},
                    {"role": "user",
                     "content": f"위 SQL에서 오류가 발생했습니다: {e}\n"
                                f"수정된 SQL만 출력하십시오."},
                ]
                state["trace"].append(
                    f"fee_sql: 1차 시도 실패 ({type(e).__name__}: {e}), 재시도")
            else:
                state["trace"].append(
                    f"fee_sql: 2차 시도도 실패 ({type(e).__name__}: {e}), "
                    f"검색 결과로 폴백")

    return state


def format_sql_results(state: dict) -> str:
    """SQL 쿼리 결과를 프롬프트에 넣을 텍스트로 포맷한다."""
    rows = state.get("sql_rows")
    if not rows:
        return ""

    sql = state.get("sql_query", "")
    lines = [f"[SQL 조회 결과]  쿼리: {sql}", ""]

    # 헤더
    cols = list(rows[0].keys())
    # 프롬프트에 넣을 때 불필요한 컬럼은 숨긴다
    skip = {"chunk_id", "page", "source_path", "class_code"}
    show = [c for c in cols if c not in skip]

    lines.append(" | ".join(show))
    lines.append(" | ".join("---" for _ in show))

    cap = min(len(rows), 30)       # 프롬프트 길이 제한
    for r in rows[:cap]:
        vals = []
        for c in show:
            v = r.get(c)
            if isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v) if v is not None else "")
        lines.append(" | ".join(vals))

    if len(rows) > cap:
        lines.append(f"… 총 {len(rows)}행 중 {cap}행만 표시")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# ③ compose — HCX-007로 답변 생성
#
# 프롬프트에서 신경 쓴 지점 네 가지. 전부 실제로 관찰한 실패에서 나왔다.
#
#   1) 오늘 날짜를 명시한다.
#      문서에 "2024.10.31 시행 예정"이라고 쓰여 있으면 LLM이 그대로 따라
#      "아직 시행 전"이라 답한다. 팀원 HCX 답변에서 실제로 나온 오류다.
#   2) 근거 밖의 지식을 쓰지 못하게 한다.
#      수수료율을 지어내면 그 순간 서비스가 끝난다.
#   3) 출처를 쪽수까지 표기하게 한다.
#      팀원 결과는 파일명만 있고 page가 전부 null이었다. 평가에서
#      retrieved_context가 채점 대상이므로 이건 점수 차이로 이어진다.
#   4) 되묻지 말고 경우를 나열하게 한다.
#      평가는 단발 GET이라 되물으면 그 문항은 0점이다.
# ══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """당신은 미래에셋증권의 연금 상담 전문가입니다.
제공된 근거 자료만 사용해 정확하게 답변합니다.

[오늘 날짜] {today}
문서에 적힌 시행일·개정일이 오늘보다 과거라면 이미 시행 중인 것입니다.
문서가 "~할 예정입니다"라고 미래형으로 서술했더라도, 그 날짜가 이미 지났다면
현재 시행 중인 제도로 판단해 답하십시오.

[규칙 1 — 아는 만큼은 답하고, 모르는 부분만 모른다고 한다]
판단 순서를 지키십시오.

  ① 근거 자료에 질문과 관련된 내용이 **조금이라도** 있으면 그것으로 답하십시오.
     질문 전체를 다루지 못해도 됩니다. 답할 수 있는 부분을 먼저 답하고,
     "다만 ○○에 대해서는 제공된 자료에서 확인되지 않습니다"라고 덧붙이십시오.
  ② 근거 자료가 질문의 주제 자체를 전혀 다루지 않을 때만 아래로 답하십시오.
     "제공된 자료에서 확인되지 않습니다."

전부 아니면 전무로 판단하지 마십시오. 근거가 있는데도 "확인되지 않습니다"라고
답하면 그것도 틀린 답변입니다. 반대로, 근거에 없는 내용을 그럴듯하게 지어내는 것은
더 위험합니다. 둘 사이에서 ①을 기본으로 삼으십시오.

[규칙 2 — 근거의 결론을 뒤집지 않는다]
근거가 "불가능하다", "제한된다", "금지된다"라고 서술하면 그 결론을 그대로
유지하십시오. 부정적 결론을 긍정으로 바꾸거나 완화하지 마십시오.
예외나 단서가 함께 적혀 있으면 그것도 같이 옮기십시오.
    예) "원칙적으로 이전 불가능. 단, 기관 보유 퇴직금은 이전 가능"

[규칙 3 — 구체적인 수치와 조건은 절대 요약하지 않는다]  ★가장 중요
근거에 수치(금액, 비율, 세율, 기간 등)가 여러 개 나열되어 있거나 조건에 따라 다르게 제시된 경우, 임의로 하나만 선택하거나 뭉뚱그리지 말고 **전부 다** 적으십시오.
  (잘못된 예: "16.5% 과세" / 잘된 예: "소득에 따라 13.2% 또는 16.5% 과세")
아래 항목은 하나도 빠뜨리지 말고 답변에 옮기십시오.
    · 금액·비율·세율 (900만원, 16.5%, 13.2%, 1/2 …)
    · 기한·기간 (60일 이내, 6개월 이상, 6주, 연장일 다음날부터 …)
    · 조건·자격 (무주택자, 만 55세 이상, 계속근로기간 1년 이상 …)
    · 절차·방법 (내점 신청, 근로자대표 동의, 고용노동부 신고 …)
    · 예외·단서 ("단, ~인 경우는 제외" 같은 문구)
요약하느라 이런 항목을 생략하면 답변이 틀린 것으로 간주됩니다.
근거에 두 가지 경우(예: 일반 퇴직연금 vs 과학기술인연금)가 나오면 둘 다 쓰십시오.

[규칙 4 — 출처를 표기한다]
모든 수치와 조건에는 출처를 함께 적으십시오.
    형식: (문서명 N쪽) 또는 (펀드명 투자설명서 N쪽, 기준일 YYYY-MM-DD)

[규칙 5 — 되묻지 않는다]
질문이 모호하면 되묻지 말고 해당하는 경우를 모두 나열해 답하십시오.
    예) "연금저축은 ~, 퇴직연금(IRP)은 ~"

[형식]
- 핵심 답변을 먼저 쓰고, 그다음 조건·예외·근거를 설명합니다.
- 불릿으로 정리하면 읽기 좋습니다.
- 길이 제한은 없습니다. 위 규칙 3을 지키는 데 필요한 만큼 쓰되,
  근거에 없는 배경 설명으로 분량을 늘리지는 마십시오."""


def format_evidence(ev: list[dict]) -> str:
    out = []
    for i, e in enumerate(ev, 1):
        who = e.get("fund_name") or e.get("doc_id") or "공통"
        head = f"[근거 {i}] {who}"
        if e.get("page"):
            head += f" {e['page']}쪽"
        if e.get("base_date"):
            head += f" (기준일 {e['base_date']})"
        out.append(f"{head}\n{e['text']}")
    return "\n\n".join(out)


def compose(state: dict) -> dict:
    ev = state.get("evidence") or []
    sql_text = format_sql_results(state)

    if not ev and not sql_text:
        state["answer"] = "제공된 자료에서 관련 근거를 찾지 못했습니다."
        state["trace"].append("compose: 근거가 없어 생성을 건너뜀")
        return state

    # 근거 블록 조립: SQL 결과가 있으면 앞에 배치한다
    parts = []
    if sql_text:
        parts.append(sql_text)
    if ev:
        parts.append(format_evidence(ev))
    context_block = "\n\n".join(parts)

    # SQL 결과가 있으면 지시를 보강한다
    if sql_text:
        user_suffix = (
            "위 근거를 사용해 답하십시오. SQL 조회 결과의 수치(보수율, 펀드명 등)를 "
            "정확히 인용하고, 근거 자료에서 추가 맥락(투자 전략, 위험 등급 등)을 "
            "보충하십시오. 근거에 나온 수치·기한·조건·절차·예외를 빠뜨리지 마십시오.")
    else:
        user_suffix = (
            "위 근거만 사용해 답하십시오. 근거에 나온 수치·기한·조건·"
            "절차·예외를 빠뜨리지 마십시오. 근거에 관련 내용이 조금이라도 "
            "있으면 그것으로 답하고, 부족한 부분만 확인되지 않는다고 "
            "밝히십시오.")

    messages = [
        {"role": "system",
         "content": SYSTEM_PROMPT.format(today=state.get("today") or date.today())},
        {"role": "user",
         "content": f"[질문]\n{state['question']}\n\n"
                    f"[근거 자료]\n{context_block}\n\n"
                    f"{user_suffix}"},
    ]
    t0 = time.monotonic()
    try:
        state["answer"] = call_hcx(messages, max_tokens=2000)
        state["trace"].append(
            f"compose: HCX-007로 답변 생성 (근거 {len(ev)}개, "
            f"{time.monotonic()-t0:.1f}초)")
    except Exception as e:                                   # noqa: BLE001
        # 생성에 실패해도 빈손으로 돌려보내지 않는다.
        # 부분적인 답이라도 200으로 주는 게 재시도를 소모하는 것보다 낫다.
        state["error"] = str(e)
        state["answer"] = fallback_answer(ev)
        state["trace"].append(f"compose: 생성 실패({type(e).__name__}) → 근거 요약으로 대체")
    return state


def fallback_answer(ev: list[dict]) -> str:
    """LLM이 죽어도 검색 근거는 있다. 그걸 요약해서라도 내보낸다."""
    lines = ["답변 생성에 실패해 검색된 근거를 그대로 제시합니다.\n"]
    for i, e in enumerate(ev[:3], 1):
        who = e.get("fund_name") or e.get("doc_id") or "공통"
        page = f" {e['page']}쪽" if e.get("page") else ""
        lines.append(f"{i}. ({who}{page}) {e['text'][:300]}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# ④ to_response — 평가 API 응답 형식
#
# 5개 필드 전부 문자열이어야 한다. 하나라도 숫자·배열이면 규격 위반이다.
# ══════════════════════════════════════════════════════════════════════

MAX_CONTEXT_CHARS = 9000   # "극단적으로 길면 초과분은 평가에 반영되지 않을 수 있다"


def to_response(state: dict) -> dict:
    ctx_parts = []
    used = 0

    # SQL 결과가 있으면 맨 앞에 넣는다 (평가에서 이 필드가 채점 대상)
    sql_rows = state.get("sql_rows")
    sql_query = state.get("sql_query", "")
    if sql_rows and sql_query:
        sql_block = format_sql_results(state)
        ctx_parts.append(sql_block)
        used += len(sql_block)

    for e in state.get("evidence") or []:
        who = e.get("fund_name") or e.get("doc_id") or "공통"
        head = f"[{e['chunk_id']}] {who}"
        if e.get("page"):
            head += f" · {e['page']}쪽"
        if e.get("base_date"):
            head += f" · 기준일 {e['base_date']}"
        block = f"{head}\n{e['text']}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        ctx_parts.append(block)
        used += len(block)

    return {
        "question_id": str(state.get("question_id") or ""),
        "question": str(state["question"]),
        "retrieved_context": "\n\n---\n\n".join(ctx_parts),
        "think_trace": "\n".join(f"[{i}] {s}"
                                 for i, s in enumerate(state.get("trace") or [], 1)),
        "answer": str(state.get("answer") or ""),
    }


# ══════════════════════════════════════════════════════════════════════
# 오케스트레이션 + 데드라인 가드
# ══════════════════════════════════════════════════════════════════════

_RETRIEVER: Retriever | None = None
_CACHE: dict[str, dict] = {}     # question_id → 응답. 재시도 대비 + 호출 절약


def get_retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


def warmup() -> None:
    """인덱스를 미리 올려둔다.

    BM25 72MB + Chroma 로딩에 20초쯤 걸린다. 이걸 첫 요청 안에서 하면
    그 요청만 데드라인 예산을 20초 까먹는다. FastAPI 기동 시점에
    (@app.on_event("startup")) 이 함수를 부르면 그 손해가 사라진다.
    """
    t0 = time.monotonic()
    get_retriever()
    print(f"인덱스 로딩 완료 ({time.monotonic() - t0:.1f}초)", file=sys.stderr)


def run(question: str, question_id: str = "", use_cache: bool = True) -> dict:
    if use_cache and question_id and question_id in _CACHE:
        return _CACHE[question_id]

    retriever = get_retriever()   # 로딩은 데드라인 측정 전에 끝내둔다

    t0 = time.monotonic()
    state = {"question": question, "question_id": question_id,
             "trace": [], "today": str(date.today())}

    for node in (route, retriever, compose):
        if time.monotonic() - t0 > DEADLINE:
            state["trace"].append(
                f"⏱ {DEADLINE}초 초과 — 남은 단계를 건너뛰고 현재까지로 응답")
            if not state.get("answer"):
                state["answer"] = fallback_answer(state.get("evidence") or [])
            break
        state = node(state)

    state["trace"].append(f"총 소요 {time.monotonic()-t0:.1f}초")
    resp = to_response(state)
    if question_id:
        _CACHE[question_id] = resp
    return resp


def answer_for_eval(question: str) -> dict:
    """eval_answers.py --adapter agent:answer_for_eval 로 쓰는 진입점."""
    return run(question)


# ══════════════════════════════════════════════════════════════════════

def selftest() -> None:
    """HCX 연결과 **통하는 파라미터 조합**을 찾는다. 검색 인덱스는 건드리지 않는다."""
    print("HCX-007 연결 확인 중…", flush=True)
    for k in ("CLOVA_API_KEY", "CLOVA_CHAT_REQUEST_ID"):
        v = os.environ.get(k, "").strip()
        print(f"  {k}: {'설정됨 (len %d)' % len(v) if v else '❌ 비어있음'}")
    url = os.environ.get("CLOVA_CHAT_URL", "").strip() or CHAT_URL_DEFAULT
    print(f"  URL: {url}")

    print("\n파라미터 조합을 순서대로 시도합니다.")
    print("  (HCX-007은 추론 모델이라 이전 모델과 body 규격이 다릅니다)\n")
    t0 = time.monotonic()
    try:
        js = call_hcx([{"role": "user", "content": "한 문장으로 자기소개 해줘."}],
                      max_tokens=512, raw=True, verbose=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"\n❌ 전부 실패했습니다.\n{e}")
        print("\n확인할 것:")
        print("  · CLOVA_CHAT_REQUEST_ID가 HCX-007용이 맞는지 "
              "(임베딩·리랭커 것과 다른 값이어야 합니다)")
        print("  · 콘솔에서 HCX-007 서비스 앱이 승인되었는지")
        return

    print(f"\n✅ 응답 수신 ({time.monotonic()-t0:.1f}초)")
    print(f"   통한 조합: {_BODY_PROFILE}")
    print("\n=== 응답 원문(앞부분) ===")
    print(json.dumps(js, ensure_ascii=False, indent=2)[:1500])
    print("\n=== 파서 결과 ===")
    try:
        print(f"  {parse_hcx(js)}")
        print("\n✅ 파서가 본문을 정상 추출했습니다. agent.py를 바로 쓸 수 있습니다.")
        print(f"\n   .env에 아래 줄을 넣어두면 매번 탐색하지 않고 바로 씁니다:")
        print(f"     CLOVA_CHAT_PROFILE={_BODY_PROFILE}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  ❌ {e}")
        print("\n   위 원문에서 답변 본문이 어느 필드에 있는지 확인하고")
        print("   parse_hcx()의 paths에 경로를 추가하면 됩니다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="연금 Agent v0")
    ap.add_argument("question", nargs="?", help="질문")
    ap.add_argument("--qid", default="", help="question_id")
    ap.add_argument("--selftest", action="store_true", help="HCX 연결만 확인")
    ap.add_argument("--show-evidence", action="store_true", help="근거 원문 출력")
    ap.add_argument("--json", action="store_true", help="평가 API 응답 형식으로 출력")
    a = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if a.selftest:
        selftest()
        return
    if not a.question:
        ap.error("질문을 입력하거나 --selftest 를 쓰세요.")

    resp = run(a.question, a.qid)

    if a.json:
        print(json.dumps(resp, ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 70)
    print(resp["answer"])
    print("=" * 70)
    print("\n[사고 과정]")
    print(resp["think_trace"])
    if a.show_evidence:
        print("\n[근거]")
        print(resp["retrieved_context"][:4000])
    else:
        n = resp["retrieved_context"].count("---") + 1 if resp["retrieved_context"] else 0
        print(f"\n[근거] {n}개 · {len(resp['retrieved_context']):,}자 "
              f"(--show-evidence 로 전문 확인)")


if __name__ == "__main__":
    main()
