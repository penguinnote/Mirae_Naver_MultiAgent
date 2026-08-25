#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연금 Agent v0 — 검색 + 답변 생성

지금까지는 검색까지만 있었다. 이 파일이 그 위에 **답변 생성**을 얹는다.
평가셋 기준선(하이브리드 k=5, 근거 충족률 88.9%)이 이 에이전트의 천장이다.
여기서 나오는 정답률이 88.9%보다 낮으면 검색이 아니라 프롬프트 문제다.

구조
----
    route  →  retrieve  →  [fee_sql]  →  [calc]  →  compose  →  to_response
   (판단)     (하이브리드)   (Text2SQL)   (순수계산)   (HCX-007)    (평가 API 형식)

    fee_sql은 route가 need_sql을 설정할 때만, calc는 need_calc를 설정할 때만
    실행된다. 둘 다 조건에 안 걸리면 건너뛴다.
    calc는 LLM을 쓰지 않는다 — 연금수령한도 계산에서 HCX-007이 8배 틀린 값을
    낸 적이 있어서(×120%를 ×(11-연차)로 잘못 적용), 순수 파이썬으로 계산해
    "이 값을 그대로 쓰라"고 넘긴다.

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
from itertools import zip_longest
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

# 2026-08-25: 5 → 8.
#
# 예전 평가셋에서는 k=5가 최적이었지만(k=20으로 늘려도 +2.8%p), 56문항 blind에서
# 놓친 항목을 원인별로 나눠보니 **46%가 "문서는 갖고 있는데 상위 5개에 못 올린 것"**
# 이었다. 데이터도 모델도 아닌 순위 문제다.
#
# 결정적인 사례: 문서 3개(45청크)를 추가한 것만으로 BM25 통계가 미세하게 바뀌어,
# 정답이 든 청크가 1위에서 5위 밖으로 밀려 IB1-95267889가 100% → 0%가 됐다.
# 5칸짜리 창이 너무 좁아 조금만 흔들려도 정답이 빠진다.
#
# 컨텍스트 여유는 충분하다 — 실측 평균 4,194자로 MAX_CONTEXT_CHARS(9,000)의
# 절반이고 상한에 걸린 문항이 0개였다. k=8이면 평균 6,700자로 여전히 여유가 있다.
TOP_K = 8
EVIDENCE_CHARS = 1500  # 청크 하나당 프롬프트에 넣을 최대 길이
DEADLINE = 240       # 주최측 타임아웃 300초 중 60초를 안전 마진으로 남긴다

# HCX 호출 재시도 정책
#
# 2026-08-24 실측으로 알아낸 것:
#   · 정상 응답은 1~25초.
#   · 그런데 계정에 호출 빈도 제한이 걸려 있고, 초과분에 429를 주는 게 아니라
#     **응답을 아예 안 준다**(서버 큐에 물림). 간격 0/15/30초는 전부 무응답,
#     60초를 쉬면 1.9초에 성공.
#   · 클라이언트가 타임아웃으로 포기해도 서버 쪽 자리는 한동안 물려 있다.
#     그래서 한 번 막히면 이후 요청이 연쇄적으로 무너진다.
#
# 여기서 나오는 설계 원칙 두 가지:
#   1) 함부로 포기하지 않는다. 타임아웃을 짧게 잡고 버리면 그 요청은 서버에서
#      계속 돌면서 자리만 먹고, 우리는 새 요청을 얹어 상황을 악화시킨다.
#   2) 재시도는 드물게, 충분히 쉬고. 실패 직후 곧바로 다시 거는 것은
#      제한을 한 번 더 때리는 행동이라 거의 확실히 또 실패한다.
HCX_CALL_TIMEOUT = 110   # 넉넉히 기다린다. 느린 생성을 중도 포기하지 않기 위해서다
HCX_MAX_ATTEMPTS = 2     # 재시도는 한 번뿐
HCX_RETRY_BACKOFF = 20   # 재시도 전 쉬는 시간
HCX_MIN_BUDGET = 40      # 남은 예산이 이보다 적으면 재시도하지 않는다
HCX_MIN_INTERVAL = 2.0   # 연속 호출 사이 최소 간격(초)

# 연속 타임아웃이 이만큼 쌓이면 재시도를 끈다.
# 제한에 걸린 상태에서 재시도는 상황을 악화시키기만 한다.
HCX_BREAKER_THRESHOLD = 2
_consecutive_timeouts = 0
_last_call_at = 0.0


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
             temperature: float = 0.2, raw: bool = False, verbose: bool = False,
             timeout: int = 120):
    """통하는 body 조합을 찾아서 호출한다. 찾은 조합은 전역에 고정한다."""
    global _BODY_PROFILE

    order = BODY_PROFILES
    if _BODY_PROFILE:
        order = ([p for p in BODY_PROFILES if p[0] == _BODY_PROFILE]
                 + [p for p in BODY_PROFILES if p[0] != _BODY_PROFILE])

    last = ""
    for name, build in order:
        extra = build(max_tokens, temperature)
        r = _post(messages, extra, max_tokens, temperature, timeout=timeout)
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


class HCXEmptyContent(RuntimeError):
    """HTTP 200인데 답변 본문이 비어 있다.

    추론 모델이 maxCompletionTokens를 사고 과정에 다 써버린 경우.
    예산을 늘려 다시 부르면 풀리므로 일반 실패와 구분한다.
    """


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

    # 여기까지 왔으면 본문이 비었다는 뜻이다.
    # HCX-007은 추론 모델이라 maxCompletionTokens를 **사고(thinking)와 답변이
    # 나눠 쓴다**. 사고가 예산을 다 먹으면 200이 오면서도 content=""가 된다.
    # 실측: maxCompletionTokens=50 → content="" + thinkingContent 가득.
    #
    # 이건 "응답 스키마를 모르는" 상황과 전혀 다르다. 예산만 늘리면 풀리므로
    # 별도 예외로 구분해서 호출부가 재시도할 수 있게 한다.
    msg = (js.get("result") or {})
    if isinstance(msg, dict):
        inner = msg.get("message")
        if isinstance(inner, dict) and inner.get("thinkingContent"):
            think_len = len(str(inner.get("thinkingContent")))
            raise HCXEmptyContent(
                f"답변 본문이 비었습니다 — 토큰을 사고에 다 썼습니다 "
                f"(thinkingContent {think_len}자). maxCompletionTokens를 늘리거나 "
                f"thinking을 끄면 해결됩니다.")

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
               "보다 싼", "보다 낮", "보다 높", "보다 비싼",
               # "A랑 B 중에 어느 쪽이 더 싼가" 형태도 결국 수치 비교다
               "더 싼", "더 저렴", "더 낮", "더 높", "더 비싼", "어느 쪽이",
               # ── 2026-08-25 추가 ──────────────────────────────────
               # 56문항 blind 감사 결과, 수수료 필드를 언급한 15문항 중 11개가
               # SQL을 못 켜고 있었다. 비교를 요청하는 가장 흔한 한국어 표현인
               # "비교해 주세요", "낮은 쪽", "어떻게 다른가요"가 통째로 빠져 있었다.
               "비교", "차이", "대비",
               "낮은", "높은", "싼", "저렴",       # "총보수가 낮은 쪽은?"
               "다른가", "다릅", "다른지",          # "어떻게 다른가요?"
               "어느", "중 어", "골라", "고르"]
SQL_SORT = ["가장 싼", "가장 비싼", "가장 낮은", "가장 높은", "가장 저렴",
            "싼 순", "낮은 순", "높은 순", "비싼 순", "저렴한 순",
            "제일 싼", "제일 낮은", "제일 저렴", "최저", "최고"]
# fee_sql()이 SQL 결과를 재정렬할 때 방향을 정하는 데 쓴다 (최고 제외한 오름차순/내림차순)
SQL_SORT_ASC = ["가장 싼", "가장 낮은", "가장 저렴", "싼 순", "낮은 순", "저렴한 순",
                "제일 싼", "제일 낮은", "제일 저렴", "최저"]
SQL_SORT_DESC = ["가장 비싼", "가장 높은", "비싼 순", "높은 순", "제일 비싼", "최고"]
SQL_AGG = ["평균", "몇 개", "몇개", "총 몇", "얼마나 저렴", "얼마나 비싼",
           "얼마나 차이"]
SQL_FIELD = ["총보수", "수수료", "보수", "판매보수", "판매수수료"]

FUND_FEES_DB = str(Path(__file__).resolve().parent / "dataset" / "fund_fees.sqlite")


# 클래스 코드 표기를 잡는다: C-P, A-E, A-e, R-A, C-P2e, S, Ae, Ce …
# 앞뒤가 한글/숫자/영문이면 매치하지 않는다(단어 중간의 우연한 문자 제외).
_CLASS_CODE_RE = re.compile(
    r"(?<![0-9A-Za-z])"
    r"(?:[ABCRSJ][0-9]?(?:-[A-Za-z][0-9]?[a-z]?)?[a-z]?)"
    r"(?![0-9A-Za-z])"
)


def _needs_sql(question: str) -> bool:
    """수치 필터·정렬·집계가 필요한 질문인지 규칙으로 판별한다.

    수수료 필드를 언급하면서 아래 중 하나라도 해당하면 켠다:
      · 비교/정렬/집계 표현
      · 클래스 코드(C-P, A-E …)나 '클래스'라는 말

    마지막 조건을 넣은 이유: "삼성ABF R-A의 총보수는 얼마인가요?" 같은
    단일 사실 조회도 DB에 정확한 값이 있는데(0.4%), 검색으로만 답하면
    투자설명서 표를 잘못 읽어 틀린다. 실측으로 확인된 실패 패턴이다.
    오탐이 나도 fee_sql은 0행이면 그 사실을 알리고 검색으로 폴백하므로
    손해가 작다.
    """
    low = question.lower()
    has_field = any(f in low for f in SQL_FIELD)
    if not has_field:
        return False
    has_compare = any(c in low for c in SQL_COMPARE)
    has_sort = any(s in low for s in SQL_SORT)
    has_agg = any(a in low for a in SQL_AGG)
    has_class = "클래스" in question or bool(_CLASS_CODE_RE.search(question))
    return has_compare or has_sort or has_agg or has_class


# ── 계산 라우팅 신호 ─────────────────────────────────────────────────
# 연금수령한도처럼 정해진 공식이 있는 산수는 LLM에게 시키지 않는다.
# evalset Q-030에서 HCX-007이 ×120%를 ×(11-연차)로 잘못 적용해
# 8배 틀린 값을 낸 전례가 재현됐다(round1·round3 두 번 다 동일 오류).
# 좁게만 켠다 — 미탐(계산 안 켜짐)은 그냥 검색 답변으로 대체되지만,
# 오탐(엉뚱한 계산을 사실처럼 제시)은 훨씬 위험하다.
_CALC_AMOUNT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:조|억|천만|백만|만)\s*원?")
_CALC_AGE_OR_CAR_RE = re.compile(r"(?:만\s*)?\d{2}\s*세|\d+\s*년\s*차|\d+\s*년째")
_CALC_TOPIC_WORDS = ["연금수령한도", "얼마까지", "얼마나 받을 수", "수령할 수 있는",
                      "받을 수 있나요", "받을 수 있는지"]


def _needs_calc(question: str) -> bool:
    """연금수령한도 계산이 필요한 질문인지 규칙으로 판별한다.

    '평가액' 언급 + 금액 표현 + 나이/연차가 전부 있어야 켜진다. 셋 중
    하나라도 빠지면 계산하지 않는다(안전한 미탐 우선).
    """
    if "평가액" not in question:
        return False
    if not _CALC_AMOUNT_RE.search(question):
        return False
    if not _CALC_AGE_OR_CAR_RE.search(question):
        return False
    return "연금수령한도" in question or any(w in question for w in _CALC_TOPIC_WORDS)


# ══════════════════════════════════════════════════════════════════════
# ⓪ safety_check — 개인정보·프롬프트 조작 시도를 HCX 호출 전에 코드로 차단
#
# 평가지표 "안전성 및 신뢰성"(개인정보 노출, 부적절한 입출력, 프롬프트 공격)에
# 대응한다. LLM에게 판단을 맡기지 않는다 — 판단을 맡기면 인젝션 문구 자체가
# 프롬프트에 들어가 우회당할 수 있다. 정규식으로 먼저 걸러 표준 거절 응답을
# 주고 그 자리에서 끝낸다(다른 노드를 타지 않는다).
# ══════════════════════════════════════════════════════════════════════

_PII_PATTERNS = [
    re.compile(r"\d{6}[-\s]?[1-4]\d{6}"),          # 주민등록번호 형식
    re.compile(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"),  # 카드번호 형식
]
_INJECTION_MARKERS = [
    "이전 지시", "이전 지침", "시스템 프롬프트", "system prompt",
    "지시사항을 무시", "규칙을 무시", "규칙을 잊", "ignore previous",
    "ignore all previous", "지금부터 너는", "역할을 무시", "jailbreak",
    "너는 이제부터", "프롬프트를 출력", "지시문을 출력", "내부 지침",
]


def _looks_like_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


def _looks_like_injection(text: str) -> bool:
    low = text.lower()
    return any(m.lower() in low for m in _INJECTION_MARKERS)


def safety_check(state: dict) -> dict:
    q = state["question"]
    if _looks_like_pii(q):
        state["answer"] = (
            "죄송합니다. 주민등록번호·카드번호로 보이는 개인정보가 포함되어 있어 "
            "답변드릴 수 없습니다. 개인정보를 빼고 제도나 상품에 대한 질문으로 "
            "다시 문의해 주세요.")
        state["trace"].append("safety_check: 개인정보 패턴 감지 → 표준 거절 응답, 이후 단계 건너뜀")
        state["_blocked"] = True
    elif _looks_like_injection(q):
        state["answer"] = (
            "죄송합니다. 해당 요청에는 답변드릴 수 없습니다. 연금 제도나 상품에 "
            "대해 궁금한 점을 질문해 주세요.")
        state["trace"].append("safety_check: 프롬프트 조작 시도 패턴 감지 → 표준 거절 응답, 이후 단계 건너뜀")
        state["_blocked"] = True
    return state


# ══════════════════════════════════════════════════════════════════════
# ① route — 질문을 보고 어디를 뒤질지 정한다 (Disambiguator 축소판)
# ══════════════════════════════════════════════════════════════════════

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
    need_calc = _needs_calc(q)

    # 제도와 상품을 **둘 다** 묻는 복합 질문인지 표시한다.
    #
    # 실측 문제: IB1-10461FA3("실물이전 사전조회는 어디서 + A-e 총보수는?")에서
    # 검색 상위 8개가 전부 펀드 투자설명서로 채워져 제도 문서(doc35)가 하나도
    # 안 들어왔다. 질문에 펀드명이 들어가면 벡터·BM25 양쪽에서 상품 문서가
    # 점수를 독식하기 때문이다. 그 결과 제도 파트를 통째로 못 답했다.
    # 복합 도메인이 56문항 중 가장 낮은 49.1%인 이유이기도 하다.
    #
    # → 한쪽이 독식하지 못하게 제도·상품을 나눠 검색하고 합친다.
    hybrid = inst >= 1 and prod >= 1 and doc_type is None

    m = FUND_CODE_RE.search(q)
    state["route"] = {
        "doc_type": doc_type,
        "hybrid": hybrid,
        "fund_code": m.group(0).upper() if m else None,
        "need_sql": need_sql,
        "need_calc": need_calc,
        "reason": f"제도 키워드 {inst}개 / 상품 키워드 {prod}개"
                  + (", 복합" if hybrid else "")
                  + (", SQL 필요" if need_sql else "")
                  + (", 계산 필요" if need_calc else ""),
    }
    state["trace"].append(
        f"route: {state['route']['reason']} → "
        f"doc_type={doc_type or '전체'}"
        + (f", fund={state['route']['fund_code']}" if m else "")
        + (", need_sql=True" if need_sql else "")
        + (", need_calc=True" if need_calc else ""))
    return state


# ══════════════════════════════════════════════════════════════════════
# ② retrieve — 하이브리드 검색 (벡터 + BM25 + RRF)
# ══════════════════════════════════════════════════════════════════════

# ── 같은 계열 펀드(단기/중장기/장기 등) 이름에서 공통 뿌리를 뽑는다 ─────────
# "미래에셋솔로몬단기국공채증권자투자신탁1호(채권)"과 "…중장기…", "…장기…"는
# 계열명("솔로몬국공채")이 같고 기간 수식어만 다르다. 이 뿌리가 같은 펀드가
# 2개 이상이고 질문에 그 뿌리가 등장하면 "비교 질문"으로 본다.
_FUND_VARIANT_WORDS = ["초단기", "중단기", "중장기", "초장기", "단기", "장기"]
_FUND_SUFFIX_WORDS = ["증권자투자신탁", "증권투자신탁", "자투자신탁", "투자신탁", "증권"]


def _fund_core(name: str) -> str:
    """펀드명에서 기간 수식어·괄호·호수·상품유형 접미사를 제거해 계열 뿌리만 남긴다."""
    core = re.sub(r"\(.*?\)", "", name or "")           # (채권)/(주식) 등 괄호 제거
    core = re.sub(r"제?\d+호", "", core)                  # "1호"/"제1호" 제거
    for w in _FUND_SUFFIX_WORDS:
        core = core.replace(w, "")
    for w in _FUND_VARIANT_WORDS:                        # 긴 것부터 — "중장기"가
        core = core.replace(w, "")                        # "단기"보다 먼저 지워져야 한다
    return core.strip()


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

        # 펀드 비교 질문 감지용 색인 — fund_code별 청크 인덱스, 계열 뿌리별 그룹
        self.chunks_by_fund: dict = {}
        fund_names: dict = {}
        for i, m in enumerate(self.meta):
            fc = m.get("fund_code")
            if not fc:
                continue
            self.chunks_by_fund.setdefault(fc, []).append(i)
            fund_names.setdefault(fc, m.get("fund_name"))
        self.fund_core_index: dict = {}
        for fc, name in fund_names.items():
            core = _fund_core(name).replace("미래에셋", "")
            if len(core) >= 2:
                self.fund_core_index.setdefault(core, []).append((fc, name))

    def _detect_compare_funds(self, question: str):
        """질문에 같은 계열(뿌리)의 펀드가 2개 이상 언급됐는지 찾는다.

        예: "솔로몬 국공채 단기·중장기·장기, 뭐가 달라요?" → 뿌리 "솔로몬국공채"
        아래 3개 fund_code가 전부 걸린다. 하나도 안 걸리면 None.
        """
        norm = re.sub(r"[\s·,/]", "", question)
        best = None
        for core, funds in self.fund_core_index.items():
            if len(funds) < 2:
                continue
            if core and core in norm:
                if best is None or len(funds) > len(best):
                    best = funds
        return best

    def _fund_evidence(self, query_emb, question: str, fund_code: str, k: int) -> list:
        """한 펀드로 범위를 좁혀 벡터+BM25를 각각 돌리고 합친다.

        전체 풀(POOL=60)로 한 번에 검색하면 계열 펀드들의 청크가 서로 거의
        동일해서(법정 문구 공유) 상위 슬롯을 한두 펀드가 다 차지하고 나머지는
        通 밀려난다. 펀드별로 따로 뽑아야 전부 근거에 들어간다.
        """
        idxs = self.chunks_by_fund.get(fund_code, [])
        if not idxs:
            return []

        picked: dict[str, dict] = {}
        try:
            res = self.col.query(query_embeddings=[query_emb], n_results=k,
                                  where={"fund_code": fund_code},
                                  include=["metadatas"])
            for i, cid in enumerate(res["ids"][0]):
                picked.setdefault(cid, {})["vec"] = i + 1
        except Exception:                                  # noqa: BLE001
            pass  # Chroma where 필터가 안 먹으면 BM25만으로도 충분하다

        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(idxs, key=lambda i: -scores[i])[:k]
        for rank, i in enumerate(ranked):
            cid = self.meta[i]["chunk_id"]
            picked.setdefault(cid, {})["bm25"] = rank + 1

        fused = sorted(picked.items(),
                       key=lambda kv: -sum(1.0 / (RRF_K + x) for x in kv[1].values()))
        out = []
        for cid, src in fused[:k]:
            m = self.md_of[cid]
            out.append({
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
        return out

    def _fuse(self, q: str, query_emb, fund, doc_type):
        """벡터 + BM25 결과를 RRF로 합쳐 순위를 매긴다.

        doc_type을 바꿔가며 여러 번 부를 수 있도록 __call__에서 떼어냈다.
        임베딩은 인자로 받으므로 몇 번을 불러도 CLOVA 호출은 늘지 않는다.
        """
        ranks: dict[str, dict[str, int]] = {}

        res = self.col.query(query_embeddings=[query_emb], n_results=POOL,
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

        return sorted(ranks.items(),
                      key=lambda kv: -sum(1.0 / (RRF_K + x) for x in kv[1].values()))

    def __call__(self, state: dict) -> dict:
        q = state["question"]
        r = state.get("route") or {}
        doc_type, fund = r.get("doc_type"), r.get("fund_code")

        # 폴백으로 재진입할 때 같은 질의를 다시 임베딩하지 않는다.
        # CLOVA 호출 1회 = 데드라인에서 그만큼 손해다.
        if "query_emb" not in state:
            state["query_emb"] = embed_query(q)

        # 같은 계열 펀드 비교 질문이면 펀드별로 따로 검색해서 합친다.
        # ("솔로몬 국공채 단기·중장기·장기" 같은 질문에서 한 펀드가 상위를
        #  독식해 나머지가 근거에서 통째로 빠지는 문제를 막는다 — 근거 완전성)
        compare_funds = self._detect_compare_funds(q)
        if compare_funds:
            per_fund_k = max(2, TOP_K // len(compare_funds))
            ev = []
            for fc, fname in compare_funds:
                ev.extend(self._fund_evidence(state["query_emb"], q, fc, per_fund_k))
            state["evidence"] = ev
            names = ", ".join(fn for _, fn in compare_funds)
            state["trace"].append(
                f"retrieve: 계열 비교 질문 감지({len(compare_funds)}개 펀드) → "
                f"펀드별 검색 [{names}] → 상위 {len(ev)}개")
            return state

        fused = self._fuse(q, state["query_emb"], fund, doc_type)

        # 복합 질문이면 제도·상품을 따로 검색해 합친다.
        # 한 쪽 문서군이 상위를 독식하는 것을 막는 것이 목적이다.
        if r.get("hybrid"):
            half = max(2, TOP_K // 2)
            inst_hits = self._fuse(q, state["query_emb"], fund, "연금문서")[:half]
            prod_hits = self._fuse(q, state["query_emb"], fund, "투자설명서")[:half]
            merged, seen = [], set()
            # 번갈아 담아 한 쪽이 앞자리를 다 차지하지 않게 한다
            for a, b in zip_longest(inst_hits, prod_hits):
                for item in (a, b):
                    if item and item[0] not in seen:
                        seen.add(item[0])
                        merged.append(item)
            # 어느 한쪽이 비면(예: 제도 문서가 안 잡힘) 전체 검색으로 메운다
            if len(merged) < TOP_K:
                for item in fused:
                    if item[0] not in seen:
                        seen.add(item[0])
                        merged.append(item)
                    if len(merged) >= TOP_K:
                        break
            if inst_hits and prod_hits:
                state["trace"].append(
                    f"retrieve: 복합 질문 → 제도 {len(inst_hits)}개 + "
                    f"상품 {len(prod_hits)}개로 나눠 검색")
                fused = merged

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
    class_label TEXT,      -- 클래스 설명 문자열. 코드가 아니라 긴 문장이다.
                           --   실제 값 예: '수수료미징구-오프라인-개 인연금(C-P)'
                           --   ⚠ 원문 PDF에서 뽑은 값이라 한글 중간에 공백이
                           --     끼어 있는 행이 28%다('개 인연금', '퇴 직연금').
                           --     그래서 class_label로 한글을 매칭하면 자주 빗나간다.
    class_code TEXT,       -- 클래스 코드. 클래스를 지정할 땐 **반드시 이 컬럼**을 쓴다.
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

■ 실제 값 (이 값만 사용하십시오 — 다른 리터럴을 지어내지 마십시오):
  class_code 전체 목록 (클래스 조건은 이 컬럼으로 거십시오):
      A, A-E, A-e, A-u, A-u2, A1, A2, A2e, AG, Ae, B, B-e,
      C, C-E, C-F, C-H, C-I, C-I2, C-P, C-P1, C-P1e, C-P2, C-P2e,
      C-Pe, C-Pe1, C-Pe2, C-R, C-RP, C-RPe, C-Re, C-W, C-e, C-g, C-i,
      C1, C2, C3, C4, CG, Ce, Cf, Cf-RP, Ci-RP, Cw,
      J-Pe, J-RPe, J-e, R, R-A, S, S-P, S-P2, S-RP, Sp
      ※ 대소문자를 구분합니다. 'A-e'와 'A-E'는 다른 클래스입니다.
  account_type: '연금저축'(48행), '퇴직연금'(53행), NULL(262행 — 연금 전용이 아닌 일반 클래스)
      ※ 빈 문자열('')이 아니라 NULL입니다. account_type = '' 는 항상 0행입니다.
  channel: '오프라인'(203행), '온라인'(121행), '온라인슈퍼'(14행), NULL(25행)
  fee_total 범위: 0.075 ~ 3.0
  fee_distribution, fee_peer_avg, fee_total_cost 에는 NULL이 많습니다.
  363행은 "펀드 × 클래스" 조합이고, 실제 펀드 종류는 73개입니다.

■ 유형별 필터 패턴:
  채권형 펀드: fund_name LIKE '%채권%'
  주식형 펀드: fund_name LIKE '%주식%'
  특정 펀드 검색: fund_name LIKE '%키워드%'
  연금저축 전용: account_type = '연금저축'
  퇴직연금 전용: account_type = '퇴직연금'
  연금계좌 전체:  account_type IS NOT NULL
  일반(연금 전용 아님): account_type IS NULL

■ 펀드명 조회 규칙 (실측상 0행이 나오는 원인 1위입니다):
  · fund_name에는 **절대 = 나 IN 을 쓰지 마십시오. 항상 LIKE 입니다.**
        올바름:  WHERE fund_name LIKE '%프리미엄크레딧알파%'
        틀림  :  WHERE fund_name = '미래에셋프리미엄크레딧알파'        ← 0행
        틀림  :  WHERE fund_name IN ('미래에셋프리미엄크레딧알파', '삼성ABF')  ← 0행
    DB의 fund_name은 '미래에셋프리미엄크레딧알파증권자투자신탁(채권)' 처럼
    정식 명칭 전체입니다. 사용자가 부르는 짧은 이름과 절대 같지 않습니다.
  · 펀드를 여러 개 비교할 때는 OR로 묶으십시오. 클래스가 펀드마다 다르면
    괄호로 짝을 지으십시오:
        WHERE (fund_name LIKE '%프리미엄크레딧알파%' AND class_code = 'A-e')
           OR (fund_name LIKE '%코리아중기채권%'   AND class_code = 'Ae')
  · 검색어는 짧고 특징적인 부분만 쓰십시오. 운용사명·'증권자투자신탁'·
    '제1호' 같은 공통어는 빼는 편이 안전합니다.
        좋음: '%클래식연금%'      나쁨: '%삼성클래식연금증권전환형투자신탁 제1호%'
  · 사용자가 클래스를 함께 말했다면 클래스 코드는 fund_name이 아니라
    class_code로 거십시오. '미래에셋프리미엄크레딧알파 A-e'는
    fund_name LIKE '%프리미엄크레딧알파%' AND class_code = 'A-e' 입니다.

■ 클래스 조회 규칙 (가장 자주 틀리는 부분입니다):
  · 클래스 코드로 거를 때는 class_code를 쓰십시오.
        올바름:  WHERE class_code IN ('A-E', 'C-E')
        틀림  :  WHERE class_label IN ('A-E', 'C-E')   ← 항상 0행입니다
    class_label은 '수수료미징구-온라인(C-E)' 같은 긴 설명이라
    코드와 정확히 일치하는 일이 없습니다.
  · 굳이 class_label을 봐야 하면 LIKE로 괄호까지 포함해 거십시오:
        WHERE class_label LIKE '%(C-E)%'
  · '총보수'는 fee_total, '총보수·비용'/'총비용'은 fee_total_cost입니다.
    두 값을 모두 물으면 둘 다 SELECT 하십시오.
  · 환매수수료·선취수수료 조건을 물으면 front_load_text도 SELECT 하십시오.

■ 규칙:
  1. SELECT 문 하나만 출력하십시오 (설명·주석 없이 SQL만)
  2. ORDER BY로 정렬하십시오 (비교·순위 질문일 때)
  3. 적절한 LIMIT을 포함하십시오 (목록이면 20, 최소/최대면 5)
  4. 같은 펀드가 여러 클래스로 나오면 가장 낮은 보수의 클래스를 기준으로 하십시오
  5. 집계(평균, 개수 등)가 필요하면 GROUP BY와 AVG/COUNT를 사용하십시오
  6. NULL 비교에 = 나 != 를 쓰지 마십시오. IS NULL / IS NOT NULL을 쓰십시오
  7. 질문이 "연금 펀드"라고만 하면 account_type으로 좁히지 마십시오.
     '연금저축', '퇴직연금'을 명시했을 때만 필터하십시오
  8. SELECT에 fund_name, class_code, class_label, fee_total은 기본으로
     포함하십시오. 사용자가 결과를 검증할 수 있어야 합니다"""


_DANGEROUS_KW = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM"
    r"|REPLACE|MERGE|TRUNCATE|GRANT|REVOKE|EXEC)\b",
    re.IGNORECASE,
)


_FUND_EQ_RE = re.compile(r"fund_name\s*=\s*'([^']*)'", re.I)
_FUND_IN_RE = re.compile(r"fund_name\s+IN\s*\(([^)]*)\)", re.I)


def _fund_like(literal: str) -> str:
    """펀드명 리터럴 하나를 LIKE 패턴으로 바꾼다.

    '미래에셋프리미엄크레딧알파 A-e' 처럼 클래스 코드가 뒤에 붙어 있으면
    떼어낸다. 클래스는 class_code로 걸러야 하고, 못 걸러도 몇 행 더 나오는
    편이 0행보다 낫다.
    """
    s = literal.strip()
    parts = s.split()
    if len(parts) > 1 and _CLASS_CODE_RE.fullmatch(parts[-1]):
        s = " ".join(parts[:-1])
    return f"fund_name LIKE '%{s.strip()}%'"


def _relax_fund_name(sql: str) -> str | None:
    """fund_name의 완전일치(= / IN)를 LIKE로 바꾼다. 바꿀 게 없으면 None.

    실측 근거: 56문항 실행에서 fee_sql이 0행을 낸 6건이 **전부** 이 패턴이었다.
    DB의 fund_name은 '…증권자투자신탁(채권)' 같은 정식 명칭이라
    사용자가 부르는 짧은 이름과 완전일치할 수가 없다.
    """
    out, changed = sql, False

    def _in_sub(m):
        nonlocal changed
        lits = re.findall(r"'([^']*)'", m.group(1))
        if not lits:
            return m.group(0)
        changed = True
        return "(" + " OR ".join(_fund_like(v) for v in lits) + ")"

    out = _FUND_IN_RE.sub(_in_sub, out)

    def _eq_sub(m):
        nonlocal changed
        changed = True
        return _fund_like(m.group(1))

    out = _FUND_EQ_RE.sub(_eq_sub, out)
    return out if changed else None


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
    """자연어 → SQL → 실행. 실패하면 1회 재시도, 그래도 실패하면 검색으로 폴백.

    route()가 need_sql을 세우지 않았으면 아무것도 하지 않고 통과한다.
    이렇게 해두면 run()의 노드 목록에 조건 분기를 넣지 않아도 되고,
    나중에 LangGraph로 옮길 때 그대로 조건부 엣지가 된다.
    """
    import sqlite3

    if not (state.get("route") or {}).get("need_sql"):
        return state

    if not os.path.exists(FUND_FEES_DB):
        state["trace"].append(
            f"fee_sql: DB를 찾지 못해 건너뜀 ({FUND_FEES_DB}) → 검색 결과로만 답변")
        return state

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

            # HCX가 만든 SQL의 ORDER BY를 믿지 않는다. "가장 싼 게 뭔가요?" 질문에서
            # GROUP BY만 쓰고 ORDER BY를 빠뜨려 LLM이 정렬 안 된 30행을 눈으로 훑다가
            # 최솟값을 놓친 사례가 실제로 있었다(Q-014, fee_total 0.15%인 NH-Amundi를
            # 두고 0.22%짜리를 "가장 싸다"고 답함). SQL을 못 믿으면 파이썬으로 다시
            # 정렬한다 — fee_total 컬럼이 있고 질문에 최저/최고 신호가 있을 때만.
            if rows and "fee_total" in rows[0]:
                low_q = q.lower()
                if any(k in low_q for k in SQL_SORT_ASC):
                    rows = sorted(rows, key=lambda r: (r.get("fee_total") is None,
                                                        r.get("fee_total")))
                    state["trace"].append("fee_sql: 최저값 질문 감지 → fee_total 오름차순 재정렬")
                elif any(k in low_q for k in SQL_SORT_DESC):
                    rows = sorted(rows, key=lambda r: (r.get("fee_total") is None,
                                                        -(r.get("fee_total") or 0)))
                    state["trace"].append("fee_sql: 최고값 질문 감지 → fee_total 내림차순 재정렬")

            # 0행이면 펀드명 완전일치 때문일 공산이 크다. LIKE로 바꿔 한 번 더.
            # LLM을 다시 부르지 않으므로 비용도 지연도 거의 없다.
            if not rows:
                relaxed = _relax_fund_name(sql)
                if relaxed:
                    try:
                        conn = sqlite3.connect(FUND_FEES_DB)
                        conn.row_factory = sqlite3.Row
                        cur = conn.execute(_validate_sql(relaxed))
                        cols = [d[0] for d in cur.description]
                        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                        conn.close()
                        state["trace"].append(
                            f"fee_sql: 0행 → 펀드명 완전일치를 LIKE로 완화해 재조회 "
                            f"({len(rows)}행)\n  {relaxed}")
                        if rows:
                            sql = relaxed
                    except Exception as e:                   # noqa: BLE001
                        state["trace"].append(
                            f"fee_sql: 완화 재조회 실패 ({type(e).__name__})")

            state["sql_query"] = sql
            state["sql_rows"] = rows
            # 0행이면 그 사실을 반드시 compose까지 전달한다.
            # 예전에는 조용히 넘어갔고, 그러면 LLM이 검색 근거의 표 조각에서
            # 숫자처럼 보이는 걸 주워 왔다 — 실제로 총보수를 "BL292"라고
            # 답한 사례가 있다(IB1-502DD2A0). 빈손이면 빈손이라고 말해야 한다.
            state["sql_empty"] = (len(rows) == 0)
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


# 한글 사이('개 인연금')와 구분자 뒤('- 오프라인') 양쪽을 다 잡는다.
_LABEL_SPACE_RE = re.compile(r"(?<=[가-힣\-])\s+(?=[가-힣])")


def _clean_label(v: str) -> str:
    """class_label의 PDF 추출 흔적을 지운다.

    원문 표에서 뽑을 때 한글 사이에 공백이 끼는 행이 28%다
    ('개 인연금', '퇴 직연금', '오 프라인'). 이걸 그대로 프롬프트에 넣으면
    LLM이 답변에도 그 띄어쓰기를 옮겨 적어서, 채점의 '개인연금' 문자열
    매칭이 빗나간다. DB는 건드리지 않고 표시할 때만 정리한다.
    """
    return _LABEL_SPACE_RE.sub("", v)


def format_sql_results(state: dict) -> str:
    """SQL 쿼리 결과를 프롬프트에 넣을 텍스트로 포맷한다."""
    rows = state.get("sql_rows")
    if not rows:
        return ""

    sql = state.get("sql_query", "")
    lines = [f"[SQL 조회 결과]  쿼리: {sql}", ""]

    # 헤더
    cols = list(rows[0].keys())
    # 프롬프트에 넣을 때 불필요한 컬럼은 숨긴다.
    # class_code는 숨기지 않는다 — 답변에서 "C-P 클래스는 …"처럼 클래스를
    # 특정해 말하려면 이 값이 프롬프트에 보여야 한다.
    skip = {"chunk_id", "page", "source_path"}
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
            elif c == "class_label" and isinstance(v, str):
                vals.append(_clean_label(v))
            else:
                vals.append(str(v) if v is not None else "")
        lines.append(" | ".join(vals))

    if len(rows) > cap:
        lines.append(f"… 총 {len(rows)}행 중 {cap}행만 표시")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# ②-c  calc — 연금수령한도는 LLM이 아니라 코드로 계산한다
#
# 공식(doc39 등 "연금수령한도 안내" 문서에서 확인):
#     연금수령한도 = 연금계좌 평가액 / (11 - 연금수령연차) × 120%
#   · 1년차: 분모 10 (한도가 가장 좁다)
#   · 10년차: 분모 1 (평가액의 120%까지)
#   · 11년차 이상: 한도 자체가 없다 — 전액 인출해도 연금수령으로 인정
#
# HCX-007에게 이 공식을 그대로 시키면 종종 ×120%를 ×(11-연차)로 잘못
# 적용해 8배 틀린 값을 낸다(evalset Q-030에서 두 라운드 연속 재현됨).
# 프롬프트로 못 고치는 산수 오류라 파이썬으로 직접 계산해 "이 값을
# 그대로 쓰라"고 근거에 넣어준다.
# ══════════════════════════════════════════════════════════════════════

def _parse_won(text: str) -> "int | None":
    """'1억', '1억 2,000만원', '100,000,000원' 같은 한글 금액 표현을 원 단위로.

    금액 단위를 하나도 못 찾으면 None을 준다 — calc를 켜지 않기 위한 신호다.
    """
    text = text.replace(",", "")
    units = {"조": 1_0000_0000_0000, "억": 1_0000_0000,
             "천만": 1000_0000, "백만": 100_0000,
             "만": 1_0000, "원": 1}
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(조|억|천만|백만|만|원)")
    total, found = 0.0, False
    for m in pattern.finditer(text):
        total += float(m.group(1)) * units[m.group(2)]
        found = True
    return int(round(total)) if found else None


def _extract_year_car(text: str) -> "tuple[int | None, str]":
    """연금수령연차를 뽑는다. 명시가 없으면 나이에서 '올해 최초 개시'로 추정한다.

    실제로 몇 년째 받고 있는지는 질문에 안 나오면 알 방법이 없다. 그 경우
    가정을 계산 결과 텍스트에 명시해 답변에서 드러나게 한다.
    """
    m = re.search(r"(\d+)\s*년\s*차", text)
    if m:
        return int(m.group(1)), "질문에 명시된 연차"
    m = re.search(r"(\d+)\s*년째", text)
    if m:
        return int(m.group(1)), "질문에 명시된 연차('~년째')"
    m = re.search(r"(?:만\s*)?(\d{2})\s*세", text)
    if m:
        age = int(m.group(1))
        if age < 55:
            return 0, f"만 {age}세는 연금수령 개시 연령(55세) 미만"
        return age - 54, f"만 {age}세를 올해 최초 연금개시로 가정 → {age-54}년차"
    return None, "연차·나이 정보를 찾지 못함"


def _format_won(amount: float) -> str:
    amount = int(round(amount))
    eok, rem = divmod(amount, 100_000_000)
    man, won = divmod(rem, 10_000)
    parts = []
    if eok:
        parts.append(f"{eok}억")
    if man:
        parts.append(f"{man:,}만원")
    if not parts or won:
        parts.append(f"{won:,}원" if won or not parts else "")
    return " ".join(p for p in parts if p)


def calc(state: dict) -> dict:
    """연금수령한도를 코드로 직접 계산한다. LLM에게 산수를 시키지 않는다."""
    q = state["question"]
    amount = _parse_won(q)
    year_car, basis = _extract_year_car(q)

    if amount is None or year_car is None:
        state["trace"].append("calc: 평가액 또는 연차/나이를 추출하지 못해 건너뜀")
        return state

    if year_car < 1:
        text = (f"[계산 결과] {basis}이므로 아직 연금수령 요건(만 55세 이상)을 "
                f"충족하지 못한 것으로 보입니다. 연금수령한도 계산 대상이 아닙니다.")
    elif year_car >= 11:
        text = (f"[계산 결과] 연금수령연차 {year_car}년차({basis})는 11년차 이상이므로 "
                f"연금수령한도가 없습니다. 평가액 전액을 인출해도 연금수령으로 "
                f"인정됩니다.")
    else:
        limit = amount / (11 - year_car) * 1.2
        text = (
            f"[계산 결과] 연금수령한도 = 연금계좌 평가액 / (11 - 연금수령연차) × 120%\n"
            f"  평가액 {_format_won(amount)} / (11 - {year_car}) × 120% "
            f"= {_format_won(limit)}\n"
            f"  ※ 연차 산정 근거: {basis}\n"
            f"  ※ 이 값은 코드로 직접 계산했습니다. 답변에서 이 숫자를 그대로 쓰고 "
            f"다시 계산하지 마십시오.")

    state["calc_result"] = text
    state["trace"].append(f"calc: {text.splitlines()[0]}")
    return state


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

[규칙 1-b — "확인되지 않는다"고 쓴 뒤 아는 수치를 덧붙이지 않는다]  ★
근거에서 못 찾았다고 밝힌 다음, "일반적으로 ○○입니다"라며 기억나는 수치를
덧붙이지 마십시오. 제도는 자주 개정되므로 그 수치는 옛날 값일 가능성이 높고,
틀린 수치를 제시하면 "모른다"보다 훨씬 나쁜 답변이 됩니다.

    나쁨: "근거에서 확인되지 않습니다. 일반적으로 합산 한도는 700만원입니다."
          (실제로는 900만원으로 개정됨 → 확실한 오답이 된다)
    좋음: "합산 한도는 제공된 자료에서 확인되지 않습니다."

근거 자료에 수치가 있으면 그 수치가 항상 우선입니다. 기억 속의 값과 다르더라도
근거를 따르십시오.

[규칙 2 — 근거의 결론을 뒤집지 않는다]
근거가 "불가능하다", "제한된다", "금지된다"라고 서술하면 그 결론을 그대로
유지하십시오. 부정적 결론을 긍정으로 바꾸거나 완화하지 마십시오.
예외나 단서가 함께 적혀 있으면 그것도 같이 옮기십시오.
    예) "원칙적으로 이전 불가능. 단, 기관 보유 퇴직금은 이전 가능"

[규칙 2-b — 사용자의 과장·유도성 전제를 그대로 수긍하지 않는다]
질문에 "엄청", "어마어마하게", "무조건", "최고로", "완전" 같은 과장된 표현이나
근거 없는 단정이 섞여 있으면, 근거 자료의 실제 수치·조건과 대조해 사실 여부를
확인하십시오. 사용자의 표현이 과장이거나 부정확하면 그대로 따라 쓰지 말고
"말씀하신 것처럼 크지는 않고, 실제로는 ~"처럼 정확한 사실로 바로잡으십시오.
근거에 있는 수치를 그대로 인용하는 것이 최선의 정정입니다.
    예) 질문 "절세 효과가 어마어마하다던데" → 근거에 "30%~50% 감면"이라고만
        있으면 "감면율은 30~50%로, 전액 비과세는 아닙니다"처럼 과장을 정정

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

[규칙 4 — 출처와 위치를 반드시 표기한다]  ★채점 항목
근거 자료의 각 블록은 이런 첫 줄로 시작합니다.
    [근거 3] doc55 1쪽
    [근거 5] 하나파워e단기채증권자투자신탁[채권] 5쪽 (기준일 2025-05-16)

**"[근거 3]" 같은 번호는 내부 표시일 뿐이므로 답변에 쓰지 마십시오.**
답변을 읽는 사람에게는 아무 의미가 없습니다. 번호 대신 그 뒤에 적힌
**문서명과 쪽수**를 옮겨 적으십시오.

    나쁨: "이는 [근거 3]에서 확인할 수 있습니다."
    좋음: "이는 doc55 1쪽에서 확인할 수 있습니다."
    나쁨: "[근거 5]에 따르면 총보수는 0.25%입니다."
    좋음: "총보수는 0.25%입니다(하나파워e단기채 투자설명서 5쪽)."

  · 인용하는 수치·조건마다 괄호로 출처를 답니다.
  · 첫 줄에 쪽수가 없으면(docx·xlsx·pptx) 문서명과 함께 본문에 보이는
    절·표·슬라이드·항목명을 씁니다.  예) (doc33 슬라이드 5 · Q5)
  · 두 상품을 비교했다면 두 투자설명서의 쪽수를 모두 씁니다.
  · SQL 조회 결과에서 가져온 수치는 어느 펀드·클래스인지 명시합니다.
        (미래에셋프리미엄크레딧알파 C-P, 보수 DB)

없는 쪽수를 지어내지는 마십시오. 근거에 위치가 없으면 문서명만 적습니다.

[규칙 5 — 정보가 부족해도 답변을 포기하지 않는다]
5-a) 질문이 둘 중 하나로 갈리는 정도로 모호하면(예: 어느 계좌인지 안 밝힘)
     되묻지 말고 해당하는 경우를 모두 나열해 답하십시오.
     예) "연금저축은 ~, 퇴직연금(IRP)은 ~"
5-b) 질문이 개인의 조건(투자성향, 계좌유형, 투자기간, 목표 금액 등)을 아예
     제공하지 않아 근거만으로는 하나의 정답을 고를 수 없는 경우(예: "좋은 상품
     추천해주세요"), 다음 두 가지를 **모두** 하십시오.
       ① 정확한 답을 드리려면 어떤 정보(투자성향·계좌유형·투자기간 등)가
          필요한지 먼저 명시하십시오.
       ② 그 다음, 근거 자료에 있는 일반적인 기준(예: 총보수가 낮은 상품,
          안정형이면 채권형)으로 답할 수 있는 만큼 답하십시오.
     이 API는 단발성 요청이라 되물어도 답을 받을 수 없습니다. 정보 부족을
     이유로 답변 자체를 생략하거나 질문만 던지고 끝내지 마십시오.

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
    calc_text = state.get("calc_result") or ""

    if not ev and not sql_text and not calc_text:
        state["answer"] = "제공된 자료에서 관련 근거를 찾지 못했습니다."
        state["trace"].append("compose: 근거가 없어 생성을 건너뜀")
        return state

    # 근거 블록 조립: 계산 결과 → SQL 결과 → 검색 근거 순으로 배치한다.
    # 계산 결과가 가장 확실한 사실이므로 맨 앞에 둔다.
    parts = []
    if calc_text:
        parts.append(calc_text)
    if sql_text:
        parts.append(sql_text)
    if ev:
        parts.append(format_evidence(ev))
    context_block = "\n\n".join(parts)

    # 계산·SQL 결과가 있으면 지시를 보강한다
    if calc_text:
        user_suffix = (
            "위 근거를 사용해 답하십시오. [계산 결과]에 나온 숫자는 코드로 이미 "
            "정확히 계산된 값이니 그대로 인용하고, 직접 다시 계산하거나 다른 "
            "값으로 바꾸지 마십시오. 근거 자료에서 관련 조건·예외도 함께 "
            "설명하십시오.")
    elif sql_text:
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

        # 보수 DB를 조회했는데 해당 행이 없었던 경우.
        # 이 사실을 알리지 않으면 LLM이 투자설명서 표 조각에서 숫자처럼
        # 생긴 걸 주워 온다(실제로 총보수를 "BL292"라고 답한 적이 있다).
        if state.get("sql_empty"):
            user_suffix += (
                "\n\n※ 보수 데이터베이스를 조회했으나 해당 펀드·클래스의 "
                "행이 없었습니다. 검색 근거에 보수율이 명시돼 있지 않다면 "
                "'확인되지 않는다'고 밝히십시오. 표 조각이나 셀 참조처럼 "
                "보이는 문자열을 보수율인 것처럼 제시하지 마십시오.")

    # ── 마지막에 한 번 더 못 박는다 ────────────────────────────────
    # 시스템 프롬프트가 3천 자를 넘다 보니 뒤쪽 규칙이 묻힌다. 실측으로
    # 확인된 두 가지 실패가 그 결과였다.
    #   · 출처 표기: 6문항 중 1개만 쪽수를 적었다. SQL 결과가 있으면
    #     "SQL 조회 결과에 따르면"만 쓰고 문서 출처를 통째로 건너뛴다.
    #   · 사전지식 수치: "근거에 없습니다" 뒤에 기억나는 700만원을 덧붙였다
    #     (실제로는 900만원으로 개정됨 → 모른다고 하느니 못한 오답).
    # 생성 직전에 읽는 자리로 옮기면 지켜질 확률이 올라간다.
    user_suffix += (
        "\n\n[답변 전 최종 확인]\n"
        "1) 인용한 수치·조건마다 출처를 괄호로 답니다. 근거 블록 첫 줄의 "
        "문서명과 쪽수를 그대로 옮기십시오. "
        "예) (doc54 4쪽), (하나파워e단기채 투자설명서 5쪽)\n"
        "   · '[근거 3]' 같은 번호만 쓰지 마십시오. 읽는 사람에게 의미가 "
        "없습니다.\n"
        "   · SQL 결과를 인용할 때도 함께 참고한 문서의 쪽수를 답니다.\n"
        "2) 근거에서 못 찾은 항목은 '확인되지 않습니다'로 끝내십시오. "
        "기억나는 수치를 '일반적으로 ○○입니다'라며 덧붙이지 마십시오 — "
        "제도는 자주 개정되어 그 값이 틀리면 확실한 오답이 됩니다."
    )

    messages = [
        {"role": "system",
         "content": SYSTEM_PROMPT.format(today=state.get("today") or date.today())},
        {"role": "user",
         "content": f"[질문]\n{state['question']}\n\n"
                    f"[근거 자료]\n{context_block}\n\n"
                    f"{user_suffix}"},
    ]
    t0 = time.monotonic()

    # 남은 데드라인 예산 안에서 재시도한다.
    # deadline_at이 없으면(단독 호출 등) DEADLINE 전체를 쓸 수 있다고 본다.
    deadline_at = state.get("_deadline_at") or (t0 + DEADLINE)

    global _consecutive_timeouts, _last_call_at

    last_exc: Exception | None = None
    max_toks = 2000
    # 본문이 비어서(사고에 예산을 다 씀) 다시 부르는 건 빈도 제한과 무관한
    # 실패다. 타임아웃 재시도 예산과 별도로 센다.
    empty_retries = 0
    attempt = 0
    while attempt < HCX_MAX_ATTEMPTS + empty_retries:
        attempt += 1
        remaining = deadline_at - time.monotonic()
        if remaining < HCX_MIN_BUDGET:
            state["trace"].append(
                f"compose: 남은 예산 {remaining:.0f}초 — 재시도 중단")
            break

        # 직전 호출과 너무 붙지 않게 한다. 빈도 제한을 다시 때리지 않기 위해서다.
        gap = time.monotonic() - _last_call_at
        if _last_call_at and gap < HCX_MIN_INTERVAL:
            time.sleep(HCX_MIN_INTERVAL - gap)

        call_timeout = int(min(HCX_CALL_TIMEOUT, remaining - 5))
        _last_call_at = time.monotonic()

        try:
            state["answer"] = call_hcx(messages, max_tokens=max_toks,
                                       timeout=call_timeout)
            _consecutive_timeouts = 0        # 성공했으니 차단기를 푼다
            note = "" if attempt == 1 else f", {attempt}번째 시도"
            state["trace"].append(
                f"compose: HCX-007로 답변 생성 (근거 {len(ev)}개, "
                f"{time.monotonic()-t0:.1f}초{note})")
            state.pop("error", None)
            return state

        except HCXEmptyContent as e:
            # 200은 왔는데 사고에 토큰을 다 썼다. 예산만 늘리면 되는 문제라
            # 타임아웃과 달리 차단기를 건드리지 않고 곧바로 다시 부른다.
            last_exc = e
            if empty_retries >= 2 or max_toks >= 8000:
                state["trace"].append(
                    f"compose: 본문이 계속 비어 있음(maxTokens {max_toks}) — 포기")
                break
            empty_retries += 1
            max_toks = min(max_toks * 2, 8000)
            state["trace"].append(
                f"compose: {attempt}번째 시도 — 본문 비어 있음(사고에 토큰 소진) "
                f"→ maxTokens {max_toks}로 재시도")
            continue

        except Exception as e:                               # noqa: BLE001
            last_exc = e
            timed_out = _is_retryable(e)
            if timed_out:
                _consecutive_timeouts += 1
            state["trace"].append(
                f"compose: {attempt}번째 시도 실패({type(e).__name__}, "
                f"timeout={call_timeout}초, 연속실패 {_consecutive_timeouts})")

            # 인증·파라미터 오류는 다시 걸어도 똑같다.
            if not timed_out:
                break

            # 빈도 제한에 걸린 상태로 보이면 재시도를 포기한다.
            # 여기서 더 던지면 뒤따르는 문항까지 같이 무너진다.
            if _consecutive_timeouts >= HCX_BREAKER_THRESHOLD:
                state["trace"].append(
                    f"compose: 연속 타임아웃 {_consecutive_timeouts}회 "
                    f"— 빈도 제한으로 보고 재시도 중단(차단기 작동)")
                break

            if attempt < HCX_MAX_ATTEMPTS + empty_retries:
                time.sleep(HCX_RETRY_BACKOFF)

    # 여기까지 왔으면 전부 실패했다.
    # 그래도 빈손으로 돌려보내지 않는다 — 검색 근거라도 실어 보낸다.
    state["error"] = str(last_exc) if last_exc else "생성 실패"
    state["answer"] = fallback_answer(ev)
    state["trace"].append(
        f"compose: 생성 실패({type(last_exc).__name__ if last_exc else 'Unknown'}) "
        f"→ 근거 요약으로 대체")
    return state


def _is_retryable(e: Exception) -> bool:
    """다시 걸어볼 만한 실패인지 판단한다.

    타임아웃·연결 끊김·5xx·429는 재시도 가치가 있고,
    인증 실패나 잘못된 파라미터는 몇 번을 걸어도 똑같다.
    """
    name = type(e).__name__
    if name in ("ReadTimeout", "ConnectTimeout", "Timeout",
                "ConnectionError", "ChunkedEncodingError"):
        return True
    msg = str(e)
    if "HTTP 5" in msg or "HTTP 429" in msg:
        return True
    return False


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

    # 계산 결과가 있으면 가장 앞에 넣는다 (평가에서 이 필드가 채점 대상)
    calc_text = state.get("calc_result")
    if calc_text:
        ctx_parts.append(calc_text)
        used += len(calc_text)

    # SQL 결과가 있으면 그다음에 넣는다
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
             "trace": [], "today": str(date.today()),
             "_deadline_at": t0 + DEADLINE}

    state = safety_check(state)
    if state.get("_blocked"):
        state["trace"].append(f"총 소요 {time.monotonic()-t0:.1f}초")
        resp = to_response(state)
        if question_id:
            _CACHE[question_id] = resp
        return resp

    # route를 먼저 실행해야 need_sql을 알 수 있다
    state = route(state)

    # 노드 파이프라인을 동적으로 구성한다.
    # fee_sql은 route가 need_sql을 설정했을 때만 실행된다.
    # 검색(retrieve)은 항상 실행한다 — SQL 결과만으로는 출처 표기가 빈약하다.
    nodes = [retriever]
    if state.get("route", {}).get("need_sql"):
        nodes.append(fee_sql)
    if state.get("route", {}).get("need_calc"):
        nodes.append(calc)
    nodes.append(compose)

    for node in nodes:
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
