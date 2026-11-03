"""README·기술제안서 fee_sql 교정 표의 9종을 하나씩 검증한다.

각 항목은 표에 적힌 'LLM이 쓴 것'을 HCX 응답으로 주입하고 fee_sql 노드
전체를 돌려, 교정 후 조회 결과가 표의 'DB 실제 값'에 닿는지 본다.
HCX는 호출하지 않는다(conftest의 run_fee_sql이 call_hcx를 바꿔 끼운다).
"""
import sqlite3

import agent


def _classes(state):
    return sorted(r["class_code"] for r in state["sql_rows"])


# ① 클래스 표기: 'Ae' / 'A-e' → DB의 'A-E'
def test_class_code_spelling(run_fee_sql):
    for written in ("Ae", "A-e", "a-E"):
        st = run_fee_sql(
            "하나파워e단기채 Ae 클래스 총보수 알려줘",
            "SELECT fund_name, class_code, fee_total FROM fund_fees "
            f"WHERE fund_name LIKE '%하나파워e단기채%' AND class_code = '{written}'")
        assert _classes(st) == ["A-E"], written
        assert "UPPER(REPLACE(class_code,'-',''))" in st["sql_query"]


# ② 판매경로: '온라인직접판매' → '온라인'·'온라인슈퍼', 슈퍼 명시 시 정확일치
def test_channel_literal(run_fee_sql):
    st = run_fee_sql(
        "NH-Amundi 하나로 단기채 온라인 클래스 총보수",
        "SELECT fund_name, class_code, channel, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%하나로%' AND channel = '온라인직접판매'")
    assert {r["channel"] for r in st["sql_rows"]} == {"온라인", "온라인슈퍼"}

    st = run_fee_sql(
        "NH-Amundi 하나로 단기채 온라인슈퍼 클래스 총보수",
        "SELECT fund_name, class_code, channel, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%하나로%' AND channel = '온라인슈퍼'")
    assert _classes(st) == ["S-P2"]


# ③ 펀드명 공백: 'NH-Amundi하나로단기채' → 'NH-Amundi 하나로 단기채 …'
def test_fund_name_spacing(run_fee_sql):
    st = run_fee_sql(
        "NH-Amundi하나로단기채 총보수 알려줘",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%NH-Amundi하나로단기채%'")
    assert st["sql_rows"], st["trace"]
    assert {r["fund_name"] for r in st["sql_rows"]} == {
        "NH-Amundi 하나로 단기채 증권투자신탁[채권]"}
    assert any("LIKE로 완화해 재조회" in t for t in st["trace"])


# ④ 펀드명 태그: '%스팍스한국엄선투자[주식]%' → '…증권자투자신탁[주식]'
def test_fund_name_tag(run_fee_sql):
    st = run_fee_sql(
        "스팍스한국엄선투자[주식] 총보수",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%스팍스한국엄선투자[주식]%'")
    assert [r["fund_name"] for r in st["sql_rows"]] == ["스팍스한국엄선투자증권자투자신탁[주식]"]


def test_fund_name_tag_ambiguous_match_is_rejected(run_fee_sql, fees_db):
    # 태그를 떼면 두 펀드에 걸린다 → 모호한 매치로 보고 채택하지 않는다
    fees_db([("KR0000000099", "스팍스한국엄선투자증권자투자신탁[채권]", "A", None,
              "오프라인", 0.4, 3)])
    st = run_fee_sql(
        "스팍스한국엄선투자[주식] 총보수",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%스팍스한국엄선투자[주식]%'")
    assert st["sql_rows"] == []
    assert st["sql_empty"] is True
    assert any("모호한 매치" in t for t in st["trace"])


# ⑤ 근거 없는 클래스: 질문에 없는 class_code='C-P' → 조건 제거
def test_ungrounded_class_condition(run_fee_sql):
    st = run_fee_sql(
        "하나파워e단기채의 총보수는 몇 %인가요?",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%하나파워e단기채%' AND class_code = 'C-E'")
    assert len(st["sql_rows"]) == 4
    assert any("근거 없음으로 판단해 제거" in t for t in st["trace"])


# ⑥ 연산자 우선순위: 코드가 만든 OR 치환은 항상 괄호로 감싼다
def test_code_generated_or_is_parenthesized(run_fee_sql):
    # channel IN (...)을 OR로 펼칠 때 괄호가 없으면 'A AND B OR C'가 되어
    # 다른 펀드의 온라인 행까지 섞여 들어온다.
    st = run_fee_sql(
        "하나파워e단기채 오프라인과 온라인 클래스 총보수",
        "SELECT fund_name, class_code, channel, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%하나파워e단기채%' "
        "AND channel IN ('오프라인', '온라인직접판매')")
    assert "AND (channel = '오프라인' OR channel LIKE '온라인%')" in st["sql_query"]
    assert {r["fund_name"] for r in st["sql_rows"]} == {"하나파워e단기채증권자투자신탁[채권]"}
    assert len(st["sql_rows"]) == 4

    # 계좌유형 이관도 같은 이유로 괄호를 씌운다
    out = agent._migrate_account_sql(
        "WHERE fund_name LIKE '%x%' AND class_code IN ('C', '개인연금')")
    assert out == ("WHERE fund_name LIKE '%x%' AND "
                   "(class_code IN ('C') OR account_type = '연금저축')")


# ⑦ 집계 오용: SQL의 정렬을 믿지 않고 최저·최고 질문이면 파이썬이 다시 정렬한다
def test_python_resort_for_superlative(run_fee_sql):
    st = run_fee_sql(
        "하나파워e단기채 클래스 중 총보수가 가장 싼 건?",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%하나파워e단기채%'")
    fees = [r["fee_total"] for r in st["sql_rows"]]
    assert fees == sorted(fees)
    assert st["sql_rows"][0]["class_code"] == "A-E"

    st = run_fee_sql(
        "하나파워e단기채 클래스 중 총보수가 가장 비싼 건?",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%하나파워e단기채%'")
    fees = [r["fee_total"] for r in st["sql_rows"]]
    assert fees == sorted(fees, reverse=True)


# ⑧ LIMIT 절단: 비교 쿼리의 최상위 LIMIT을 실행 전에 지운다
def test_limit_removed_for_multi_class_compare(run_fee_sql):
    st = run_fee_sql(
        "삼성코리아중기채권 Ae와 Ce 중 총보수가 낮은 쪽은?",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%코리아중기채권%' AND class_code IN ('Ae','Ce') "
        "ORDER BY fee_total ASC LIMIT 1")
    assert _classes(st) == ["Ae", "Ce"]
    assert "LIMIT 1" not in st["sql_query"]


def test_limit_removed_for_multi_fund_or(run_fee_sql):
    st = run_fee_sql(
        "고배당포커스연금저축 C와 코어밸류연금저축 C 중 더 낮은 쪽을 골라주세요",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE (fund_name LIKE '%고배당포커스연금저축%' AND class_code = 'C') "
        "OR (fund_name LIKE '%코어밸류연금저축%' AND class_code = 'C') "
        "ORDER BY fee_total ASC LIMIT 1")
    assert len({r["fund_name"] for r in st["sql_rows"]}) == 2


# ⑨ 계좌유형 혼입: class_code IN ('개인연금') → account_type = '연금저축'
def test_account_literal_in_class_code(run_fee_sql):
    st = run_fee_sql(
        "하나파워e단기채를 개인연금으로 가입하면 클래스 총보수가 얼마인가요?",
        "SELECT fund_name, class_code, account_type, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%하나파워e단기채%' "
        "AND UPPER(REPLACE(class_code,'-','')) IN ('개인연금')")
    assert _classes(st) == ["C-P"]
    assert "account_type = '연금저축'" in st["sql_query"]


def test_real_class_with_account_word_still_matches(run_fee_sql):
    # 'S-퇴직'은 진짜 클래스다. 부분일치로 이관하면 이 행을 못 찾는다.
    st = run_fee_sql(
        "코어밸류연금저축 S-퇴직 클래스 총보수",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%코어밸류연금저축%' AND class_code = 'S-퇴직'")
    assert _classes(st) == ["S-퇴직"]


def test_zero_rows_is_reported(run_fee_sql):
    st = run_fee_sql(
        "없는펀드이름 총보수",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%없는펀드이름%'")
    assert st["sql_rows"] == []
    assert st["sql_empty"] is True


def test_fee_sql_skips_without_need_sql(fees_db, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("HCX를 부르면 안 된다")
    monkeypatch.setattr(agent, "call_hcx", boom)
    st = {"question": "DC와 DB 차이", "route": {"need_sql": False}, "trace": []}
    assert agent.fee_sql(st) is st
    assert "sql_rows" not in st


def test_fixture_db_matches_production_schema(fees_db):
    conn = sqlite3.connect(agent.FUND_FEES_DB)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fund_fees)")]
    conn.close()
    for c in ("fund_name", "class_code", "account_type", "channel", "fee_total", "page"):
        assert c in cols
