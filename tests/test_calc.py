"""calc(연금수령한도)·tax_credit(세액공제)·tax_bracket(연금소득세율 구간) 계산."""
import pytest

import agent


def _state(question, evidence=None):
    return {"question": question, "trace": [], "evidence": evidence or []}


# ── 연금수령한도 ──────────────────────────────────────────────────────

class TestPensionLimit:
    def test_formula(self):
        st = agent.calc(_state("연금계좌 평가액 1억원, 연금수령 3년차면 연금수령한도는 얼마?"))
        # 1억 / (11 - 3) × 120% = 1,500만원
        assert "= 1,500만원" in st["calc_result"]
        assert "질문에 명시된 연차" in st["calc_result"]

    def test_year_from_age_assumes_first_year_at_55(self):
        st = agent.calc(_state("만 57세이고 연금계좌에 2억원이 있으면 올해 연금수령한도는?"))
        # 57세 → 3년차 가정, 2억 / 8 × 1.2 = 3,000만원
        assert "3년차" in st["calc_result"]
        assert "= 3,000만원" in st["calc_result"]

    def test_year_ten_denominator_one(self):
        st = agent.calc(_state("평가액 5천만원, 10년차 연금수령한도"))
        assert "= 6,000만원" in st["calc_result"]

    def test_eleventh_year_has_no_limit(self):
        st = agent.calc(_state("평가액 3억원인데 12년차면 연금수령한도는?"))
        assert "연금수령한도가 없습니다" in st["calc_result"]

    def test_under_55_not_eligible(self):
        st = agent.calc(_state("만 50세, 평가액 1억원이면 연금수령한도는?"))
        assert "충족하지 못한" in st["calc_result"]

    def test_skips_without_amount_or_year(self):
        st = agent.calc(_state("연금수령한도가 뭔가요?"))
        assert "calc_result" not in st

    @pytest.mark.parametrize("text, won", [
        ("1억", 100_000_000),
        ("1억 2,000만원", 120_000_000),
        ("100,000,000원", 100_000_000),
        ("3천만원", 30_000_000),
        ("연금", None),
    ])
    def test_parse_won(self, text, won):
        assert agent._parse_won(text) == won

    @pytest.mark.parametrize("won, text", [
        (15_000_000, "1,500만원"),
        (120_000_000, "1억 2,000만원"),
        (0, "0원"),
    ])
    def test_format_won(self, won, text):
        assert agent._format_won(won) == text


# ── 연금계좌 세액공제 ─────────────────────────────────────────────────

# 근거 문서 표(doc41) 형식. 한도·세율은 이 문장에서 읽는다.
DOC41 = {
    "chunk_id": "doc41_p2_0001",
    "text": ("| 총급여 5,500만 원 이하 (종합소득금액 4,500만원 이하) | 16.5% | "
             "연 900만 원 (연금저축 단독 600만원) |\n"
             "| 총급여 5,500만 원 초과 | 13.2% | 연 900만 원 (연금저축 단독 600만원) |"),
}


class TestTaxCredit:
    def test_high_income_both_accounts(self):
        st = agent.tax_credit(_state(
            "총급여 7천만원인 직장인이 연금저축에 300만원, IRP에 500만원을 납입하면 "
            "세액공제는 얼마인가요?", [DOC41]))
        # min(300,600) + 500 = 800만원 × 13.2% = 105.6만원
        assert "13.2%" in st["calc_result"]
        assert "**105.6만원**" in st["calc_result"]

    def test_pension_cap_then_total_cap(self):
        st = agent.tax_credit(_state(
            "총급여 5천만원, 연금저축에 800만원과 IRP에 500만원을 넣으면 세액공제 얼마?",
            [DOC41]))
        # 연금저축 800 → 단독한도 600, +500 = 1,100 → 합산한도 900 × 16.5% = 148.5만원
        assert "단독한도 600만원 적용" in st["calc_result"]
        assert "합산한도 900만원 적용" in st["calc_result"]
        assert "**148.5만원**" in st["calc_result"]

    def test_limits_come_from_evidence_not_constants(self):
        # 근거의 숫자를 바꾸면 결과도 따라 바뀐다
        changed = {"chunk_id": "x", "text": DOC41["text"].replace("900만", "1,000만")}
        st = agent.tax_credit(_state(
            "총급여 5천만원, IRP에 1,000만원 납입 시 세액공제는?", [changed]))
        assert "**165만원**" in st["calc_result"]

    def test_no_evidence_no_result(self):
        st = agent.tax_credit(_state(
            "총급여 5천만원, IRP에 500만원 납입 시 세액공제는?"))
        assert "calc_result" not in st

    def test_multiple_incomes_not_handled(self):
        st = agent.tax_credit(_state(
            "총급여 5천만원과 총급여 7천만원일 때 IRP 500만원 세액공제 차이는?", [DOC41]))
        assert "calc_result" not in st


# ── 연금소득세율 나이 구간 ────────────────────────────────────────────

BRACKET = {
    "chunk_id": "doc39_p0_0005",
    "text": "연금소득세는 만55세~69세는 5.5%, 만70세~79세는 4.4%, 만80세 이상은 3.3%입니다.",
}


@pytest.mark.parametrize("age, rate", [(60, "5.5"), (72, "4.4"), (79, "4.4"), (85, "3.3")])
def test_tax_bracket(age, rate):
    st = agent.tax_bracket(_state(f"만 {age}세 연금소득세율은?", [BRACKET]))
    assert f"연금소득세율 {rate}%" in st["calc_result"]


def test_tax_bracket_ignores_multiple_ages():
    st = agent.tax_bracket(_state("만 60세와 만 75세의 세율 차이는?", [BRACKET]))
    assert "calc_result" not in st
