"""문자열 리터럴 안의 괄호·키워드를 SQL 구조로 오인하지 않는지 확인한다.

펀드명에는 '(채권)'·'1호(주식)' 같은 괄호가 흔하고, LLM이 그 일부를 잘라
LIKE '%1호(채%' 같은 리터럴을 만든다. 따옴표를 무시하고 괄호 깊이를 세면
짝 없는 '('가 깊이를 1로 묶어 둬서 뒤따르는 최상위 OR·ORDER BY를 놓친다.
"""
import agent


def test_split_top_level_ignores_paren_in_literal():
    text = "fund_name LIKE '%1호(채%' OR fund_name LIKE '%코어밸류%'"
    assert agent._split_top_level(text, "OR") == [
        "fund_name LIKE '%1호(채%'", "fund_name LIKE '%코어밸류%'"]


def test_split_top_level_ignores_keyword_in_literal():
    text = "fund_name LIKE '%A OR B%' OR fund_name LIKE '%C%'"
    assert agent._split_top_level(text, "OR") == [
        "fund_name LIKE '%A OR B%'", "fund_name LIKE '%C%'"]


def test_split_top_level_handles_escaped_quote():
    text = "fund_name = 'it''s (x' OR fund_name = 'y'"
    assert agent._split_top_level(text, "OR") == [
        "fund_name = 'it''s (x'", "fund_name = 'y'"]


def test_split_where_suffix_ignores_paren_in_literal():
    sql = ("SELECT * FROM fund_fees WHERE fund_name LIKE '%1호(채%' "
           "ORDER BY fee_total LIMIT 1")
    pre, where, suf = agent._split_where_suffix(sql)
    assert where.strip() == "fund_name LIKE '%1호(채%'"
    assert suf == "ORDER BY fee_total LIMIT 1"


def test_split_where_suffix_ignores_tail_keyword_in_literal():
    sql = "SELECT * FROM fund_fees WHERE fund_name LIKE '%LIMIT%' LIMIT 5"
    _, where, suf = agent._split_where_suffix(sql)
    assert where.strip() == "fund_name LIKE '%LIMIT%'"
    assert suf == "LIMIT 5"


def test_split_or_groups_with_paren_in_literal():
    sql = ("SELECT * FROM fund_fees WHERE fund_name LIKE '%고배당포커스연금저축1호(주%' "
           "OR fund_name LIKE '%코어밸류%' ORDER BY fee_total LIMIT 1")
    prefix, conds, suffix = agent._split_or_groups(sql)
    assert conds == ["fund_name LIKE '%고배당포커스연금저축1호(주%'",
                     "fund_name LIKE '%코어밸류%'"]
    assert suffix == "ORDER BY fee_total LIMIT 1"
