"""fee_sql이 쓰는 SQL 문자열 처리 함수의 단위 테스트. DB를 쓰지 않는다."""
import pytest

import agent


# ── _validate_sql ─────────────────────────────────────────────────────

class TestValidateSql:
    def test_appends_limit_and_strips_semicolon(self):
        assert (agent._validate_sql("SELECT * FROM fund_fees;")
                == "SELECT * FROM fund_fees LIMIT 50")

    def test_keeps_existing_limit(self):
        sql = "SELECT * FROM fund_fees LIMIT 3"
        assert agent._validate_sql(sql) == sql

    @pytest.mark.parametrize("sql", [
        "DELETE FROM fund_fees",
        "WITH x AS (SELECT 1) SELECT * FROM x",
    ])
    def test_rejects_non_select(self, sql):
        with pytest.raises(ValueError, match="SELECT만"):
            agent._validate_sql(sql)

    @pytest.mark.parametrize("sql, kw", [
        ("SELECT * FROM fund_fees WHERE 1=1 OR DROP", "DROP"),
        ("SELECT 1 FROM fund_fees; ATTACH DATABASE 'x' AS y", "ATTACH"),
        ("SELECT 1 FROM fund_fees WHERE PRAGMA", "PRAGMA"),
    ])
    def test_rejects_dangerous_keywords(self, sql, kw):
        with pytest.raises(ValueError, match=kw):
            agent._validate_sql(sql)

    def test_replace_function_is_allowed_but_replace_into_is_not(self):
        ok = "SELECT * FROM fund_fees WHERE REPLACE(fund_name,' ','') LIKE '%a%'"
        assert agent._validate_sql(ok).startswith(ok)
        with pytest.raises(ValueError, match="REPLACE"):
            agent._validate_sql("SELECT 1 FROM t WHERE REPLACE INTO t VALUES (1)")

    def test_rejects_multiple_statements(self):
        with pytest.raises(ValueError, match="여러 문장"):
            agent._validate_sql("SELECT 1 FROM t; SELECT 2 FROM t")


# ── _split_top_level ──────────────────────────────────────────────────

class TestSplitTopLevel:
    def test_splits_only_outside_parentheses(self):
        text = "(a = 1 OR b = 2) OR c = 3"
        assert agent._split_top_level(text, "OR") == ["(a = 1 OR b = 2)", "c = 3"]

    def test_and_split(self):
        text = "x = 1 AND (y = 2 OR z = 3) AND w = 4"
        assert agent._split_top_level(text, "AND") == [
            "x = 1", "(y = 2 OR z = 3)", "w = 4"]

    def test_keyword_is_case_insensitive_and_word_bounded(self):
        # 'ORDER'나 'COLOR' 안의 OR는 구분자가 아니다
        text = "color = 'red' or fund_name LIKE '%ORDER%'"
        assert agent._split_top_level(text, "OR") == [
            "color = 'red'", "fund_name LIKE '%ORDER%'"]

    def test_no_keyword_returns_whole(self):
        assert agent._split_top_level("  a = 1  ", "OR") == ["a = 1"]


# ── _split_where_suffix ───────────────────────────────────────────────

class TestSplitWhereSuffix:
    def test_no_where(self):
        assert agent._split_where_suffix("SELECT * FROM fund_fees") is None

    def test_splits_order_by_and_limit(self):
        sql = "SELECT a FROM t WHERE x = 1 AND y = 2 ORDER BY a LIMIT 5"
        pre, where, suf = agent._split_where_suffix(sql)
        assert pre == "SELECT a FROM t WHERE"
        assert where.strip() == "x = 1 AND y = 2"
        assert suf == "ORDER BY a LIMIT 5"

    def test_limit_inside_parentheses_is_not_the_tail(self):
        sql = ("SELECT a FROM t WHERE x IN (SELECT x FROM u LIMIT 1) "
               "ORDER BY a")
        pre, where, suf = agent._split_where_suffix(sql)
        assert where.strip() == "x IN (SELECT x FROM u LIMIT 1)"
        assert suf == "ORDER BY a"


# ── _split_or_groups ──────────────────────────────────────────────────

class TestSplitOrGroups:
    def test_top_level_or_per_fund(self):
        sql = ("SELECT * FROM fund_fees WHERE (fund_name LIKE '%A%' AND class_code = 'C')"
               " OR (fund_name LIKE '%B%' AND class_code = 'Ae') ORDER BY fee_total")
        prefix, conds, suffix = agent._split_or_groups(sql)
        assert prefix == "SELECT * FROM fund_fees WHERE"
        assert conds == ["(fund_name LIKE '%A%' AND class_code = 'C')",
                         "(fund_name LIKE '%B%' AND class_code = 'Ae')"]
        assert suffix == "ORDER BY fee_total"

    def test_distributes_common_condition(self):
        sql = ("SELECT * FROM fund_fees WHERE (fund_name LIKE '%A%' OR fund_name LIKE '%B%')"
               " AND channel = '오프라인'")
        _, conds, _ = agent._split_or_groups(sql)
        assert conds == ["(fund_name LIKE '%A%') AND channel = '오프라인'",
                         "(fund_name LIKE '%B%') AND channel = '오프라인'"]

    def test_aggregate_or_group_by_is_left_alone(self):
        assert agent._split_or_groups(
            "SELECT MIN(fee_total) FROM fund_fees WHERE fund_name LIKE '%A%'"
            " OR fund_name LIKE '%B%'") is None
        assert agent._split_or_groups(
            "SELECT fund_name FROM fund_fees WHERE fund_name LIKE '%A%'"
            " OR fund_name LIKE '%B%' GROUP BY fund_name") is None

    def test_or_without_fund_name_on_each_side_is_not_split(self):
        assert agent._split_or_groups(
            "SELECT * FROM fund_fees WHERE fund_name LIKE '%A%' OR channel = '온라인'") is None


# ── 클래스 · 계좌유형 · 판매경로 정규화 ───────────────────────────────

class TestNormalizeClassSql:
    def test_eq(self):
        assert (agent._normalize_class_sql("WHERE class_code = 'A-e'")
                == "WHERE UPPER(REPLACE(class_code,'-','')) = 'AE'")

    def test_in(self):
        assert (agent._normalize_class_sql("WHERE class_code IN ('Ae', 'c-e')")
                == "WHERE UPPER(REPLACE(class_code,'-','')) IN ('AE', 'CE')")

    def test_unrelated_sql_unchanged(self):
        sql = "SELECT * FROM fund_fees WHERE channel = '온라인'"
        assert agent._normalize_class_sql(sql) == sql


class TestNormalizeChannelSql:
    @pytest.mark.parametrize("literal, cond", [
        ("온라인직접판매", "channel LIKE '온라인%'"),
        ("다이렉트", "channel LIKE '온라인%'"),
        ("온라인슈퍼", "channel = '온라인슈퍼'"),
        ("창구", "channel = '오프라인'"),
        ("오프라인", "channel = '오프라인'"),
    ])
    def test_eq(self, literal, cond):
        assert agent._normalize_channel_sql(f"WHERE channel = '{literal}'") == f"WHERE {cond}"

    def test_unknown_literal_unchanged(self):
        sql = "WHERE channel = '방문판매'"
        assert agent._normalize_channel_sql(sql) == sql

    def test_in_becomes_parenthesized_or(self):
        out = agent._normalize_channel_sql("WHERE channel IN ('오프라인', '온라인직접판매')")
        assert out == "WHERE (channel = '오프라인' OR channel LIKE '온라인%')"

    def test_in_with_unknown_literal_left_whole(self):
        sql = "WHERE channel IN ('오프라인', '방문판매')"
        assert agent._normalize_channel_sql(sql) == sql


class TestNormalizeAccountSql:
    def test_synonym_to_db_value(self, fees_db):
        assert (agent._normalize_account_sql("WHERE account_type = '개인연금'")
                == "WHERE account_type = '연금저축'")
        assert (agent._normalize_account_sql("WHERE account_type = 'IRP'")
                == "WHERE account_type = '퇴직연금'")

    def test_in_dedupes(self, fees_db):
        assert (agent._normalize_account_sql(
            "WHERE account_type IN ('개인연금', '연금저축', 'DC')")
            == "WHERE account_type IN ('연금저축', '퇴직연금')")

    def test_unknown_literal_unchanged(self, fees_db):
        sql = "WHERE account_type = 'ISA'"
        assert agent._normalize_account_sql(sql) == sql

    def test_only_maps_to_values_present_in_db(self, fees_db, monkeypatch):
        monkeypatch.setattr(agent, "_ACCOUNT_TYPES_CACHE", {None: 3, "연금저축": 2})
        sql = "WHERE account_type = 'IRP'"
        assert agent._normalize_account_sql(sql) == sql


# ── 펀드명 공백 · 태그 정규화 ─────────────────────────────────────────

class TestFundNameRelax:
    def test_spread_turns_spaces_into_wildcards(self):
        assert agent._spread("미래에셋솔로몬 국공채") == "미래에셋솔로몬%국공채"

    def test_fund_like_drops_trailing_class_code(self):
        assert (agent._fund_like("미래에셋프리미엄크레딧알파 A-e")
                == "REPLACE(fund_name,' ','') LIKE '%미래에셋프리미엄크레딧알파%'")

    def test_eq_becomes_like(self):
        assert (agent._relax_fund_name("SELECT * FROM t WHERE fund_name = '삼성 코리아'")
                == "SELECT * FROM t WHERE REPLACE(fund_name,' ','') LIKE '%삼성%코리아%'")

    def test_in_becomes_or_of_likes(self):
        assert (agent._relax_fund_name("WHERE fund_name IN ('가나', '다라')")
                == "WHERE (REPLACE(fund_name,' ','') LIKE '%가나%'"
                   " OR REPLACE(fund_name,' ','') LIKE '%다라%')")

    def test_existing_like_gets_column_spaces_removed(self):
        assert (agent._relax_fund_name("WHERE fund_name LIKE '%NH-Amundi하나로단기채%'")
                == "WHERE REPLACE(fund_name,' ','') LIKE '%NH-Amundi하나로단기채%'")

    def test_nothing_to_relax(self):
        assert agent._relax_fund_name("SELECT * FROM t WHERE channel = '온라인'") is None


class TestStripFundLiterals:
    def test_strips_asset_tag(self):
        assert (agent._strip_fund_literals(
            "WHERE REPLACE(fund_name,' ','') LIKE '%스팍스한국엄선투자[주식]%'")
            == "WHERE REPLACE(fund_name,' ','') LIKE '%스팍스한국엄선투자%'")

    def test_strips_legal_suffix_and_series(self):
        assert (agent._strip_fund_literals(
            "WHERE fund_name LIKE '%미래에셋퇴직플랜단기증권자투자신탁 제1호%'")
            == "WHERE fund_name LIKE '%미래에셋퇴직플랜단기%'")

    def test_leaves_too_short_core(self):
        assert agent._strip_fund_literals("WHERE fund_name LIKE '%가[주식]%'") is None

    def test_nothing_to_strip(self):
        assert agent._strip_fund_literals("WHERE fund_name LIKE '%삼성코리아%'") is None


# ── 계좌유형 혼입 · 근거 없는 클래스 · LIMIT ───────────────────────────

class TestMigrateAccountSql:
    def test_in_with_only_account_literal(self):
        assert (agent._migrate_account_sql(
            "WHERE UPPER(REPLACE(class_code,'-','')) IN ('개인연금')")
            == "WHERE account_type = '연금저축'")

    def test_mixed_in_keeps_real_class_with_or(self):
        assert (agent._migrate_account_sql("WHERE class_code IN ('C-P', '퇴직연금')")
                == "WHERE (class_code IN ('C-P') OR account_type = '퇴직연금')")

    def test_real_class_containing_account_word_is_not_migrated(self):
        sql = "WHERE class_code = 'S-퇴직'"
        assert agent._migrate_account_sql(sql) == sql


class TestDropUngroundedClassCond:
    SQL = "SELECT * FROM fund_fees WHERE fund_name LIKE '%라이프사이클2030%' AND class_code = 'C-P'"

    def test_dropped_without_class_signal(self):
        out = agent._drop_ungrounded_class_cond(self.SQL, "미래에셋라이프사이클2030의 총보수는 몇 %인가요?")
        assert out.endswith("AND 1=1")

    @pytest.mark.parametrize("question", [
        "라이프사이클2030 C-P 총보수는?",
        "라이프사이클2030 연금 클래스 총보수는?",
    ])
    def test_kept_with_class_signal(self, question):
        assert agent._drop_ungrounded_class_cond(self.SQL, question) == self.SQL


def test_drop_top_level_limit():
    assert agent._drop_top_level_limit("ORDER BY fee_total ASC LIMIT 1") == "ORDER BY fee_total ASC"
    assert agent._drop_top_level_limit("ORDER BY fee_total") == "ORDER BY fee_total"


def test_has_multi_literal_in():
    assert agent._has_multi_literal_in("WHERE class_code IN ('AE', 'CE')")
    assert not agent._has_multi_literal_in("WHERE class_code IN ('AE')")
    assert not agent._has_multi_literal_in("WHERE class_code = 'AE'")
