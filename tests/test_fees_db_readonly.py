"""수수료 DB는 읽기 전용으로 연다.

런타임은 fund_fees.sqlite를 조회만 한다. 쓰기 가능한 연결로 열면
_validate_sql을 빠져나간 문장 하나가 DB를 바꿀 수 있고, 파일이 없을 때
sqlite3가 빈 DB를 새로 만들어 버려 '테이블 없음'이라는 엉뚱한 오류로 번진다.
"""
import sqlite3

import pytest

import agent


def test_all_runtime_connections_are_read_only(run_fee_sql, monkeypatch):
    calls = []
    real_connect = sqlite3.connect

    def spy(database, *a, **k):
        calls.append((str(database), k.get("uri", False)))
        return real_connect(database, *a, **k)

    monkeypatch.setattr(sqlite3, "connect", spy)
    # 캐시 3종 + 본조회 + 완화 재조회 + _run_sql_rows를 모두 지나가는 질문
    st = run_fee_sql(
        "스팍스한국엄선투자[주식] 총보수",
        "SELECT fund_name, class_code, fee_total FROM fund_fees "
        "WHERE fund_name LIKE '%스팍스한국엄선투자[주식]%'")
    agent._known_fund_shorthands()
    assert st["sql_rows"]
    assert calls
    for db, uri in calls:
        assert uri is True, db
        assert db.startswith("file:") and db.endswith("?mode=ro"), db


def test_connection_rejects_writes(fees_db):
    conn = agent._connect_fees()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM fund_fees")
    finally:
        conn.close()


def test_missing_db_is_not_created(tmp_path, monkeypatch):
    missing = tmp_path / "없음" / "fund_fees.sqlite"
    missing.parent.mkdir()
    monkeypatch.setattr(agent, "FUND_FEES_DB", str(missing))
    monkeypatch.setattr(agent, "_ACCOUNT_TYPES_CACHE", None)
    with pytest.raises(sqlite3.OperationalError):
        agent._account_types()
    assert not missing.exists()
