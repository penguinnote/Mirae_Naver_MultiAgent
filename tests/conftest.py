"""공용 픽스처.

테스트는 API 키·Chroma 인덱스·실제 fund_fees.sqlite 없이 돈다.
수수료 DB가 필요한 테스트는 아래 fees_db 픽스처가 만든 소형 DB를 쓴다.
행은 실제 DB의 표기 흔들림(A-E / Ae / C-e 공존, 띄어쓴 펀드명, 온라인슈퍼
채널 등)을 재현하도록 골랐고, 수치는 테스트용이다.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402

FEE_COLUMNS = ("fund_code", "fund_name", "class_code", "account_type",
               "channel", "fee_total", "page")

FEE_ROWS = [
    ("KR0000000001", "하나파워e단기채증권자투자신탁[채권]", "A", None, "오프라인", 0.1736, 12),
    ("KR0000000001", "하나파워e단기채증권자투자신탁[채권]", "C-E", None, "온라인", 0.1303, 12),
    ("KR0000000001", "하나파워e단기채증권자투자신탁[채권]", "A-E", None, "온라인", 0.1236, 12),
    ("KR0000000001", "하나파워e단기채증권자투자신탁[채권]", "C-P", "연금저축", "오프라인", 0.177, 12),
    ("KR0000000002", "NH-Amundi 하나로 단기채 증권투자신탁[채권]", "Ae", None, "온라인", 0.17, 8),
    ("KR0000000002", "NH-Amundi 하나로 단기채 증권투자신탁[채권]", "S-P2", "연금저축", "온라인슈퍼", 0.15, 8),
    ("KR0000000002", "NH-Amundi 하나로 단기채 증권투자신탁[채권]", "C", None, "오프라인", 0.5, 8),
    ("KR0000000003", "스팍스한국엄선투자증권자투자신탁[주식]", "A", None, "오프라인", 0.9, 20),
    ("KR0000000004", "삼성코리아중기채권증권자투자신탁", "Ae", None, "온라인", 0.3, 15),
    ("KR0000000004", "삼성코리아중기채권증권자투자신탁", "Ce", None, "온라인", 0.4, 15),
    ("KR0000000005", "미래에셋고배당포커스연금저축증권자투자신탁1호(주식)", "C-e", "연금저축", "온라인", 0.8, 30),
    ("KR0000000005", "미래에셋고배당포커스연금저축증권자투자신탁1호(주식)", "C", "연금저축", "오프라인", 1.2, 30),
    ("KR0000000006", "미래에셋코어밸류연금저축증권자투자신탁(주식)", "C", "연금저축", "오프라인", 1.1, 31),
    ("KR0000000006", "미래에셋코어밸류연금저축증권자투자신탁(주식)", "S-퇴직", "퇴직연금", "온라인슈퍼", 0.25, 31),
]


def _reset_db_caches():
    agent._ACCOUNT_TYPES_CACHE = None
    agent._CLASS_CODES_CACHE = None
    agent._KNOWN_FUND_SHORTHANDS = None


@pytest.fixture
def fees_db(tmp_path, monkeypatch):
    """테스트용 fund_fees DB를 만들고 agent가 그 파일을 읽게 한다.

    경로에 공백과 한글을 넣어 URI 인코딩이 필요한 경우도 함께 확인한다.
    반환값은 행을 더 넣을 때 쓰는 함수다(캐시도 함께 비운다).
    """
    db_dir = tmp_path / "수수료 DB"
    db_dir.mkdir()
    path = db_dir / "fund_fees.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE fund_fees (fund_code TEXT, fund_name TEXT, base_date TEXT,"
        " class_label TEXT, class_code TEXT, account_type TEXT, channel TEXT,"
        " front_load_text TEXT, fee_total REAL, fee_distribution REAL,"
        " fee_peer_avg REAL, fee_total_cost REAL, chunk_id TEXT, page INTEGER,"
        " source_path TEXT)")

    def add_rows(rows):
        c = sqlite3.connect(path)
        c.executemany(
            f"INSERT INTO fund_fees ({', '.join(FEE_COLUMNS)}) VALUES (?,?,?,?,?,?,?)",
            rows)
        c.commit()
        c.close()
        _reset_db_caches()

    conn.close()
    add_rows(FEE_ROWS)
    monkeypatch.setattr(agent, "FUND_FEES_DB", str(path))
    _reset_db_caches()
    yield add_rows
    _reset_db_caches()


@pytest.fixture
def run_fee_sql(fees_db, monkeypatch):
    """HCX 대신 정해 둔 SQL을 돌려주게 하고 fee_sql 노드를 실행한다."""
    def _run(question, llm_sql):
        monkeypatch.setattr(agent, "call_hcx", lambda *a, **k: llm_sql)
        state = {"question": question, "route": {"need_sql": True}, "trace": []}
        return agent.fee_sql(state)
    return _run
