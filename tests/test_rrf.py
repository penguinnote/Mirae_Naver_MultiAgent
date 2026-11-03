"""RRF 융합은 search.rrf_fuse 하나만 쓴다."""
import agent
import search


def test_rrf_scores_and_order():
    ranks = {
        "both_mid": {"vec": 3, "bm25": 3},
        "vec_top": {"vec": 1},
        "bm25_top": {"bm25": 1},
    }
    out = [cid for cid, _ in search.rrf_fuse(ranks)]
    # 2/(60+3) > 1/(60+1): 양쪽에 다 걸린 청크가 한쪽 1위보다 앞선다
    assert out[0] == "both_mid"
    # 동점은 입력 순서를 유지한다
    assert out[1:] == ["vec_top", "bm25_top"]


def test_rrf_keeps_source_ranks():
    out = search.rrf_fuse({"a": {"vec": 2, "bm25": 5}})
    assert out == [("a", {"vec": 2, "bm25": 5})]


def test_agent_uses_the_same_function():
    assert agent.rrf_fuse is search.rrf_fuse
