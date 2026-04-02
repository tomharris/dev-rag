from unittest.mock import MagicMock, patch
import pytest
from devrag.retrieve.reranker import Reranker
from devrag.types import SearchResult


@patch("devrag.retrieve.reranker.CrossEncoder")
def test_reranker_reorders_by_relevance(mock_ce_class):
    mock_model = MagicMock()
    mock_ce_class.return_value = mock_model
    mock_model.rank.return_value = [
        {"corpus_id": 2, "score": 9.5},
        {"corpus_id": 0, "score": 7.2},
        {"corpus_id": 1, "score": 1.1},
    ]
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = [
        SearchResult(chunk_id="chunk_1", text="auth code", score=0.9, metadata={}),
        SearchResult(chunk_id="chunk_2", text="unrelated", score=0.8, metadata={}),
        SearchResult(chunk_id="chunk_3", text="login handler", score=0.7, metadata={}),
    ]
    results = reranker.rerank("how does authentication work", candidates, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_3"
    assert results[1].chunk_id == "chunk_1"
    assert results[0].score > results[1].score


@patch("devrag.retrieve.reranker.CrossEncoder")
def test_reranker_with_fewer_candidates_than_top_k(mock_ce_class):
    mock_model = MagicMock()
    mock_ce_class.return_value = mock_model
    mock_model.rank.return_value = [{"corpus_id": 0, "score": 5.0}]
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = [SearchResult(chunk_id="c1", text="only one", score=1.0, metadata={})]
    results = reranker.rerank("query", candidates, top_k=5)
    assert len(results) == 1


@patch("devrag.retrieve.reranker.CrossEncoder")
def test_reranker_empty_candidates(mock_ce_class):
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    results = reranker.rerank("query", [], top_k=5)
    assert results == []
