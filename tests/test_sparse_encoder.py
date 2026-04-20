from unittest.mock import MagicMock, patch

import numpy as np
from qdrant_client.models import SparseVector

from devrag.ingest.sparse_encoder import BM25SparseEncoder


def _fake_embedding(indices, values):
    emb = MagicMock()
    emb.indices = np.array(indices)
    emb.values = np.array(values)
    return emb


def test_encode_empty_list():
    enc = BM25SparseEncoder()
    assert enc.encode([]) == []


def test_encode_produces_sparse_vectors():
    enc = BM25SparseEncoder()
    fake_model = MagicMock()
    fake_model.embed.return_value = iter([
        _fake_embedding([1, 2], [0.5, 0.3]),
        _fake_embedding([3, 4], [0.7, 0.2]),
    ])
    with patch.object(enc, "_get_model", return_value=fake_model):
        result = enc.encode(["hello world", "foo bar"])
    assert len(result) == 2
    assert isinstance(result[0], SparseVector)
    assert result[0].indices == [1, 2]
    assert result[0].values == [0.5, 0.3]


def test_encode_handles_empty_text():
    enc = BM25SparseEncoder()
    fake_model = MagicMock()
    fake_model.embed.return_value = iter([_fake_embedding([7], [0.9])])
    with patch.object(enc, "_get_model", return_value=fake_model):
        result = enc.encode(["", "hello"])
    assert result[0].indices == []
    assert result[0].values == []
    assert result[1].indices == [7]


def test_encode_query_empty_returns_empty_sparse():
    enc = BM25SparseEncoder()
    result = enc.encode_query("")
    assert result.indices == []
    assert result.values == []


def test_encode_query_produces_sparse_vector():
    enc = BM25SparseEncoder()
    fake_model = MagicMock()
    fake_model.query_embed.return_value = iter([_fake_embedding([5], [1.2])])
    with patch.object(enc, "_get_model", return_value=fake_model):
        result = enc.encode_query("query text")
    assert result.indices == [5]
    assert result.values == [1.2]
