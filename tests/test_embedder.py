import httpx
import pytest
import respx

from devrag.ingest.embedder import OllamaEmbedder


@respx.mock
def test_embed_single_text():
    respx.post("http://localhost:11434/api/embed").respond(json={
        "model": "nomic-embed-text",
        "embeddings": [[0.1, 0.2, 0.3]],
    })
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed(["hello world"])
    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3]


@respx.mock
def test_embed_batch():
    respx.post("http://localhost:11434/api/embed").respond(json={
        "model": "nomic-embed-text",
        "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    })
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed(["a", "b", "c"])
    assert len(result) == 3


@respx.mock
def test_embed_large_batch_splits_requests():
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        data = request.content.decode()
        import json
        body = json.loads(data)
        n = len(body["input"])
        return httpx.Response(200, json={
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2]] * n,
        })

    respx.post("http://localhost:11434/api/embed").mock(side_effect=handler)
    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        ollama_url="http://localhost:11434",
        batch_size=2,
    )
    result = embedder.embed(["a", "b", "c", "d", "e"])
    assert len(result) == 5
    assert call_count == 3


@respx.mock
def test_embed_query():
    respx.post("http://localhost:11434/api/embed").respond(json={
        "model": "nomic-embed-text",
        "embeddings": [[0.1, 0.2, 0.3]],
    })
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed_query("search query")
    assert result == [0.1, 0.2, 0.3]


@respx.mock
def test_embed_empty_list():
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed([])
    assert result == []


@respx.mock
def test_embed_filters_empty_texts():
    """Empty/whitespace texts get zero vectors without hitting the API."""
    respx.post("http://localhost:11434/api/embed").respond(json={
        "model": "nomic-embed-text",
        "embeddings": [[0.1, 0.2, 0.3]],
    })
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed(["", "hello", "   "])
    assert len(result) == 3
    assert result[1] == [0.1, 0.2, 0.3]
    assert result[0] == [0.0, 0.0, 0.0]
    assert result[2] == [0.0, 0.0, 0.0]


@respx.mock
def test_embed_all_empty_texts():
    """All-empty input returns empty lists without calling API."""
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    result = embedder.embed(["", "  ", "\n"])
    assert len(result) == 3
    assert all(v == [] for v in result)


@respx.mock
def test_embed_error_logs_response_body():
    """Non-2xx responses log the body before raising."""
    respx.post("http://localhost:11434/api/embed").respond(
        status_code=400,
        json={"error": "empty input"},
    )
    embedder = OllamaEmbedder(model="nomic-embed-text", ollama_url="http://localhost:11434")
    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed(["valid text"])
