from pathlib import Path

from devrag.config import DevragConfig, load_config


def test_default_config_has_expected_values():
    config = DevragConfig()
    assert config.embedding.model == "nomic-embed-text"
    assert config.embedding.provider == "ollama"
    assert config.embedding.ollama_url == "http://localhost:11434"
    assert config.embedding.batch_size == 64
    assert config.vector_store.backend == "chromadb"
    assert config.retrieval.top_k == 20
    assert config.retrieval.final_k == 5
    assert config.retrieval.rerank is True
    assert config.code.chunk_max_tokens == 512
    assert config.code.respect_gitignore is True
    assert "*.min.js" in config.code.exclude_patterns
    assert "node_modules/**" in config.code.exclude_patterns


def test_load_config_from_yaml(tmp_dir):
    yaml_content = """
embedding:
  model: all-MiniLM-L6-v2
  provider: sentence-transformers
retrieval:
  top_k: 30
  final_k: 10
"""
    config_path = tmp_dir / ".devrag.yaml"
    config_path.write_text(yaml_content)
    config = load_config(project_dir=tmp_dir)
    assert config.embedding.model == "all-MiniLM-L6-v2"
    assert config.embedding.provider == "sentence-transformers"
    assert config.retrieval.top_k == 30
    assert config.retrieval.final_k == 10
    assert config.vector_store.backend == "chromadb"
    assert config.code.chunk_max_tokens == 512


def test_load_config_no_file_returns_defaults(tmp_dir):
    config = load_config(project_dir=tmp_dir)
    assert config.embedding.model == "nomic-embed-text"


def test_load_config_user_dir_fallback(tmp_dir, monkeypatch):
    user_config_dir = tmp_dir / "user_config"
    user_config_dir.mkdir()
    (user_config_dir / "devrag.yaml").write_text("embedding:\n  model: custom-model\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_dir / "user_config_parent"))
    config = load_config(project_dir=tmp_dir, user_config_dir=user_config_dir)
    assert config.embedding.model == "custom-model"
