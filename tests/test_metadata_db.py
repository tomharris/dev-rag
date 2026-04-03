from devrag.stores.metadata_db import MetadataDB


def test_file_hash_store_and_check(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("src/foo.py", "abc123")
    assert db.get_file_hash("src/foo.py") == "abc123"
    assert db.get_file_hash("src/bar.py") is None


def test_file_hash_update(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("src/foo.py", "hash1")
    db.set_file_hash("src/foo.py", "hash2")
    assert db.get_file_hash("src/foo.py") == "hash2"


def test_remove_file_hash(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("src/foo.py", "abc")
    db.remove_file("src/foo.py")
    assert db.get_file_hash("src/foo.py") is None


def test_get_all_indexed_files(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("a.py", "h1")
    db.set_file_hash("b.py", "h2")
    files = db.get_all_indexed_files()
    assert set(files) == {"a.py", "b.py"}


def test_chunk_source_mapping(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_chunk_source("chunk_1", "src/foo.py", 10, 25)
    db.set_chunk_source("chunk_2", "src/foo.py", 30, 50)
    chunks = db.get_chunks_for_file("src/foo.py")
    assert set(chunks) == {"chunk_1", "chunk_2"}


def test_remove_file_clears_chunks(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_file_hash("src/foo.py", "abc")
    db.set_chunk_source("chunk_1", "src/foo.py", 1, 10)
    db.remove_file("src/foo.py")
    assert db.get_chunks_for_file("src/foo.py") == []


def test_fts_index_and_search(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.upsert_fts("chunk_1", "def authenticate_user(username, password)")
    db.upsert_fts("chunk_2", "class DatabaseConnection with pooling support")
    db.upsert_fts("chunk_3", "function to parse JSON configuration files")
    results = db.search_fts("authenticate", limit=5)
    assert len(results) >= 1
    assert results[0][0] == "chunk_1"
    assert results[0][1] < 0


def test_fts_update_existing(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.upsert_fts("chunk_1", "original text about authentication")
    db.upsert_fts("chunk_1", "updated text about database pooling")
    results = db.search_fts("database pooling", limit=5)
    assert len(results) >= 1
    assert results[0][0] == "chunk_1"


def test_fts_delete(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.upsert_fts("chunk_1", "some text")
    db.delete_fts(["chunk_1"])
    results = db.search_fts("some text", limit=5)
    assert len(results) == 0


def test_pr_sync_cursor_set_and_get(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_pr_sync_cursor("acme/backend", "2026-03-15T10:00:00Z")
    assert db.get_pr_sync_cursor("acme/backend") == "2026-03-15T10:00:00Z"
    assert db.get_pr_sync_cursor("other/repo") is None


def test_pr_sync_cursor_update(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_pr_sync_cursor("acme/backend", "2026-03-01T00:00:00Z")
    db.set_pr_sync_cursor("acme/backend", "2026-03-15T10:00:00Z")
    assert db.get_pr_sync_cursor("acme/backend") == "2026-03-15T10:00:00Z"


def test_pr_chunk_source_mapping(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_pr_chunk_source("chunk_pr_1", "acme/backend", 1234)
    db.set_pr_chunk_source("chunk_pr_2", "acme/backend", 1234)
    db.set_pr_chunk_source("chunk_pr_3", "acme/backend", 5678)
    chunks = db.get_chunks_for_pr("acme/backend", 1234)
    assert set(chunks) == {"chunk_pr_1", "chunk_pr_2"}


def test_delete_chunks_for_pr(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_pr_chunk_source("c1", "acme/backend", 100)
    db.upsert_fts("c1", "some pr text")
    db.delete_chunks_for_pr("acme/backend", 100)
    assert db.get_chunks_for_pr("acme/backend", 100) == []
    assert db.search_fts("some pr text", limit=5) == []


def test_delete_fts_for_file(tmp_dir):
    db = MetadataDB(str(tmp_dir / "meta.db"))
    db.set_chunk_source("c1", "foo.py", 1, 10)
    db.set_chunk_source("c2", "foo.py", 11, 20)
    db.set_chunk_source("c3", "bar.py", 1, 10)
    db.upsert_fts("c1", "text one")
    db.upsert_fts("c2", "text two")
    db.upsert_fts("c3", "text three")
    db.delete_fts_for_file("foo.py")
    assert db.search_fts("text one", limit=5) == []
    results = db.search_fts("text three", limit=5)
    assert len(results) == 1
    assert results[0][0] == "c3"
