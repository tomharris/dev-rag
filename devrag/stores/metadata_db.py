from __future__ import annotations

import sqlite3
from pathlib import Path


class MetadataDB:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunk_sources (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_sources_file
                ON chunk_sources(file_path);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(chunk_id UNINDEXED, content);

            CREATE TABLE IF NOT EXISTS pr_sync_cursors (
                repo TEXT PRIMARY KEY,
                last_synced TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pr_chunk_sources (
                chunk_id TEXT PRIMARY KEY,
                repo TEXT NOT NULL,
                pr_number INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pr_chunk_sources_repo_pr
                ON pr_chunk_sources(repo, pr_number);
        """)
        self._conn.commit()

    def get_file_hash(self, file_path: str) -> str | None:
        row = self._conn.execute(
            "SELECT hash FROM file_hashes WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None

    def set_file_hash(self, file_path: str, hash_value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO file_hashes (file_path, hash) VALUES (?, ?)",
            (file_path, hash_value),
        )
        self._conn.commit()

    def remove_file(self, file_path: str) -> None:
        self.delete_fts_for_file(file_path)
        self._conn.execute("DELETE FROM chunk_sources WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM file_hashes WHERE file_path = ?", (file_path,))
        self._conn.commit()

    def get_all_indexed_files(self) -> list[str]:
        rows = self._conn.execute("SELECT file_path FROM file_hashes").fetchall()
        return [r[0] for r in rows]

    def set_chunk_source(self, chunk_id: str, file_path: str, line_start: int, line_end: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO chunk_sources (chunk_id, file_path, line_start, line_end) VALUES (?, ?, ?, ?)",
            (chunk_id, file_path, line_start, line_end),
        )
        self._conn.commit()

    def get_chunks_for_file(self, file_path: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunk_sources WHERE file_path = ?", (file_path,)
        ).fetchall()
        return [r[0] for r in rows]

    def upsert_fts(self, chunk_id: str, content: str) -> None:
        self._conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
        self._conn.execute("INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)", (chunk_id, content))
        self._conn.commit()

    def delete_fts(self, chunk_ids: list[str]) -> None:
        placeholders = ",".join("?" for _ in chunk_ids)
        self._conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)
        self._conn.commit()

    def delete_fts_for_file(self, file_path: str) -> None:
        chunk_ids = self.get_chunks_for_file(file_path)
        if chunk_ids:
            self.delete_fts(chunk_ids)

    def search_fts(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (query, limit),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_pr_sync_cursor(self, repo: str) -> str | None:
        row = self._conn.execute(
            "SELECT last_synced FROM pr_sync_cursors WHERE repo = ?", (repo,)
        ).fetchone()
        return row[0] if row else None

    def set_pr_sync_cursor(self, repo: str, last_synced: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO pr_sync_cursors (repo, last_synced) VALUES (?, ?)",
            (repo, last_synced),
        )
        self._conn.commit()

    def set_pr_chunk_source(self, chunk_id: str, repo: str, pr_number: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO pr_chunk_sources (chunk_id, repo, pr_number) VALUES (?, ?, ?)",
            (chunk_id, repo, pr_number),
        )
        self._conn.commit()

    def get_chunks_for_pr(self, repo: str, pr_number: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM pr_chunk_sources WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchall()
        return [r[0] for r in rows]

    def delete_chunks_for_pr(self, repo: str, pr_number: int) -> None:
        chunk_ids = self.get_chunks_for_pr(repo, pr_number)
        if chunk_ids:
            self.delete_fts(chunk_ids)
            placeholders = ",".join("?" for _ in chunk_ids)
            self._conn.execute(
                f"DELETE FROM pr_chunk_sources WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
