from __future__ import annotations

import sqlite3
from pathlib import Path


class MetadataDB:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate_add_repo_column()
        self._create_tables()

    def _migrate_add_repo_column(self) -> None:
        """Migrate file_hashes and chunk_sources to include repo column."""
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(file_hashes)").fetchall()
        }
        if not cols or "repo" in cols:
            return  # Table doesn't exist yet or already migrated

        self._conn.executescript("""
            ALTER TABLE file_hashes RENAME TO _file_hashes_old;

            CREATE TABLE file_hashes (
                repo TEXT NOT NULL DEFAULT '',
                file_path TEXT NOT NULL,
                hash TEXT NOT NULL,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (repo, file_path)
            );

            INSERT INTO file_hashes (repo, file_path, hash, last_indexed)
                SELECT '', file_path, hash, last_indexed FROM _file_hashes_old;

            DROP TABLE _file_hashes_old;
        """)

        chunk_cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(chunk_sources)").fetchall()
        }
        if chunk_cols and "repo" not in chunk_cols:
            self._conn.executescript("""
                ALTER TABLE chunk_sources RENAME TO _chunk_sources_old;

                CREATE TABLE chunk_sources (
                    chunk_id TEXT PRIMARY KEY,
                    repo TEXT NOT NULL DEFAULT '',
                    file_path TEXT NOT NULL,
                    line_start INTEGER,
                    line_end INTEGER
                );

                INSERT INTO chunk_sources (chunk_id, repo, file_path, line_start, line_end)
                    SELECT chunk_id, '', file_path, line_start, line_end FROM _chunk_sources_old;

                DROP TABLE _chunk_sources_old;
            """)

        self._conn.commit()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS code_repos (
                repo TEXT PRIMARY KEY,
                repo_path TEXT NOT NULL,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS file_hashes (
                repo TEXT NOT NULL DEFAULT '',
                file_path TEXT NOT NULL,
                hash TEXT NOT NULL,
                last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (repo, file_path)
            );

            CREATE TABLE IF NOT EXISTS chunk_sources (
                chunk_id TEXT PRIMARY KEY,
                repo TEXT NOT NULL DEFAULT '',
                file_path TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_sources_file
                ON chunk_sources(file_path);

            CREATE INDEX IF NOT EXISTS idx_chunk_sources_repo_file
                ON chunk_sources(repo, file_path);

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

            CREATE TABLE IF NOT EXISTS issue_sync_cursors (
                repo TEXT PRIMARY KEY,
                last_synced TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS issue_chunk_sources (
                chunk_id TEXT PRIMARY KEY,
                repo TEXT NOT NULL,
                issue_number INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_issue_chunk_sources_repo_issue
                ON issue_chunk_sources(repo, issue_number);

            CREATE TABLE IF NOT EXISTS jira_sync_cursors (
                cursor_key TEXT PRIMARY KEY,
                last_synced TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jira_chunk_sources (
                chunk_id TEXT PRIMARY KEY,
                instance_url TEXT NOT NULL,
                ticket_key TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_jira_chunk_sources_instance_ticket
                ON jira_chunk_sources(instance_url, ticket_key);

            CREATE TABLE IF NOT EXISTS slite_sync_cursors (
                workspace_id TEXT PRIMARY KEY,
                last_synced TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS slite_chunk_sources (
                chunk_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                page_id TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_slite_chunk_sources_workspace_page
                ON slite_chunk_sources(workspace_id, page_id);

            CREATE TABLE IF NOT EXISTS chunk_collections (
                chunk_id TEXT PRIMARY KEY,
                collection TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_collections_collection
                ON chunk_collections(collection);

            CREATE TABLE IF NOT EXISTS query_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                collections TEXT NOT NULL,
                vector_ms REAL,
                bm25_ms REAL,
                rerank_ms REAL,
                total_ms REAL,
                result_count INTEGER,
                classification TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    def register_repo(self, repo: str, repo_path: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO code_repos (repo, repo_path) VALUES (?, ?)",
            (repo, repo_path),
        )
        self._conn.commit()

    def get_all_repos(self) -> list[tuple[str, str]]:
        rows = self._conn.execute("SELECT repo, repo_path FROM code_repos").fetchall()
        return [(r[0], r[1]) for r in rows]

    def remove_repo(self, repo: str) -> None:
        chunk_ids = self._conn.execute(
            "SELECT chunk_id FROM chunk_sources WHERE repo = ?", (repo,)
        ).fetchall()
        if chunk_ids:
            ids = [r[0] for r in chunk_ids]
            self.delete_fts(ids)
        self._conn.execute("DELETE FROM chunk_sources WHERE repo = ?", (repo,))
        self._conn.execute("DELETE FROM file_hashes WHERE repo = ?", (repo,))
        self._conn.execute("DELETE FROM code_repos WHERE repo = ?", (repo,))
        self._conn.commit()

    def get_file_hash(self, file_path: str, repo: str = "") -> str | None:
        row = self._conn.execute(
            "SELECT hash FROM file_hashes WHERE repo = ? AND file_path = ?", (repo, file_path)
        ).fetchone()
        return row[0] if row else None

    def set_file_hash(self, file_path: str, hash_value: str, repo: str = "") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO file_hashes (repo, file_path, hash) VALUES (?, ?, ?)",
            (repo, file_path, hash_value),
        )
        self._conn.commit()

    def remove_file(self, file_path: str, repo: str = "") -> None:
        self.delete_fts_for_file(file_path, repo=repo)
        self._conn.execute(
            "DELETE FROM chunk_sources WHERE repo = ? AND file_path = ?", (repo, file_path)
        )
        self._conn.execute(
            "DELETE FROM file_hashes WHERE repo = ? AND file_path = ?", (repo, file_path)
        )
        self._conn.commit()

    def get_indexed_files_for_repo(self, repo: str = "") -> list[str]:
        rows = self._conn.execute(
            "SELECT file_path FROM file_hashes WHERE repo = ?", (repo,)
        ).fetchall()
        return [r[0] for r in rows]

    def get_all_indexed_files(self) -> list[str]:
        rows = self._conn.execute("SELECT file_path FROM file_hashes").fetchall()
        return [r[0] for r in rows]

    def set_chunk_source(self, chunk_id: str, file_path: str, line_start: int, line_end: int, repo: str = "") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO chunk_sources (chunk_id, repo, file_path, line_start, line_end) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, repo, file_path, line_start, line_end),
        )
        self._conn.commit()

    def get_chunks_for_file(self, file_path: str, repo: str = "") -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunk_sources WHERE repo = ? AND file_path = ?", (repo, file_path)
        ).fetchall()
        return [r[0] for r in rows]

    def upsert_fts(self, chunk_id: str, content: str) -> None:
        self._conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
        self._conn.execute("INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)", (chunk_id, content))
        self._conn.commit()

    def delete_fts(self, chunk_ids: list[str]) -> None:
        placeholders = ",".join("?" for _ in chunk_ids)
        self._conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)
        self._conn.execute(f"DELETE FROM chunk_collections WHERE chunk_id IN ({placeholders})", chunk_ids)
        self._conn.commit()

    def delete_fts_for_file(self, file_path: str, repo: str = "") -> None:
        chunk_ids = self.get_chunks_for_file(file_path, repo=repo)
        if chunk_ids:
            self.delete_fts(chunk_ids)

    def set_chunk_collection(self, chunk_id: str, collection: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO chunk_collections (chunk_id, collection) VALUES (?, ?)",
            (chunk_id, collection),
        )
        self._conn.commit()

    def delete_chunk_collections(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        self._conn.execute(f"DELETE FROM chunk_collections WHERE chunk_id IN ({placeholders})", chunk_ids)
        self._conn.commit()

    def search_fts(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (query, limit),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def search_fts_scoped(self, query: str, collections: list[str], limit: int = 20) -> list[tuple[str, float]]:
        """BM25 search scoped to specific collections. Falls back to unscoped if chunk_collections is empty."""
        has_mappings = self._conn.execute("SELECT 1 FROM chunk_collections LIMIT 1").fetchone()
        if not has_mappings:
            return self.search_fts(query, limit)
        placeholders = ",".join("?" for _ in collections)
        rows = self._conn.execute(
            f"SELECT f.chunk_id, bm25(chunks_fts) as score "
            f"FROM chunks_fts f "
            f"JOIN chunk_collections cc ON cc.chunk_id = f.chunk_id "
            f"WHERE chunks_fts MATCH ? AND cc.collection IN ({placeholders}) "
            f"ORDER BY score LIMIT ?",
            [query, *collections, limit],
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

    def get_issue_sync_cursor(self, repo: str) -> str | None:
        row = self._conn.execute(
            "SELECT last_synced FROM issue_sync_cursors WHERE repo = ?", (repo,)
        ).fetchone()
        return row[0] if row else None

    def set_issue_sync_cursor(self, repo: str, last_synced: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO issue_sync_cursors (repo, last_synced) VALUES (?, ?)",
            (repo, last_synced),
        )
        self._conn.commit()

    def set_issue_chunk_source(self, chunk_id: str, repo: str, issue_number: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO issue_chunk_sources (chunk_id, repo, issue_number) VALUES (?, ?, ?)",
            (chunk_id, repo, issue_number),
        )
        self._conn.commit()

    def get_chunks_for_issue(self, repo: str, issue_number: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM issue_chunk_sources WHERE repo = ? AND issue_number = ?",
            (repo, issue_number),
        ).fetchall()
        return [r[0] for r in rows]

    def delete_chunks_for_issue(self, repo: str, issue_number: int) -> None:
        chunk_ids = self.get_chunks_for_issue(repo, issue_number)
        if chunk_ids:
            self.delete_fts(chunk_ids)
            placeholders = ",".join("?" for _ in chunk_ids)
            self._conn.execute(
                f"DELETE FROM issue_chunk_sources WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            self._conn.commit()

    def get_jira_sync_cursor(self, cursor_key: str) -> str | None:
        row = self._conn.execute(
            "SELECT last_synced FROM jira_sync_cursors WHERE cursor_key = ?", (cursor_key,)
        ).fetchone()
        return row[0] if row else None

    def set_jira_sync_cursor(self, cursor_key: str, last_synced: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO jira_sync_cursors (cursor_key, last_synced) VALUES (?, ?)",
            (cursor_key, last_synced),
        )
        self._conn.commit()

    def set_jira_chunk_source(self, chunk_id: str, instance_url: str, ticket_key: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO jira_chunk_sources (chunk_id, instance_url, ticket_key) VALUES (?, ?, ?)",
            (chunk_id, instance_url, ticket_key),
        )
        self._conn.commit()

    def get_chunks_for_jira_ticket(self, instance_url: str, ticket_key: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM jira_chunk_sources WHERE instance_url = ? AND ticket_key = ?",
            (instance_url, ticket_key),
        ).fetchall()
        return [r[0] for r in rows]

    def delete_chunks_for_jira_ticket(self, instance_url: str, ticket_key: str) -> None:
        chunk_ids = self.get_chunks_for_jira_ticket(instance_url, ticket_key)
        if chunk_ids:
            self.delete_fts(chunk_ids)
            placeholders = ",".join("?" for _ in chunk_ids)
            self._conn.execute(
                f"DELETE FROM jira_chunk_sources WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            self._conn.commit()

    def get_slite_sync_cursor(self, workspace_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT last_synced FROM slite_sync_cursors WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
        return row[0] if row else None

    def set_slite_sync_cursor(self, workspace_id: str, last_synced: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO slite_sync_cursors (workspace_id, last_synced) VALUES (?, ?)",
            (workspace_id, last_synced),
        )
        self._conn.commit()

    def set_slite_chunk_source(self, chunk_id: str, workspace_id: str, page_id: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO slite_chunk_sources (chunk_id, workspace_id, page_id) VALUES (?, ?, ?)",
            (chunk_id, workspace_id, page_id),
        )
        self._conn.commit()

    def get_chunks_for_slite_page(self, workspace_id: str, page_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM slite_chunk_sources WHERE workspace_id = ? AND page_id = ?",
            (workspace_id, page_id),
        ).fetchall()
        return [r[0] for r in rows]

    def delete_chunks_for_slite_page(self, workspace_id: str, page_id: str) -> None:
        chunk_ids = self.get_chunks_for_slite_page(workspace_id, page_id)
        if chunk_ids:
            self.delete_fts(chunk_ids)
            placeholders = ",".join("?" for _ in chunk_ids)
            self._conn.execute(
                f"DELETE FROM slite_chunk_sources WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            self._conn.commit()

    def log_query_metric(self, query: str, collections: list[str], vector_ms: float,
                         bm25_ms: float, rerank_ms: float, total_ms: float,
                         result_count: int, classification: str) -> None:
        self._conn.execute(
            "INSERT INTO query_metrics (query, collections, vector_ms, bm25_ms, rerank_ms, total_ms, result_count, classification) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (query, ",".join(collections), vector_ms, bm25_ms, rerank_ms, total_ms, result_count, classification),
        )
        self._conn.commit()

    def get_query_metrics(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT query, collections, vector_ms, bm25_ms, rerank_ms, total_ms, result_count, classification, timestamp "
            "FROM query_metrics ORDER BY timestamp DESC LIMIT ?", (limit,),
        ).fetchall()
        return [{"query": r[0], "collections": r[1], "vector_ms": r[2], "bm25_ms": r[3],
                 "rerank_ms": r[4], "total_ms": r[5], "result_count": r[6],
                 "classification": r[7], "timestamp": r[8]} for r in rows]

    def get_query_stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*), AVG(total_ms), AVG(result_count), AVG(vector_ms), AVG(bm25_ms), AVG(rerank_ms) "
            "FROM query_metrics"
        ).fetchone()
        return {"total_queries": row[0], "avg_total_ms": row[1] or 0.0,
                "avg_result_count": row[2] or 0.0, "avg_vector_ms": row[3] or 0.0,
                "avg_bm25_ms": row[4] or 0.0, "avg_rerank_ms": row[5] or 0.0}

    def close(self) -> None:
        self._conn.close()
