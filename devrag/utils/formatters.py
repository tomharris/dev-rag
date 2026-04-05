from __future__ import annotations

from devrag.types import DocIndexStats, IndexStats, IssueSyncStats, PRSyncStats, SearchResult


def format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        chunk_type = r.metadata.get("chunk_type", "")
        if "issue_number" in r.metadata and chunk_type in ("description", "comment"):
            issue_num = r.metadata.get("issue_number", "?")
            issue_title = r.metadata.get("issue_title", "")
            if chunk_type == "comment":
                comment_author = r.metadata.get("comment_author", "")
                lines.append(f"### {i}. [Issue #{issue_num}] Comment by {comment_author}")
            else:
                issue_author = r.metadata.get("issue_author", "")
                lines.append(f"### {i}. [Issue #{issue_num}] {issue_title} (by {issue_author})")
            lines.append("```")
            text_lines = r.text.strip().split("\n")
            preview = "\n".join(text_lines[:10])
            if len(text_lines) > 10:
                preview += "\n# ... (truncated)"
            lines.append(preview)
            lines.append("```")
            lines.append("")
        elif chunk_type in ("diff", "description", "review_comment"):
            pr_num = r.metadata.get("pr_number", "?")
            pr_title = r.metadata.get("pr_title", "")
            file_path = r.metadata.get("file_path", "")
            if chunk_type == "review_comment":
                reviewer = r.metadata.get("reviewer", "")
                lines.append(f"### {i}. [PR #{pr_num}] Review comment by {reviewer} on {file_path}")
            elif chunk_type == "description":
                pr_author = r.metadata.get("pr_author", "")
                lines.append(f"### {i}. [PR #{pr_num}] {pr_title} (by {pr_author})")
            else:
                lines.append(f"### {i}. [PR #{pr_num}] {pr_title} — {file_path}")
            lines.append("```diff")
            text_lines = r.text.strip().split("\n")
            preview = "\n".join(text_lines[:10])
            if len(text_lines) > 10:
                preview += "\n# ... (truncated)"
            lines.append(preview)
            lines.append("```")
            lines.append("")
        elif chunk_type == "document":
            file_path = r.metadata.get("file_path", "unknown")
            section_path = r.metadata.get("section_path", "")
            entity_name = r.metadata.get("entity_name", section_path)
            lines.append(f"### {i}. [{entity_name}] {file_path}")
            if section_path:
                lines.append(f"*Section: {section_path}*")
            lines.append("```")
            text_lines = r.text.strip().split("\n")
            preview = "\n".join(text_lines[:10])
            if len(text_lines) > 10:
                preview += "\n# ... (truncated)"
            lines.append(preview)
            lines.append("```")
            lines.append("")
        else:
            file_path = r.metadata.get("file_path", "unknown")
            line_range = r.metadata.get("line_range", [])
            entity_name = r.metadata.get("entity_name", "")
            language = r.metadata.get("language", "")
            location = file_path
            if line_range:
                location += f":{line_range[0]}-{line_range[1]}"
            lines.append(f"### {i}. {entity_name} ({location})")
            lines.append(f"```{language}")
            text_lines = r.text.strip().split("\n")
            preview = "\n".join(text_lines[:10])
            if len(text_lines) > 10:
                preview += "\n# ... (truncated)"
            lines.append(preview)
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def format_index_stats(stats: IndexStats) -> str:
    parts = [
        f"Scanned {stats.files_scanned} files",
        f"Indexed {stats.files_indexed} files ({stats.chunks_created} chunks)",
        f"Skipped {stats.files_skipped} unchanged files",
    ]
    if stats.files_removed:
        parts.append(f"Removed {stats.files_removed} deleted files")
    return ". ".join(parts) + "."


def format_doc_index_stats(stats: DocIndexStats) -> str:
    return f"Scanned {stats.files_scanned} files. Indexed {stats.files_indexed} files ({stats.chunks_created} chunks)."


def format_pr_sync_stats(stats: PRSyncStats) -> str:
    parts = [
        f"Fetched {stats.prs_fetched} PRs",
        f"Indexed {stats.prs_indexed} PRs ({stats.chunks_created} chunks)",
    ]
    if stats.prs_skipped:
        parts.append(f"Skipped {stats.prs_skipped} unchanged PRs")
    return ". ".join(parts) + "."


def format_issue_sync_stats(stats: IssueSyncStats) -> str:
    parts = [
        f"Fetched {stats.issues_fetched} issues",
        f"Indexed {stats.issues_indexed} issues ({stats.chunks_created} chunks)",
    ]
    if stats.issues_skipped:
        parts.append(f"Skipped {stats.issues_skipped} issues (PRs or unchanged)")
    return ". ".join(parts) + "."
