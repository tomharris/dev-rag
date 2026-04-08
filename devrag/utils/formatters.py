from __future__ import annotations

from devrag.types import DocIndexStats, IndexStats, IssueSyncStats, JiraSyncStats, PRSyncStats, SearchResult, SliteSyncStats


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
            preview = "\n".join(text_lines[:50])
            if len(text_lines) > 50:
                preview += "\n# ... (truncated)"
            lines.append(preview)
            lines.append("```")
            lines.append("")
        elif "ticket_key" in r.metadata and chunk_type in ("description", "comment"):
            ticket_key = r.metadata.get("ticket_key", "?")
            ticket_summary = r.metadata.get("ticket_summary", "")
            if chunk_type == "comment":
                comment_author = r.metadata.get("comment_author", "")
                lines.append(f"### {i}. [Jira {ticket_key}] Comment by {comment_author}")
            else:
                lines.append(f"### {i}. [Jira {ticket_key}] {ticket_summary}")
            lines.append("```")
            text_lines = r.text.strip().split("\n")
            preview = "\n".join(text_lines[:50])
            if len(text_lines) > 50:
                preview += "\n# ... (truncated)"
            lines.append(preview)
            lines.append("```")
            lines.append("")
        elif chunk_type == "slite_page":
            page_title = r.metadata.get("page_title", "Untitled")
            section_path = r.metadata.get("section_path", "")
            page_url = r.metadata.get("page_url", "")
            lines.append(f"### {i}. [Slite] {page_title}")
            if section_path:
                lines.append(f"*Section: {section_path}*")
            if page_url:
                lines.append(f"*URL: {page_url}*")
            lines.append("```")
            text_lines = r.text.strip().split("\n")
            preview = "\n".join(text_lines[:50])
            if len(text_lines) > 50:
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
            preview = "\n".join(text_lines[:50])
            if len(text_lines) > 50:
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
            preview = "\n".join(text_lines[:50])
            if len(text_lines) > 50:
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
            preview = "\n".join(text_lines[:50])
            if len(text_lines) > 50:
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
    if stats.files_empty:
        parts.append(f"{stats.files_empty} files produced no chunks")
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


def format_slite_sync_stats(stats: SliteSyncStats) -> str:
    parts = [
        f"Fetched {stats.pages_fetched} Slite pages",
        f"Indexed {stats.pages_indexed} pages ({stats.chunks_created} chunks)",
    ]
    if stats.pages_skipped:
        parts.append(f"Skipped {stats.pages_skipped} pages")
    if stats.pages_errored:
        parts.append(f"Errored {stats.pages_errored} pages")
    return ". ".join(parts) + "."


def format_jira_sync_stats(stats: JiraSyncStats) -> str:
    parts = [
        f"Fetched {stats.tickets_fetched} Jira tickets",
        f"Indexed {stats.tickets_indexed} tickets ({stats.chunks_created} chunks)",
    ]
    if stats.tickets_skipped:
        parts.append(f"Skipped {stats.tickets_skipped} tickets")
    return ". ".join(parts) + "."
