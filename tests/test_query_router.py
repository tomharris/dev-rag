from devrag.retrieve.query_router import QueryRouter


def test_code_query():
    router = QueryRouter()
    assert router.route("how does the auth middleware work") == ["code_chunks"]


def test_pr_query_why():
    router = QueryRouter()
    collections = router.route("why did we switch from JWT to PKCE")
    assert "pr_diffs" in collections
    assert "pr_discussions" in collections


def test_pr_query_change():
    router = QueryRouter()
    assert "pr_diffs" in router.route("when did we change the database schema")


def test_pr_query_migrate():
    router = QueryRouter()
    collections = router.route("why did we migrate to Redis")
    assert "pr_diffs" in collections
    assert "pr_discussions" in collections


def test_usage_query():
    router = QueryRouter()
    collections = router.route("where is refreshToken used")
    assert "code_chunks" in collections
    assert "pr_diffs" in collections


def test_ambiguous_query():
    router = QueryRouter()
    collections = router.route("tell me about authentication")
    assert "code_chunks" in collections
    assert "pr_diffs" in collections
    assert "pr_discussions" in collections


def test_code_only_scope():
    assert QueryRouter().route("how does auth work", scope="code") == ["code_chunks"]


def test_prs_only_scope():
    assert set(QueryRouter().route("how does auth work", scope="prs")) == {"pr_diffs", "pr_discussions"}


def test_docs_query_policy():
    router = QueryRouter()
    assert "documents" in router.route("what is our API versioning policy")


def test_docs_query_architecture():
    router = QueryRouter()
    assert "documents" in router.route("describe the system architecture")


def test_docs_query_spec():
    router = QueryRouter()
    assert "documents" in router.route("what does the design spec say about caching")


def test_docs_query_architecture_includes_code():
    router = QueryRouter()
    collections = router.route("describe the system architecture")
    assert "documents" in collections
    assert "code_chunks" in collections


def test_docs_only_scope():
    assert QueryRouter().route("anything", scope="docs") == ["documents"]


def test_docs_only_scope_stays_narrow():
    """Explicit docs scope must not be widened to include code."""
    assert QueryRouter().route("describe the system architecture", scope="docs") == ["documents"]


def test_ambiguous_query_includes_docs():
    router = QueryRouter()
    assert "documents" in router.route("tell me about authentication")


def test_issue_query_bug():
    router = QueryRouter()
    collections = router.route("is there a bug with login")
    assert "issue_descriptions" in collections
    assert "issue_discussions" in collections


def test_issue_query_filed():
    router = QueryRouter()
    collections = router.route("was a ticket filed for this")
    assert "issue_descriptions" in collections


def test_issues_only_scope():
    assert set(QueryRouter().route("anything", scope="issues")) == {"issue_descriptions", "issue_discussions"}


def test_all_collections_include_issues():
    from devrag.retrieve.query_router import ALL_COLLECTIONS
    assert "issue_descriptions" in ALL_COLLECTIONS
    assert "issue_discussions" in ALL_COLLECTIONS


def test_slack_only_scope():
    assert QueryRouter().route("anything", scope="slack") == ["slack_messages"]


def test_slack_query_keyword():
    router = QueryRouter()
    assert "slack_messages" in router.route("what was discussed in slack about the deploy")


def test_slack_query_includes_code():
    """A Slack-named code subject (e.g. 'slack client') must still reach code."""
    router = QueryRouter()
    collections = router.route("slack client")
    assert "slack_messages" in collections
    assert "code_chunks" in collections


def test_jira_query_includes_code():
    """A Jira-named code subject (e.g. 'jira client') must still reach code."""
    router = QueryRouter()
    collections = router.route("jira client implementation")
    assert "jira_descriptions" in collections
    assert "code_chunks" in collections


def test_all_collections_include_slack():
    from devrag.retrieve.query_router import ALL_COLLECTIONS
    assert "slack_messages" in ALL_COLLECTIONS


def test_session_discussed_still_routes_to_sessions():
    """The slack intent must not steal the existing 'we discussed' session routing."""
    assert QueryRouter().route("what did we discuss last session") == ["session_logs"]


def test_classify_labels_intents():
    router = QueryRouter()
    assert router.classify("how does the auth middleware work") == "code"
    assert router.classify("where is refreshToken used") == "usage"
    assert router.classify("why did we switch from JWT to PKCE") == "pr"
    assert router.classify("describe the system architecture") == "doc"
    assert router.classify("what did we discuss last session") == "session"


def test_classify_unrouted_for_ambiguous_query():
    assert QueryRouter().classify("tell me about authentication") == "unrouted"


def test_classify_reports_explicit_scope():
    assert QueryRouter().classify("how does auth work", scope="code") == "scope:code"


def test_classify_agrees_with_route():
    """classify() and route() must never disagree about which rule matched."""
    from devrag.retrieve.query_router import _COMPILED_RULES
    router = QueryRouter()
    by_name = {name: colls for name, _, colls in _COMPILED_RULES}
    for query in ["how does auth work", "where is X used", "why did we migrate to Redis",
                  "describe the system architecture", "is there a bug with login",
                  "what was discussed in slack", "jira epic status"]:
        label = router.classify(query)
        assert router.route(query) == by_name[label], query


def test_route_returns_a_copy_callers_cannot_corrupt():
    router = QueryRouter()
    router.route("how does auth work").append("junk")
    assert router.route("how does auth work") == ["code_chunks"]


def test_issue_pattern_does_not_match_the_word_file():
    """'file' is one of the commonest words in a code question.

    `\\bfiled?\\b` matched it, routing "…skip a file" to issue collections only —
    which returned zero results.
    """
    router = QueryRouter()
    assert router.route("how does incremental indexing decide to skip a file") == ["code_chunks"]
    assert router.classify("how are per-file failures isolated") != "issue"


def test_issue_pattern_does_not_match_the_word_report():
    router = QueryRouter()
    assert router.classify("how does the report generator work") != "issue"


def test_issue_pattern_still_matches_filed_and_file_a_bug():
    router = QueryRouter()
    assert router.classify("was a ticket filed for this") == "issue"
    assert router.classify("where do I file a bug") == "issue"
    assert router.classify("who reported this") == "issue"


def test_source_intents_all_include_code():
    """A query naming a source usually still means the code behind it."""
    router = QueryRouter()
    for query in ["how is a Slite page fetched", "jira client implementation",
                  "was a ticket filed for this", "why did we migrate to Redis",
                  "slack client", "describe the system architecture"]:
        assert "code_chunks" in router.route(query), query


def test_session_intent_stays_code_free():
    """A question about a past conversation is not answered by code."""
    assert QueryRouter().route("what did we discuss last session") == ["session_logs"]


def test_explicit_scope_still_narrows_to_one_source():
    router = QueryRouter()
    assert router.route("how is a Slite page fetched", scope="slite") == ["slite_pages"]
    assert set(router.route("why did we migrate", scope="prs")) == {"pr_diffs", "pr_discussions"}


def test_wants_history_catches_why_questions_the_intent_table_mislabels():
    """Intent is first-match-wins, so a Slack-worded history question is labelled
    `slack` and never reaches the `pr` rule — but it still wants history."""
    router = QueryRouter()
    assert router.classify("why did we throttle Slack web API calls") == "slack"
    assert router.wants_history("why did we throttle Slack web API calls")


def test_wants_history_covers_why_is_are_do_forms():
    """_PR_PATTERNS only had `why did we` / `why was` / `why were`."""
    router = QueryRouter()
    for q in ["why is python pinned below 3.14", "why are Qdrant upserts batched",
              "why do we index repo docs alongside code",
              "when did we change the database schema", "why did we migrate to Redis"]:
        assert router.wants_history(q), q


def test_wants_history_is_false_for_how_and_where_questions():
    router = QueryRouter()
    for q in ["how does the embedder handle blank input", "where is the encoder defined",
              "what fields does the Chunk dataclass have",
              "how are markdown documents split into sections"]:
        assert not router.wants_history(q), q
