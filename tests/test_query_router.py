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


def test_docs_only_scope():
    assert QueryRouter().route("anything", scope="docs") == ["documents"]


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
