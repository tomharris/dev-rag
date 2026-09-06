from unittest.mock import MagicMock

from devrag.ingest.migrations import backfill_bare_repo_name


def test_backfill_rewrites_slug_to_bare_name():
    store = MagicMock()
    store.set_payload.return_value = 7
    updated = backfill_bare_repo_name(store, ["pr_diffs", "pr_discussions"], "acme/backend")
    assert updated == 14
    coll, where, payload = store.set_payload.call_args_list[0].args
    assert coll == "pr_diffs"
    assert where == {"repo": "acme/backend"}
    assert payload == {"repo": "backend", "repo_full": "acme/backend"}


def test_backfill_is_a_noop_for_an_already_bare_name():
    store = MagicMock()
    assert backfill_bare_repo_name(store, ["pr_diffs"], "backend") == 0
    store.set_payload.assert_not_called()


def test_backfill_survives_a_store_error():
    """A migration must never block the sync it precedes."""
    store = MagicMock()
    store.set_payload.side_effect = RuntimeError("collection missing")
    assert backfill_bare_repo_name(store, ["pr_diffs"], "acme/backend") == 0
