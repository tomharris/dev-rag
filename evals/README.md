# Retrieval eval sets

Ground truth for `devrag eval run <file> --output <results>`.

## Why this exists

Stage 0 of the graph-search investigation. The published evidence on graph
retrieval is consistent about *where* it helps — roughly +10 points on
multi-hop and architectural questions, and a wash on single-hop lookups — so
the only way to decide whether DevRAG should grow a graph layer is to know
which of those two shapes it is currently failing at. That needs a labeled
baseline. Before this, `devrag eval` had nothing to run.

## Case format

One JSON object per line:

| field | meaning |
|---|---|
| `query` | the query text, phrased the way a user would actually type it |
| `expected_files` | repo-relative paths a correct answer must surface |
| `expected_prs` | PR numbers a correct answer must surface |
| `hop_type` | `single` (answer lives in one file) or `multi` (needs 2+ files connected) |
| `filters` | metadata filters passed through to search, as `devrag search --repo` would |
| `note` | free text; not used by the harness |

Paths are written **repo-relative** even though `code_indexer` currently stores
absolute paths. `devrag/eval.py:match_expected_file` matches on whole path
segments in either direction, so these cases keep working both today and after
paths are normalized in the index.

## devrag.jsonl

40 cases against this repo, 20 `single` and 20 `multi`, each filtered to
`repo: dev-rag`. The filter is deliberate: it holds index composition constant
so a score change is attributable to ranking rather than to which other repos
happen to be indexed that week. Measuring cross-repo noise is a separate
experiment — drop the filters to run it.

One case ("what do I need to change to add a new language…") is kept in its
natural phrasing even though the router's `change to` pattern sends it to PR
collections only. It is a routing failure, not a ranking one, and the
`classification` field in the results file is what distinguishes the two.

## Running

Query sets are committed; run outputs go to `evals/runs/`, which is gitignored
(they embed absolute local paths).

```bash
devrag eval run evals/devrag.jsonl --output evals/runs/baseline.jsonl
devrag eval compare evals/runs/baseline.jsonl evals/runs/after-change.jsonl
```

Reranking, `top_k` and `final_k` come from config, so record what they were
when comparing runs. The active-repo boost is **off** by default here
(`--prefer-repo` enables it) because it depends on the cwd and would make runs
non-reproducible.

## Baseline — 2026-09-05

`devrag/` reindexed (939 code + 431 doc chunks), reranking on, `top_k: 20`,
`final_k: 5`, `max_per_source: 2`, repo boost off.

```
Precision@5: 0.270   Recall@5: 0.581   MRR: 0.456

By hop_type:
  multi   n=20  P@5=0.330  R@5=0.512  MRR=0.471
  single  n=20  P@5=0.210  R@5=0.650  MRR=0.442
```

Precision is capped by construction: with a single-file expectation and
`max_per_source: 2`, P@5 cannot exceed 0.4. Track recall and MRR.

**Multi-hop and single-hop score about the same (MRR 0.471 vs 0.442).** That is
the opposite of the profile that justifies a graph layer — the literature's case
for graphs rests on multi-hop being the weak half. On this evidence the current
ranking is not failing specifically at multi-hop, so the Stage 3 graph work has
no measured deficit to close yet.

What is failing is **routing**, not ranking. Mean MRR by intent label:

| intent | n | MRR |
|---|---|---|
| slack | 4 | 0.750 |
| code | 15 | 0.633 |
| unrouted | 12 | 0.396 |
| doc | 1 | 0.333 |
| usage | 3 | 0.222 |
| issue / jira / slite / pr | 5 | 0.000 |

Five of the thirteen zero-MRR queries never had a chance, because the router
sent them somewhere the answer could not be:

1. **`\bfiled?\b` in `_ISSUE_PATTERNS` matches the word "file".** "how does
   incremental indexing decide to skip a file" and "how are per-file failures
   isolated…" both route to issue collections only and return **zero results**.
   "file" is among the most common words in a code question.
2. **The `slite` intent routes to `slite_pages` + `documents` with no code**,
   unlike the `slack` and `jira` intents which were deliberately widened to
   include `code_chunks`. "how is a Slite page fetched as markdown" cannot reach
   `slite_client.py`.
3. **The `pr` intent routes to PR collections only**, with no code fallback —
   "what do I need to change to add a new language" matches on `change to`.
4. **`unrouted` (12 of 40 cases) fans out to all 11 collections** and scores
   0.396, well under the 0.633 of queries that reach `code_chunks` directly.

The `usage` intent — "where is X defined / used", exactly the shape an
AST call graph would serve — scores 0.222 over 3 cases. That is the one signal
pointing toward Stage 3, but 3 cases is too thin to act on; widen that slice
before drawing a conclusion.

Reranking is **~87% of query latency** (850ms of 976ms on a warm run,
`query_metrics`), which is the cost baseline any added retrieval stage has to
be measured against.

## After the router fixes — 2026-09-05

Same index, same config; only `query_router.py` changed.

```
              baseline   router-fix
Precision@5      0.270      0.305
Recall@5         0.581      0.644
MRR              0.456      0.521

multi   MRR      0.471      0.588    R@5  0.512 -> 0.613
single  MRR      0.442      0.454    R@5  0.650 -> 0.675
```

Exactly four queries moved, all of them previously-broken ones, and nothing
regressed. **Queries returning zero results: 4 → 0.**

| query | baseline → fix | MRR |
|---|---|---|
| "what do I need to change to add a new language…" | `pr` → `pr`+code | 0.00 → 1.00 |
| "how are per-file failures isolated…" | `issue` → `unrouted` | 0.00 → 1.00 |
| "how does incremental indexing decide to skip a file" | `issue` → `code` | 0.00 → 0.33 |
| "how is a Slite page fetched as markdown" | `slite` → `slite`+code | 0.00 → 0.25 |

The multi-hop gain (+0.117 MRR) is larger than the single-hop one (+0.012), but
it is **not** evidence about multi-hop reasoning: three of the four repaired
queries happen to be tagged `multi`. It is a routing repair showing up in a
20-case bucket, not a retrieval-quality change.

### Still open

- **`unrouted` is now the largest bucket** (13 of 40) at MRR 0.442, against
  0.615 for queries that reach `code_chunks` directly. Fanning out to all 11
  collections shares one `top_k: 20` candidate pool across all of them, so code
  competes with Slack and session noise. This is a *ranking* question, not a
  routing one — the honest fix is a per-collection candidate budget or a larger
  `top_k` on the fan-out path, and it needs its own measurement.
- **`usage` remains the weakest real intent** (MRR 0.222, n=3) — "where is X
  defined / used" is exactly the shape an AST call graph serves, and it is the
  one signal still pointing at Stage 3. Widen that slice before acting on it.
- `_DOC_PATTERNS` matches bare `\bprocess\b`, `\bstandard\b` and
  `\bconvention\b`, which are ordinary code words. It is not a zero-result bug
  (the doc intent includes code), so it was left alone.
