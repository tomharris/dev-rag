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

## After path normalization (Stage 1) — 2026-09-06

```
              router-fix   stage1
Precision@5        0.305    0.295
Recall@5           0.644    0.625
MRR                0.521    0.492
```

**This is corpus drift, not a regression from the change.** Path normalization
cannot affect ranking: a chunk's embedded text is `# In class X` + doc comment +
raw code, and contains no path (verified against the live index). What moved is
the corpus. The router-fix run scored a 939-chunk index built *before* Stage 0's
own new code existed; the stage1 run scores a 968-chunk index that includes
`metrics.py`, `migrations.py`, the new tests, and the edits to `query_router.py`,
`config.py` and `git.py`. Six queries moved — three down, three up — and every
file that entered a top-5 is one this session created or edited.

**Methodological caveat: this eval set indexes the repo it measures.** Editing
dev-rag moves the baseline under you. Two runs are only comparable if the index
was built from the same commit. When comparing a retrieval change, reindex once
and run both configurations against that same index — don't compare across a
reindex that pulled in new source.

What Stage 1 actually delivers is not a score:

- **PR-to-code join: 0/760 → 612/760** chunks on the live index (the remaining
  148 are files deleted or renamed since those PRs — correctly unjoinable).
- `search --file-path internal/ingest/roster.go` now returns the code *and* the
  three PRs that touched it, from one filter.
- PR/issue chunks can finally receive the active-repo boost, which compares
  against `infer_repo()`'s bare name.

## Related-chunk expansion (Stage 2) — 2026-09-06

Four configurations, one fixed index (968 dev-rag code chunks + 1249 PR chunks
from 71 synced PRs), reranking on. `devrag.jsonl` is 40 code questions;
`devrag-join.jsonl` is 16 "why did X change" questions keyed on `expected_prs`.

|                         | code P@5 | code R@5 | code MRR | join P@5 | join R@5 | join MRR |
|-------------------------|---------:|---------:|---------:|---------:|---------:|---------:|
| off (baseline)          |    0.315 |    0.669 |    0.524 |    0.288 |    0.688 |    0.688 |
| on, 5/2/10              |    0.340 |    0.665 |    0.542 |    0.338 |    0.875 |    0.740 |
| on, tightened 3/1/3     |    0.325 |    0.660 |    0.518 |    0.300 |    0.750 |    0.703 |
| on, gated to routed     |    0.315 |    0.669 |    0.524 |    0.300 |    0.688 |    0.688 |
| **on, re-seated (ship)**|**0.395** |**0.544** |**0.457** |**0.338** |**0.953** |**0.776** |

**No configuration won both sets, so expansion ships off by default** with a
per-query `--expand` opt-in. What each row taught:

- **Tightening the budget was worse on both axes.** Fewer, shallower expansions
  gave back most of the join win without protecting the code set. Rejected.
- **Gating expansion to the collections the router already chose was inert** —
  byte-identical to the baseline. The three join queries it was supposed to help
  are worded around Slack, so the router labels them `slack` and routes away
  from PR collections entirely. The router's intent labels do not track which
  queries want history, so they cannot govern expansion. Rejected.
- **Re-seating** (an expanded chunk is placed under the result that pulled it in,
  after ranking, and dropped if that anchor didn't survive) is the shipped rule.
  It fixes a real failure the unrestricted version had: for "how does the Jira
  client authenticate" the reranker put two historical diffs of
  `jira_client.py` *above* the implementation — the query asked what the code
  does and got what it used to do.

### Read the code-set numbers carefully

The apparent gain in the 5/2/10 row is largely a **metric artifact**: 8 of its 21
expansion-added results are PR *diff* chunks whose `file_path` matches an
`expected_files` entry, so `compute_metrics` credits them as if the code had been
found. Per query the same row is 1 improved / 5 regressed. Precision rising while
recall falls (0.395 vs 0.544 in the shipped row) is the same effect: expanded
chunks match the expected file but consume `final_k` slots that other expected
files needed.

The join set has no such artifact — its hits are `expected_prs`, which only a PR
chunk can satisfy — and there the result is unambiguous: **R@5 0.688 → 0.953,
MRR 0.688 → 0.776, and no query regressed.**

### What this means for the graph question

This is the file-path edge, the only edge the corpus actually has (issue and Jira
chunks carry no `file_path`). One hop over it is a decisive win on history
questions and a real cost on "how does this work" questions, and the cost is
slot competition, not bad retrieval. A richer graph would face the same
constraint: the binding limit is `final_k`, not the edges.

### Budgeting slots by query shape (the follow-up)

The Stage 2 conclusion was that the cost is *slot competition, not bad
retrieval*. That predicts a fix: spend slots on history only when the query asks
about history. Tested, same fixed index:

|                          | code P@5 | code R@5 | code MRR | join P@5 | join R@5 | join MRR |
|--------------------------|---------:|---------:|---------:|---------:|---------:|---------:|
| off                      |    0.315 |    0.669 |    0.524 |    0.288 |    0.688 |    0.688 |
| always (re-seated)       |    0.395 |    0.544 |    0.457 |    0.338 |    0.953 |    0.776 |
| **auto (shipped)**       |**0.315** |**0.669** |**0.524** |**0.338** |**0.953** |**0.776** |

**The trade-off dissolves.** `auto` is identical to `off` on the code set and
identical to `always` on the history set — the full win, at zero cost.

The signal is `QueryRouter.wants_history()`: `_PR_PATTERNS` plus a bare
`\bwhy\b`, checked **order-independently**. That last part is the whole trick.
`classify()` is first-match-wins, so "why did we throttle Slack web API calls"
is labelled `slack` and never reaches the `pr` rule — which is exactly why
gating on routed collections was inert. A separate, unordered test sees it.

Scoring the rule against the two sets: **16/16** history questions, **1/40**
false positives on code questions. The bare `\bwhy\b` is what lifts recall from
11/16 — `_PR_PATTERNS` had `why did we` / `why was` / `why were` but not
`why is` / `why are` / `why do`.

Per query, on the history set: 5 improved (4 of them from zero), 1 regressed
(MRR 0.50 → 0.25 on "why did we move BM25 into Qdrant", recall still 1.00 — the
right PR is present, just ranked lower). On the code set the gate fired once, on
"what do I need to change to add a new language" — and expansion reached **zero**
results, so nothing changed. That query already routes as `pr` anyway.

**Caveat worth keeping in view.** These two eval sets are cleanly separated by
exactly the property the gate tests, and `devrag-join.jsonl` was written as "why
did X change" questions — so this partly measures that the detector agrees with
how the set was built. What is *not* circular: the rule is a generic `\bwhy\b`
rather than anything tuned per query, and the 1/40 false-positive rate is
measured against 40 code questions written before the rule existed. Real traffic
will be messier than a clean split. `expand_max_results` exists for that case: it
caps how many expanded chunks may outrank real results, so a misfire costs at
most a couple of slots rather than the answer. `query_metrics` records how often
the gate fires in practice.

### Where this leaves the graph question

The Stage 2 note said a richer graph would hit the same `final_k` wall. It
would — but the wall is now known to be passable by *asking what the query
wants* before spending slots. Any future edge type (an AST call graph, Stage 3)
should ship with its own shape gate rather than expanding unconditionally; the
measured lesson is that edges are cheap and slots are not.
