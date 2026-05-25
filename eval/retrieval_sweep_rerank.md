# Retrieval Sweep + Batched Rerank

- Index: `outputs/rag_index` (chunks=145197, vectors=(145197,1024) float32)
- Dataset: `sample_data/ragas_esg_eval_dataset.csv`  rows evaluated = 31 (8/company for 4 companies; TotalEnergies has only 7)
- Reachable: **31/31** (100%)  row-set-vs-prior: **True**
- Embedding model: `BAAI/bge-m3`  alignment: positional (len-verified + retrieved-ids-subset-checked, from prior run)
- Rerank: model=`BAAI/bge-reranker-v2-m3` endpoint=`https://albert.api.etalab.gouv.fr/v1/rerank` probe_ok=True

## Pool-recall sweep (raw full-corpus cosine; FREE)

| K | recall_hit% (≥90 in top-K) | recall (h+p)% (≥70 in top-K) |
|---|---|---|
| 100 | 0.0% | 29.0% |
| 250 | 0.0% | 41.9% |
| 500 | 0.0% | 51.6% |
| 1000 | 3.2% | 58.1% |

### Rank distribution of first partial-hit (reachable rows only)

| bucket | count |
|---|---|
| 1-8 | 3 |
| 9-100 | 6 |
| 101-250 | 4 |
| 251-500 | 3 |
| 501-1000 | 2 |
| 1001-inf | 13 |

### Cross-check vs prior report card's dense pool@100 (~29%)

- This sweep recall_ph@100 = **29.0%**  (hit@100 = 0.0%)
- Prior dense pool@100 (h+p) = ~29.0%
- **MATCH** within tolerance. The app's dense path appears to be plain top-K cosine (no dedup/cap/MMR observed).

## Realized rerank (batched cross-encoder over CE_POOL_K=100)

| strategy | base top-8 (h+p)% | pool@100 ceiling (h+p)% | CE top-8 (h+p)% realized | gain (pp) | modal batch | truncated | ce_failed |
|---|---|---|---|---|---|---|---|
| dense_ce | 9.7% | 29.0% | 12.9% | +3.23 | 20 | 0 | 0 |
| narrative_ce | 0.0% | 29.0% | 3.2% | +3.23 | 20 | 0 | 0 |

## Verdicts

1. **Recall climbs with K** (+29.0% from K=100 to K=1000) — gold is often just below the top-100 cutoff. A wider candidate pool helps.
2. **Dense cross-encoder barely moves the needle** (+3.2pp of 19.4pp available). Either the headroom isn't real (the partial matches in the pool aren't truly relevant) or the reranker doesn't help on table-heavy chunks.

_Manifest: outputs/retrieval_sweep_rerank.run_manifest.json_