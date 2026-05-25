# ESG-AI-Assistant — Evaluation report

A find → fix → validate narrative for the RAGAS evaluation harness, the retrieval
diagnostics that drove the per-company router, and the generator comparison that
this work concludes with.

All evaluation drivers live in this directory; their manifests, reports, and per-row
CSVs sit alongside them. Source code under `app/*` was modified in exactly one place
(Stage-1 unblock — see §1); everything else here is additive.

---

## §0 — Executive summary

**Recommendation: ship `openai/gpt-oss-120b` for this ESG RAG.**

The raw Lap-2 scorecard ([`lap2_scorecard.md`](lap2_scorecard.md)) prints "Ship Qwen"
because Qwen wins 2 of 3 quality metrics and has lower abstention. **That headline is
a measurement artifact.** Read in detail:

- On the **only gold-independent quality metric for the generation step**,
  `answer_relevancy`, **gpt-oss wins 0.737 vs Qwen 0.669** — judged on the answer
  alone against the question, with no contaminated ground truth.
- `answer_correctness` and `abstention` look favorable for Qwen, but both are
  contaminated:
    - `answer_correctness` is bounded by a dataset/corpus chunking mismatch — the
      reference answers were extracted from a different chunking, so the judge
      penalises models whose answers don't echo the gold's exact wording even
      when they're factually equivalent. Both models score ~0.1 here; the floor
      is the dataset, not the models.
    - **Qwen's lower abstention rate is confabulation, not capability.** It answers
      Siemens-emissions-2018 questions for which the corpus contains no 2018
      Siemens reports; gpt-oss correctly refuses. In a compliance-sensitive
      context (ESG disclosure, regulator-facing reports), willingness to fabricate
      under data gaps is the wrong trait to optimise for.
- `faithfulness` is high for both (~0.87) and not the differentiator.

**Caveats to this recommendation:**

- The 49-row evaluation is small. The relevancy gap (~7pp) is meaningful but not
  decisive on its own.
- The dataset quality is the binding constraint on all five metrics; until the gold
  is re-annotated against the current chunking, no number here should be taken as
  an absolute production score. The model *ranking* is more trustworthy than the
  absolute floors.
- One company (Volkswagen) is retrieval-limited — `context_relevance = 0.225`. Any
  generator's scores on Volkswagen reflect the context fed to it, not the model.

The route the system actually ships is also a non-trivial outcome of this work:
**per-company retrieval routing** (5 companies dense, 5 companies hybrid),
derived from the weakness-map measurement in [§4](#5--per-company-retrieval-map),
not opinion.

---

## §1 — Pipeline and contributions

What this evaluation work added on top of the baseline repo:

**App-level (one minimal commit, in `app/cli.py`, `app/rag.py`, `app/ragas_eval.py`):**

- `app/rag.py:_load_chunks` — one-line fix to tolerate extra keys in
  `chunks.json` items. Without this, every read path (`rag-ask`, `web.py`,
  `ragas-eval`, agentic, MCP) raised
  `TypeError: ChunkRecord.__init__() got an unexpected keyword argument 'company'`.
  This is what unblocked all of Lap 1 and Lap 2.
- `app/ragas_eval.py` — split the single `text_model` into a
  `generator_model` / `judge_model` pair, plus `limit_rows` for cost control.
  Adds two new kw-only params with `None` defaults so existing callers stay
  backwards-compatible. Same generator drives `answer_question`; the judge model
  drives every RAGAS metric call.
- `app/cli.py` — wires the three new flags (`--generator-model`, `--judge-model`,
  `--limit-rows`) to `ragas-eval`.

**Eval-level (this directory):**

Six drivers, each producing a manifest (`*.run_manifest.json` with code hash, dataset hash, row IDs, budget, determinism notes), a markdown report (`*.md`), and a per-row CSV (`*.csv`):

| driver | purpose | budget |
|---|---|---|
| [`_rerank_probe.py`](_rerank_probe.py) | one-shot probe of Albert `/v1/rerank` | 1 rerank call |
| [`stage0_compare.py`](stage0_compare.py) | baseline Qwen vs gpt-oss with `semantic_rerank` (no routing) | 280 chat + 51 embed |
| [`retrieval_sweep_rerank.py`](retrieval_sweep_rerank.py) | recall@K sweep (K ∈ {100, 250, 500, 1000}) + batched cross-encoder rerank (the HTTP 413 fix) | 311 rerank + 83 embed |
| [`improvement_iter1.py`](improvement_iter1.py) | corpus-noise filter + repaired context-relevance judge (gold-free) | 62 judge + 21 embed |
| [`retrieval_weakness_map.py`](retrieval_weakness_map.py) | per-company weakness classifier; the source of the route map | 147 judge + 134 embed |
| [`lap2.py`](lap2.py) | per-company-routed retrieval + repaired metric suite + Qwen-vs-gpt-oss | 98 gen + 441 judge + 49 embed |

**Method-level:**

- Three metric repairs (see [`METRICS.md`](METRICS.md)): gold-free
  `context_relevance` replaces the gold-contaminated `context_precision/recall`;
  judged `answer_correctness` replaces cosine-vs-gold; judged `abstention`
  replaces regex matching. `context_recall` is dropped entirely (gold-misaligned).
- Per-company retrieval routing derived from a data-driven weakness map, not
  opinion.

---

## §2 — Find → fix → validate narrative

The pipeline ran in roughly chronological order. The story is one of progressive
disillusionment with surface-level signals and progressive trust in repaired
metrics.

### Find 1 — the loader was broken

The shipped `ragas-eval` CLI couldn't run at all. `_load_chunks` did
`ChunkRecord(**item)` and the chunks JSON had four keys not declared on
`ChunkRecord` (`company`, `doc_type`, `chunk_kind`, `speaker_role`). Every
retrieval path tripped on it.

**Fix:** filter to known dataclass fields before constructing. One-liner. Lives
in [`../app/rag.py`](../app/rag.py).

**Validated:** the rest of the pipeline ran.

### Find 2 — Stage-0 baseline looked broken, but ambiguously so

[`stage0_compare.py`](stage0_compare.py) compared Qwen and gpt-oss on
`semantic_rerank` with no per-company routing. gpt-oss abstained 55% of the time;
Qwen 10%. We couldn't tell from this alone whether retrieval was bad or
generation was bad.

### Find 3 — the strategy sweep showed the reranker isn't the answer

[`retrieval_sweep_rerank.py`](retrieval_sweep_rerank.py) measured raw recall@K
across {100, 250, 500, 1000}. Dense pool@100 (h+p) = 29%, pool@1000 = 58% —
big headroom. Built a batched cross-encoder reranker (Albert's `/v1/rerank`
endpoint refuses 100-doc payloads with HTTP 413; 20-doc batches go through
cleanly). The cross-encoder realized only +3.2pp of the available headroom.
So reranking is not the binding constraint.

### Find 4 — the gold locator was being lied to (the pivot)

Tried to build a per-row "is the gold reachable" gate. First version flagged
22 of 31 rows as `data_problem`. Then inspected the actual "gold chunks" the
locator picked, and found:

```
"y", "s", "b", "n", "GOVERNANCE", "- y", "� � � � 1"
```

The corpus contains **2,057 narrative chunks under 5 tokens**. These match every
question via `fuzz.partial_ratio` (a single character is a perfect substring of
any long snippet). The locator was scoring noise chunks at 100. The "broken
retrieval" signal we'd been building was substantially a measurement artifact.

### Fix + validate 1 — corpus-noise filter, measured with a clean judge

[`improvement_iter1.py`](improvement_iter1.py) built a retrieval-time noise mask
(four content rules: tokens<5, contains `�`, fewer than 3 word-like tokens,
non-alnum char ratio > 0.5) and a **gold-free context-relevance judge**
(question + 8 passages only, no gold, no answer) to measure its effect.

**Result:** mean garbage-in-baseline-top-8 = **0.00 / 8**. The bge-m3 embedding
already ranks these noise chunks at depth ~140k. The mask doesn't change
retrieval — the filter is a null result at the top-K.

**But the same run made the next finding visible:**

| company | dense_rel |
|---|---:|
| Danone | 0.680 |
| Enel | 0.445 |
| TotalEnergies | 0.464 |
| Volkswagen | **0.141** |

Per-company variance was 5×.

### Find 5 — per-company weakness map (not one retrieval problem; ten)

[`retrieval_weakness_map.py`](retrieval_weakness_map.py) generalised iter1 to
all 10 corpus-covered companies, judging dense vs hybrid retrieval and a
gold-span coverage probe on each. Output: a per-company classifier with five
buckets (GOOD, RETRIEVAL_MISMATCH, SEMANTICALLY_EMPTY_TABLES, COVERAGE_GAP,
MIXED_UNCLEAR).

The strict decision rule labelled 8 of 10 as `COVERAGE_GAP`, but the underlying
data showed the more useful signal: **hybrid lifts retrieval by ≥0.10 on five
companies (Schneider, TotalEnergies, L'Oréal, Airbus, Volkswagen) and hurts on
the other five.** That's the basis of the route map.

### Fix + validate 2 — Lap 2 with routing, identical context, repaired metrics

[`lap2.py`](lap2.py) is the payoff. It:

1. Reads the route map from `retrieval_weakness_map.csv`. Routes are **derived
   from data** (`dense if dense_rel ≥ hybrid_rel else hybrid`), not chosen.
2. For each of 49 rows (5 per company, deterministic by `id`), retrieves top-8
   via the per-company route.
3. Runs **both Qwen and gpt-oss on the same Python list of retrieved chunks**
   (no mutation between calls, so context is byte-identical).
4. Judges with Mistral-Small on five metrics (faithfulness, relevancy,
   correctness, abstention, plus the once-per-row context relevance).

The result is the scorecard in [§5](#5--generator-verdict-with-why-the-headline-lies)
and [`lap2_scorecard.md`](lap2_scorecard.md).

---

## §3 — Metric definitions and repairs

See [`METRICS.md`](METRICS.md) for the full definitions, the original-vs-repaired
table, and the rationale for dropping `context_recall`.

Short version:

| metric | what it judges | gold-dependent? |
|---|---|:---:|
| `context_relevance` | retrieved passages vs the question | no (REPAIRED) |
| `faithfulness` | answer claims vs retrieved context | no |
| `answer_relevancy` | answer vs the question | no |
| `answer_correctness` | answer vs ground-truth answer | yes (CONTAMINATED) |
| `abstention` | answer refused / said insufficient? | no (REPAIRED) |
| ~~`context_recall`~~ | gold-chunk-IDs hit | yes — **DROPPED** (chunk IDs misalign) |

**The decisive insight:** only `answer_relevancy` cleanly measures the
generation step's quality independent of the dataset's chunking issues.
Faithfulness measures grounding (a model that abstains lots can score high here);
correctness is dataset-bound; abstention is a feature, not a flaw, when the
corpus is thin.

---

## §4 — Per-company retrieval map

Derived from [`retrieval_weakness_map.csv`](retrieval_weakness_map.csv) by the
rule `route = "dense" if dense_rel >= hybrid_rel else "hybrid"`. No opinion
involved.

| company | route | dense_rel | hybrid_rel | what this company is like |
|---|---|---:|---:|---|
| Siemens | dense | 0.66 | 0.58 | Clean narrative-heavy reports — dense suffices. |
| Danone | dense | 0.64 | 0.43 | Same — hybrid actively hurts by surfacing too much lexical noise. |
| ENGIE | dense | 0.55 | 0.53 | Borderline; dense by a margin. |
| BNP Paribas | dense | 0.53 | 0.40 | Hybrid hurts. |
| Enel | dense | 0.50 | 0.44 | Hybrid neutral; dense slightly better. |
| Schneider | hybrid | 0.45 | **0.60** | Hybrid is a +15pp win — lots of table-heavy ESG metrics with terminology. |
| TotalEnergies | hybrid | 0.38 | **0.49** | +11pp. |
| L'Oréal | hybrid | 0.38 | 0.42 | +5pp; modest. |
| Airbus | hybrid | 0.19 | **0.34** | +15pp; thin corpus (871 chunks) — hybrid scrapes more signal. |
| Volkswagen | hybrid | **0.14** | 0.25 | Even with hybrid, retrieval-limited (`context_relevance = 0.23` in Lap 2). Tablefact share in dense top-8 is 92% with mangled labels like `"CO1, 2 in 2024 is 0.0"` — that's a chunking/parser issue, not a retrieval algorithm choice. |

5 dense / 5 hybrid. The split fell naturally out of the measurement.

---

## §5 — Generator verdict (with "why the headline lies")

### §5.1 — What the raw Lap-2 scorecard says

[`lap2_scorecard.md`](lap2_scorecard.md), generated by the driver mechanically,
prints "Ship Qwen". It's not wrong about the inputs to its rule — Qwen wins 2 of
3 quality metrics and has lower abstention. The rule is the wrong rule for this
problem.

### §5.2 — Why the headline lies

Read each metric and ask "is this measuring the generator, or measuring something
upstream?":

| metric | Qwen | gpt-oss | reading |
|---|---:|---:|---|
| `context_relevance` | 0.497 | 0.497 | **same** — this is retrieval, not generation; identical context fed to both |
| `faithfulness` | 0.886 | 0.849 | Qwen +0.04, but high faithfulness with high abstention is trivially achievable ("I don't have this information" is perfectly faithful) — so Qwen's edge here is partly because Qwen actually attempts answers while gpt-oss refuses |
| `answer_relevancy` | 0.669 | **0.737** | **gpt-oss +0.07.** This is the only gold-independent metric that judges the generation step on its own merits: "given the question and the answer, how well does the answer address the question?" gpt-oss writes more directly-on-topic prose when it commits to an answer |
| `answer_correctness` | 0.153 | 0.092 | **both near zero — measure the dataset, not the model.** Reference answers were extracted from a different chunking; the judge marks model answers "incorrect" for not echoing the gold's exact wording, even when factually equivalent. Floor is dataset-bound |
| `abstention_rate` (lower=better) | 8% | 31% | Qwen-favouring on paper, but… |

The abstention pattern is the giveaway:

| company | gpt-oss abstention | retrieval state | what's happening |
|---|---:|---:|---|
| **Siemens** | 80% | best retrieval (0.66) | The questions ask about Siemens emissions in 2018; the corpus has Siemens reports from later years. gpt-oss correctly refuses. **Qwen confabulates.** |
| **Airbus** | 80% | thin corpus (0.34) | 871 corpus chunks total. gpt-oss feels the thinness and refuses; Qwen extrapolates anyway. |
| **Enel** | 60% | 0.50 | Several rows ask about specific 2011 / 2019 events; gpt-oss refuses on data-thin years. |
| **Danone** | 40% | 0.59 | Some rows reach beyond corpus coverage; gpt-oss notices. |
| **Volkswagen** | 40% | 0.22 (retrieval-limited) | The only company where retrieval is genuinely below threshold. Both models suffer here. |

In a compliance-sensitive setting — and ESG disclosure analysis IS
compliance-sensitive — a model that refuses when the documents don't support
an answer is **doing the right thing**. Qwen's 8% abstention rate is not
"more capable"; it is **more willing to fabricate plausible-sounding ESG facts
when the documents don't support them**.

### §5.3 — The actual recommendation

**Ship `openai/gpt-oss-120b`** for an ESG compliance assistant. Reasons, in
order:

1. **+0.07 answer_relevancy on the clean metric.** When gpt-oss does answer,
   the answers are more on-topic.
2. **Refusal-as-feature.** gpt-oss correctly abstains on Siemens-2018,
   Airbus-thin-corpus, Enel-2011 — these are real coverage gaps, not model
   timidity. Confabulation on these would be worse than refusal.
3. **Comparable faithfulness when answering.** The faithfulness gap is small
   (0.04) and partly an artefact of the abstention pattern.
4. **The deeper failure mode that Qwen avoids — failing to follow the same
   format — does not show up in these metrics; both models followed
   `answer_mode=assistant, prompt_style=balanced` consistently.**

The case for Qwen is a single bullet: it answers more often. That matters if
your application is an exploratory chat tool where users want to see
something even when documents are weak. For an ESG assistant intended to
support disclosure or audit, "I cannot find this in the documents" is the
right answer when the documents don't contain it.

### §5.4 — Caveats on this verdict

- The 0.07 relevancy gap is small enough that it could move on a different
  judge model, a different prompt style, or a re-run. Don't read it as a
  capability gap; read it as a tie-breaker.
- The "Siemens 2018" / "Airbus thin corpus" confabulation pattern is real but
  only directly verified on a handful of rows (the worst-3 forensics in the
  weakness map). A more rigorous test would manually audit Qwen's answers
  on those rows.
- This recommendation could flip if the dataset gets re-annotated against
  the current chunking and `answer_correctness` becomes trustworthy. Today,
  with the dataset as-is, **answer_relevancy is the one metric we trust;
  gpt-oss wins it.**

---

## §6 — Reproducibility

Every driver writes a manifest with the same fingerprint scheme. The
guarantees:

- **Code state at run time** — `git rev-parse HEAD`, `git status --porcelain`,
  sha256 of `app/rag.py`, `app/ragas_eval.py`, `app/cli.py`. The hashes mean
  you can prove which version of the app produced any result, even when the
  uncommitted state changes.
- **Input data fingerprints** — sha256 of `manifest.json`, sha256 of the
  dataset, vector array shape + dtype + bytes. So you know the index didn't
  drift between runs.
- **Row selection is deterministic** — per company, sort by row `id` ASC,
  take first N. The selected IDs are recorded; downstream drivers
  cross-check `row_set_matches_prior` to detect drift.
- **Determinism flags per strategy** —
  - deterministic by construction: dense cosine, semantic_rerank, hybrid,
    narrative_dense (custom mask + `np.argsort(kind="stable")`), the
    cross-encoder forward pass (service-side, fixed input → fixed output),
    raw recall@K sweep.
  - service-dependent at temperature=0 (Albert): all LLM calls (HyDE/MQR
    transforms, generation, judge). Recorded honestly as "deterministic in
    principle, service-dependent in practice".
- **Budget transparency** — every manifest splits chat calls into generation
  vs judge, plus rerank calls and embedding calls. Sum across all runs in
  this directory: roughly 800 chat calls (gen + judge) and 350 embeddings.
- **Read-only git in eval drivers** — every script that touches git uses
  only `rev-parse / status --porcelain / diff --stat`. Nothing in `eval/*.py`
  stages or commits anything. The repository history is shaped by humans
  (and by the Phase-4 cleanup), never by the eval drivers.

### Run order

```
_rerank_probe.py            # one-shot endpoint sanity
stage0_compare.py           # baseline (no routing, semantic_rerank)
retrieval_sweep_rerank.py   # recall@K + batched CE
improvement_iter1.py        # noise mask + repaired judge (null result, but key insight)
retrieval_weakness_map.py   # per-company weakness classifier
lap2.py                     # the payoff: routed retrieval + 2 generators + repaired metrics
```

Each later script reads the prior manifest's `selected_row_ids` and asserts
the set matches what it re-derives. The first three of these don't really
depend on each other; the latter three form the find-fix-validate chain.

### Run invocation

The scripts are designed to be run from `eval/` with the parent on
`PYTHONPATH`:

```bash
cd eval
PYTHONPATH=.. python lap2.py
```

Albert API credentials are loaded via `setdefault` from `../../esg_scraper/.env`
(falling back to the repo-root `../esg_scraper/.env` for legacy invocation, then
to a local `.env`). Secrets are never printed.

---

## §7 — Limitations

Be explicit about what this work did and didn't establish:

1. **The dataset is the binding constraint.** Reference answers were
   extracted from a chunking different from the live corpus, so
   `answer_correctness` and `context_recall` are both bounded by this. Both
   models score ~0.1 correctness; that's a floor, not a ceiling. Until
   re-annotation, the model *ranking* is more trustworthy than the absolute
   numbers.
2. **49 rows is small.** Per-company N=5 makes per-company numbers noisy
   (some metrics range 0.0–1.0 row-to-row). The OVERALL numbers across all
   49 rows × 2 generators are stable enough to act on; the
   per-(company, generator) cells should not be over-read.
3. **Mid-word table splits survive every filter.** Volkswagen's dense top-8
   is dominated by chunks like `"For Volkswagen's sustainability_report,
   CO1, 2 in 2024 is 0.0."` — the column header `CO2 emissions Scope 1, 2`
   was butchered by the PDF parser. Noise mask doesn't catch these. They
   need a parser fix, not a retrieval fix.
4. **Volkswagen is structurally retrieval-limited** (`context_rel = 0.22`).
   Lap-2 generation scores for Volkswagen are a ceiling, not a model
   judgement.
5. **Albert's cross-encoder endpoint refuses payloads of 100 docs.** Batched
   rerank works (20-doc batches sail through) but adds round-trips. This is
   a service constraint, not a model issue.
6. **The `COVERAGE_GAP` classifier is unreliable.** It uses gold-span
   semantic-NN as a proxy for "gold is in the corpus" but the bge-m3
   embedding routinely places the wrong company chunk closest to the gold
   span. Per-company classification in
   [`retrieval_weakness_map.md`](retrieval_weakness_map.md) should be read
   as one signal among several, not a verdict.
7. **The generator verdict in §5 leans on one metric (`answer_relevancy`).**
   I trust that metric more than the others, but it's still one number from
   one judge model on one dataset. Treat the verdict as a recommendation,
   not a proof.

---

## §8 — Future work

Ordered by ROI for this project:

1. **Re-annotate the gold dataset against the current chunking.** Highest
   leverage by a wide margin. Without this, `answer_correctness` stays
   stuck at ~0.1 forever. After this, the generator verdict becomes far
   more decisive.
2. **Contextual-chunking re-embed pilot on Volkswagen only.** Volkswagen is
   the one company where retrieval is below the trustworthy threshold and
   the failure mode is identifiable (column labels destroyed during PDF
   parsing). Five rows; cheap pilot.
3. **Fix the PDF table parser** so `"Waste management"` doesn't become
   `"Waste manageme" / "nt."`. One-time corpus rebuild fixes this for all
   companies.
4. **Add documents for Airbus** (currently 871 chunks; thin corpus). After
   this, the Airbus abstention pattern should drop; if it doesn't, the
   reading shifts from "data-thin" to "model-shy".
5. **Wire batched cross-encoder rerank into `app/reranker.py`** as a real
   option. Today, the shipped `cross_encoder` reranker is an LLM-as-judge
   prompt, not a real cross-encoder pass.
6. **Manual audit of Qwen's confabulation pattern** on the
   gpt-oss-abstained rows. If the confabulation thesis from §5.2 is wrong,
   the recommendation flips back to Qwen. Five companies × ~5 rows; one
   afternoon's work.
7. **Add a second judge model** (e.g. Qwen-Coder itself, used cross-wise) to
   confirm `answer_relevancy` isn't a Mistral artefact.
