# Metric definitions and repairs

Every metric below is computed by [`lap2.py`](lap2.py) on every (row, generator)
pair, except `context_relevance` which is computed once per row (generator-free)
and `context_recall` which is dropped from this evaluation entirely.

The judge for every judge call is
`mistralai/Mistral-Small-3.2-24B-Instruct-2506`. Generation is done by either
`Qwen/Qwen3-Coder-30B-A3B-Instruct` or `openai/gpt-oss-120b`. The generator
never judges its own work.

---

## Original vs repaired summary

| metric | original (from `app/ragas_eval.py`) | what's wrong | repaired form |
|---|---|---|---|
| `context_relevance` (a.k.a. context_precision) | judge with `Q + answer + retrieved chunks`; "is each chunk relevant?" | the answer leaks into the judgement — a thorough answer makes irrelevant chunks look relevant | **Q + 8 passages only.** No answer, no gold. Per-passage verdict ∈ `{relevant, partially_relevant, not_relevant}`. Score = (#rel + 0.5·#partial)/8. Computed once per row (generator-independent). |
| `faithfulness` | judge with `Q + answer + retrieved chunks`; "is each answer claim supported by the context?" | sound as-is; reused unchanged | same |
| `answer_relevancy` | judge with `Q + answer`; "does the answer address the question?" | sound as-is; reused unchanged | same |
| `answer_correctness` | embedding cosine between answer and ground-truth answer | high cosine on similar topics doesn't mean factually correct; embeddings are biased toward shared vocabulary | **judge** with `Q + ground_truth + model_answer`; "is the answer factually consistent with the reference? `correct (1.0) / partially_correct (0.5) / incorrect (0.0)`". `reason` recorded. |
| `abstention` | regex match: `r"\bI cannot\b\|\bnot available\b\|\binsufficient\b\|\bunable\b"` | misses paraphrases; trivially gamed | **judge** with `Q + answer`; "did the answer abstain — refuse, say insufficient/unavailable — rather than attempt? `yes / no`". `abstain=yes` overrides `answer_correctness → 0` (floored, not averaged). |
| `context_recall` | gold chunk-ID intersection in retrieved top-K | dataset's `chunk_id` field uses naming `chunk-NNNNN`; the live corpus uses `year_kind_hash_pdf_nNNNN`. Cannot be located reliably in the live index. | **DROPPED.** Recorded in the Lap-2 manifest's `dropped_metrics` list with this rationale. The trustworthy retrieval signal is `context_relevance` (gold-free). |

---

## Detail per metric

### `context_relevance` — repaired

**Purpose:** measure retrieval quality (how relevant are the top-8 passages to
the question?) WITHOUT contaminating the measurement with what was eventually
generated or with the gold answer.

**Prompt (verbatim from `lap2.py:CONTEXT_RELEVANCE_INSTRUCTION`):**

> You are a retrieval-quality judge. For each numbered passage, decide whether
> it helps answer the question:
> - "relevant": directly provides information that answers the question
> - "partially_relevant": related to the question but missing key details
> - "not_relevant": unrelated, contentless, or noise
>
> Then judge whether the question can be answered using these passages as a
> whole ("yes" | "partial" | "no"). If answerable != "yes", add a one-sentence
> "why_not" explaining what's missing.
>
> Output STRICT JSON ONLY:
> `{"passages":[{"index":<int>,"verdict":"..."}],"answerable":"...","why_not":"..."}`

**Aggregation:** `score = (#relevant + 0.5 * #partially_relevant) / n_passages`,
rounded to 4 dp. Computed once per row.

**Lap 2 corpus-wide mean:** **0.497** (range across companies: 0.225 - 0.662).

**Why it's the trustworthy retrieval signal:** independent of the answer step
(no contamination by what the model wrote), independent of the gold (not
contaminated by the dataset's chunking mismatch), and per-passage
fine-grained rather than a binary chunk-ID intersection.

---

### `faithfulness` — reused, unchanged

**Purpose:** for each claim in the model's answer, is it supported by the
retrieved context?

**Implementation:** uses `app.ragas_eval.score_faithfulness` directly,
parameterised by the Mistral judge. Returns `score = #supported_claims /
#total_claims`.

**Lap 2 means:** Qwen **0.886** | gpt-oss **0.849**.

**Caveat:** a model that abstains can score artificially high on
faithfulness — "I cannot find this in the documents" is faithful by
construction. The faithfulness gap between Qwen and gpt-oss is therefore
not pure capability; it's confounded by gpt-oss's higher abstention.

---

### `answer_relevancy` — reused, unchanged

**Purpose:** does the model's answer address the question?

**Implementation:** uses `app.ragas_eval.score_answer_relevancy` directly,
parameterised by the Mistral judge. The prompt asks the judge to grade
relevance 0.0-1.0 in 0.1 increments.

**Lap 2 means:** Qwen **0.669** | gpt-oss **0.737**.

**Why this is decisive in §5 of the README:** of the five metrics, this one
is the most directly about *generation quality* and the least contaminated by
upstream issues:

- It doesn't use the gold (so dataset chunking misalignment doesn't enter).
- It doesn't use the retrieved context (so retrieval quality doesn't enter
  — the same retrieved context was passed to both generators anyway).
- It judges the answer's relevance to the question, which is exactly what
  a user cares about.
- Abstention costs you here (an abstaining answer is not relevant) but
  doesn't artificially help.

---

### `answer_correctness` — repaired (judged, not cosine)

**Purpose:** is the answer factually consistent with the dataset's reference
answer?

**Prompt (verbatim from `lap2.py:CORRECTNESS_INSTRUCTION`):**

> You are judging answer correctness against a reference (gold) answer. Is the
> model's answer factually consistent with, AND does it cover the key facts of,
> the reference? The reference may be a partial excerpt — judge factual
> consistency, NOT verbatim overlap.
> - "correct": factually consistent AND covers the key facts → score 1.0
> - "partially_correct": factually consistent on some points but missing or
>   vague on others → 0.5
> - "incorrect": contradicts or misses the key facts → 0.0
>
> Output STRICT JSON ONLY:
> `{"verdict":"...","score":1.0|0.5|0.0,"reason":"<one sentence>"}`

**Abstention floor:** if `abstention.abstain == "yes"`, `answer_correctness` is
overridden to 0.0 regardless of the judge verdict. Recorded in
`correctness_floored_by_abstain`.

**Lap 2 means:** Qwen **0.153** | gpt-oss **0.092**.

**Why both are so low — and why this number doesn't mean what it looks like:**
the dataset's `ground_truth_answer` field was extracted from a different
chunking of the same documents. The verbatim sentences the judge expects often
aren't in the chunks the live retriever surfaces, even when both exist in the
underlying PDF. The judge marks an answer "incorrect" if it doesn't echo the
same facts as the gold — but a factually correct answer from a different
sentence in the same document gets the same penalty.

**This metric is dataset-bound, not model-bound.** The relative ranking
(Qwen +0.06) might be real but is dominated by the gpt-oss abstention floor
penalty (every abstained row contributes 0 to gpt-oss's average). If you
re-annotate the dataset against the current chunking, expect both numbers to
roughly double; the ranking could flip.

---

### `abstention` — repaired (judged, not regex)

**Purpose:** did the model refuse to answer / say the information is
insufficient or unavailable, rather than attempting an answer?

**Prompt (verbatim from `lap2.py:ABSTENTION_INSTRUCTION`):**

> You are judging whether an answer ABSTAINED instead of attempting to answer.
> Did the answer refuse, say the information is not available / not provided
> / insufficient / cannot be determined, rather than attempt a substantive
> answer? A short but substantive answer that picks a stance is NOT abstaining.
> An answer that mostly says "I cannot find this in the documents",
> "no information available", "insufficient data", "unable to determine" IS
> abstaining.
>
> Output STRICT JSON ONLY:
> `{"abstain":"yes|no","reason":"<one sentence>"}`

**Output:** binary per row, aggregated per company and per generator as a rate.

**Lap 2 means:** Qwen **8%** | gpt-oss **31%**.

**Why "lower is better" is the wrong framing for compliance:** abstaining when
the documents don't support an answer is the correct behaviour for an ESG
disclosure assistant. See README §5.2 for the per-company pattern (gpt-oss
abstains heavily on Siemens-2018, Airbus-thin-corpus, Enel-2011 — all real
coverage gaps).

**The double-penalty design:** abstention floors `answer_correctness` to 0.
This means abstaining costs the model on the correctness metric even when
abstention is the right answer. We accept this so that the correctness number
doesn't reward abstainers for "perfect faithfulness on no claims".

---

### `context_recall` — DROPPED

**Why dropped:** the dataset's `ground_truth.chunk_id` references chunk
identifiers like `chunk-13166`, `chunk-14193`, etc. The live corpus uses
identifiers like `2024_urd_c0294fa9_pdf_n0218`. These naming schemes don't
overlap; the gold chunk IDs cannot be located in the current index.

We explored fuzz-based gold-text location ("find the corpus chunk most similar
to the dataset's gold span") in [`improvement_iter1.py`](improvement_iter1.py)
and [`retrieval_weakness_map.py`](retrieval_weakness_map.py). The result: too
unreliable to gate decisions on — the semantic-NN of a gold span is routinely
*not* the gold chunk itself when the corpus contains other related text.

Recorded in `lap2.run_manifest.json` under `dropped_metrics`:

```json
{"metric": "context_recall",
 "reason": "Gold-misaligned: dataset chunk_id naming scheme (chunk-NNNNN)
            doesn't match the live corpus (year_kind_hash_pdf_nNNNN), so
            chunk-level recall over the current index is unreliable.
            Replaced by gold-free context_relevance (Stage 2) for retrieval
            health."}
```

**The replacement for retrieval recall** is `context_relevance` (above) plus the
recall@K sweep done once for diagnostic purposes in
[`retrieval_sweep_rerank.py`](retrieval_sweep_rerank.py) (dense pool@100 = 29%
(h+p), pool@1000 = 58%).

---

## Aggregation conventions

- **Per-row scores** range 0.0–1.0 (or 0/1 for abstention).
- **Per-(company, generator)** means are simple arithmetic means over the
  rows for that pair.
- **OVERALL per generator** is the arithmetic mean across all rows for that
  generator (49 rows).
- **OVERALL across all rows × generators** is the mean of 98 rows.
- **Abstention rate** is `#abstained / #rows` per (company, generator).
- **Context relevance** is computed once per row (gen-independent) and
  reported as a single per-company column in the scorecard.

---

## What we did NOT measure

For honesty:

- **Hallucination rate against external ground truth** (not the dataset's
  gold) — would require manual auditing of the 98 (row, generator) answers.
  Out of scope.
- **Latency / cost per query** — these are not retrieval-quality or
  generation-quality metrics. The budget block in each manifest captures
  call counts but not wall-clock.
- **Multi-document reasoning** (questions that synthesise across companies)
  — the dataset's `v2_retrieval` subset has cross-document questions but
  Lap 2 used only the single-doc rows.
- **End-to-end RAGAS metrics from the `ragas` package** — we reused the
  individual scoring helpers (`score_faithfulness`, `score_answer_relevancy`)
  but did not run the full RAGAS evaluator. The repaired suite above is
  Lap-2-specific.
