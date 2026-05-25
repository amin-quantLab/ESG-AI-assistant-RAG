# Stage 0 - Generator comparison (Qwen vs gpt-oss)

- Retrieval: `semantic_rerank` (candidate_k=100, top_k=8)
- Answer config: answer_mode=`assistant`, prompt_style=`balanced`
- Companies: Danone, Enel, TotalEnergies, Volkswagen; rows/company=5; judge=`mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- Metric means are **micro-averaged (pooled rows)**; non-answered rows count as 0.

## Primary table (Qwen | gpt-oss)

| Company | faithfulness | answer_relevancy | context_precision | context_recall | answer_correctness | f1 |
|---|---|---|---|---|---|---|
| Danone | 0.804 \| 0.809 | 0.620 \| 0.660 | 0.825 \| 0.600 | 0.200 \| 0.200 | 0.617 \| 0.618 | 0.200 \| 0.154 |
| Enel | 0.884 \| 0.927 | 0.580 \| 0.740 | 0.700 \| 0.450 | 0.300 \| 0.300 | 0.550 \| 0.515 | 0.274 \| 0.261 |
| TotalEnergies | 0.923 \| 0.959 | 0.800 \| 0.840 | 0.400 \| 0.375 | 0.378 \| 0.400 | 0.566 \| 0.554 | 0.259 \| 0.242 |
| Volkswagen | 0.811 \| 0.823 | 0.700 \| 0.640 | 0.550 \| 0.225 | 0.200 \| 0.200 | 0.572 \| 0.545 | 0.044 \| 0.044 |
| **OVERALL (mean)** | **0.855 \| 0.880** | **0.675 \| 0.720** | **0.619 \| 0.412** | **0.269 \| 0.275** | **0.576 \| 0.558** | **0.194 \| 0.175** |

## Extended diagnostics (OVERALL row only, Qwen | gpt-oss)

| hallucination_free | citation_coverage | unsupported_claims_score | n_answered/n_rows | n_refusal |
|---|---|---|---|---|
| 0.250 \| 0.350 | 0.000 \| 0.000 | 0.442 \| 0.368 | 20/20 \| 20/20 | 2 \| 11 |

## Winners (OVERALL, primary metrics)

- **faithfulness**: gpt-oss by 0.024 (0.855 vs 0.880)
- **answer_relevancy**: gpt-oss by 0.045 (0.675 vs 0.720)
- **context_precision**: Qwen by 0.206 (0.619 vs 0.412)
- **context_recall**: gpt-oss by 0.006 (0.269 vs 0.275)
- **answer_correctness**: Qwen by 0.018 (0.576 vs 0.558)
- **f1**: Qwen by 0.019 (0.194 vs 0.175)

> **REFUSAL CALLOUT**: gpt-oss refused 9 more rows overall (Qwen=2, gpt-oss=11). Feed into the later prompt-redaction loop.
