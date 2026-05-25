# RAGAS Lap 2 scorecard

- Eval companies: **10**; rows: **49** (cap 5/company, first-by-id)
- Generators: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (Qwen) vs `openai/gpt-oss-120b` (gpt-oss)
- Judge: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`  (Mistral) on every judge call
- Answer config: answer_mode=`assistant`, prompt_style=`balanced`
- Routing: per-company decision from `outputs/retrieval_weakness_map.csv` (dense if dense_rel ≥ hybrid_rel)
- Identical retrieved context per row passed to BOTH generators (same Python list, no mutation between calls).

## Routing map applied

| company | route | dense_rel | hybrid_rel |
|---|---|---:|---:|
| Volkswagen | **hybrid** | 0.138 | 0.250 |
| Airbus | **hybrid** | 0.188 | 0.338 |
| L'Oréal | **hybrid** | 0.375 | 0.422 |
| TotalEnergies | **hybrid** | 0.375 | 0.487 |
| Schneider | **hybrid** | 0.450 | 0.600 |
| Enel | **dense** | 0.500 | 0.438 |
| BNP Paribas | **dense** | 0.525 | 0.400 |
| ENGIE | **dense** | 0.550 | 0.525 |
| Danone | **dense** | 0.637 | 0.425 |
| Siemens | **dense** | 0.662 | 0.575 |

## Primary scorecard  —  per company  (Qwen | gpt-oss)

| company | route | context_rel | faithfulness (Q\|g) | answer_relevancy (Q\|g) | answer_correctness (Q\|g) | abstention_rate (Q\|g) |
|---|---|---:|---|---|---|---|
| Volkswagen | hybrid | 0.225 ⚠ | 1.000 \| 0.857 | 0.700 \| 0.640 | 0.100 \| 0.100 | 40% \| 40% |
| Airbus | hybrid | 0.338 | 0.911 \| 1.000 | 0.420 \| 0.540 | 0.000 \| 0.000 | 0% \| 80% |
| L'Oréal | hybrid | 0.406 | 0.868 \| 0.919 | 0.725 \| 0.800 | 0.125 \| 0.000 | 25% \| 0% |
| TotalEnergies | hybrid | 0.487 | 0.925 \| 0.845 | 0.800 \| 0.860 | 0.200 \| 0.000 | 0% \| 0% |
| Schneider | hybrid | 0.600 | 0.795 \| 0.900 | 0.800 \| 0.820 | 0.300 \| 0.200 | 0% \| 0% |
| Enel | dense | 0.500 | 0.773 \| 0.710 | 0.580 \| 0.780 | 0.000 \| 0.000 | 20% \| 60% |
| BNP Paribas | dense | 0.525 | 0.877 \| 1.000 | 0.800 \| 0.840 | 0.300 \| 0.100 | 0% \| 0% |
| ENGIE | dense | 0.625 | 0.932 \| 0.673 | 0.740 \| 0.820 | 0.100 \| 0.300 | 0% \| 0% |
| Danone | dense | 0.588 | 0.851 \| 0.838 | 0.680 \| 0.700 | 0.200 \| 0.100 | 0% \| 40% |
| Siemens | dense | 0.662 | 0.923 \| 0.760 | 0.460 \| 0.580 | 0.200 \| 0.100 | 0% \| 80% |
| **OVERALL** | mixed | **0.497** | **0.886 \| 0.849** | **0.669 \| 0.737** | **0.153 \| 0.092** | **8% \| 31%** |

⚠ = context_relevance < 0.30 (retrieval-limited; generation scores here read as a ceiling).

## Generator overall (means across all rows)

| metric | Qwen | gpt-oss | winner |
|---|---:|---:|---|
| faithfulness | 0.886 | 0.849 | **Qwen** |
| answer_relevancy | 0.669 | 0.737 | **gpt-oss** |
| answer_correctness | 0.153 | 0.092 | **Qwen** |
| abstention_rate | 0.082 | 0.306 | **Qwen** |

Abstention split: Qwen abstained on **4/49 (8%)**; gpt-oss abstained on **15/49 (31%)**.

## Headline generator verdict

On the three quality metrics (faithfulness, answer_relevancy, answer_correctness): Qwen wins 2/3, gpt-oss wins 1/3. Lower-abstention winner: **Qwen**.

**Ship: `Qwen`** for this ESG RAG, based on the corpus-wide means above. If your priority is willingness-to-answer (lower abstention), weight that more — see the abstention split.

## Per-company flags

| company | route | context_rel | flag |
|---|---|---:|---|
| Volkswagen | hybrid | 0.225 | retrieval-limited (ceiling) |
| Airbus | hybrid | 0.338 | — |
| L'Oréal | hybrid | 0.406 | — |
| TotalEnergies | hybrid | 0.487 | — |
| Schneider | hybrid | 0.600 | — |
| Enel | dense | 0.500 | — |
| BNP Paribas | dense | 0.525 | — |
| ENGIE | dense | 0.625 | — |
| Danone | dense | 0.588 | — |
| Siemens | dense | 0.662 | — |

## Read

Separate **retrieval health** from **generation health**. Retrieval health is `context_relevance` per company (gold-free; corpus-wide mean **0.497**, range **0.225–0.662**). Retrieval-limited companies (context_rel < 0.30): **Volkswagen** — generation scores on these companies are bounded by the context fed to the model.

Generation health is faithfulness/relevancy/correctness on top of that retrieved context (corpus-wide means: Qwen **0.89**/**0.67**/**0.15**, gpt-oss **0.85**/**0.74**/**0.09**). Over-refusal concentrates in gpt-oss (**31%** vs Qwen **8%**). That gap drags gpt-oss's answer_correctness down because abstain=yes floors correctness to 0.


_Manifest: outputs/lap2.run_manifest.json_