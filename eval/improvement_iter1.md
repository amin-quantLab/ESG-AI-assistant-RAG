# Improvement iteration 1 — corpus-noise filter, judged with Mistral-Small

- Rows: **31**  (row-set vs prior manifest: **True**)
- Judge: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`  embedding: `BAAI/bge-m3`
- Noise mask thresholds: MIN_TOKENS=5, MIN_WORDLIKE=3, MAX_NONALNUM_RATIO=0.5, plus \ufffd presence

## FOUND

Corpus-noise mask rule counts (rules are independent — same chunk can trip several):

| rule | count |
|---|---:|
| `too_short_lt5` | 2057 |
| `replacement_char` | 13 |
| `too_few_wordlike_lt3` | 2486 |
| `nonalnum_ratio_gt_0_5` | 50 |
| **TOTAL UNIQUE NOISE (union)** | **2638 (1.8% of corpus)** |

Mean garbage-in-baseline-top-8 across 31 rows: **0.00 / 8** (0%).

Example dropped chunks:

- [`too_short_lt5`] `2024_integrated_report_ad513d33_pdf_n0021` kind=`narrative` → 'y'
- [`too_short_lt5`] `2024_integrated_report_ad513d33_pdf_n0084` kind=`narrative` → 's'
- [`replacement_char`] `2022_sustainability_report_ad78e211_pdf_t2170` kind=`table_fact` → "In Enel's table on page 123, row '0 � 1' has hat occurre: 0 � 2."
- [`replacement_char`] `2022_sustainability_report_ad78e211_pdf_t2171` kind=`table_fact` → "In Enel's table on page 123, row '1 � 2' has hat occurre: 2 � 4."
- [`too_few_wordlike_lt3`] `2024_integrated_report_ad513d33_pdf_n0016` kind=`narrative` → '. 14 14 13 13 12 12 1 u 1 % n, 7'
- [`too_few_wordlike_lt3`] `2024_integrated_report_ad513d33_pdf_n0021` kind=`narrative` → 'y'
- [`nonalnum_ratio_gt_0_5`] `2024_sustainability_report_44182ca9_pdf_t0021` kind=`table_fact` → "In BNP Paribas's table on page 2, row 'Introduction' has column 1: ....................................................."
- [`nonalnum_ratio_gt_0_5`] `2024_sustainability_report_44182ca9_pdf_t0022` kind=`table_fact` → "In BNP Paribas's table on page 2, row 'Serving indiv' has column 1: idual customers ...................................."

**Known gap (not fixed by this filter):** mid-word table splits like `"In Danone's table on page 318, row 'Waste manageme' has column 1: nt."` still pass — those are a separate parser bug (cell-merge / character-cluster split).

## FIX

Boolean noise mask aligned to `vectors.npy[i]`. A chunk is masked if ANY rule fires:

1. token count < 5
2. text contains Unicode replacement character `\ufffd` (`�`)
3. fewer than 3 word-like tokens (regex `[A-Za-z]{3,}`)
4. non-alnum / non-space char ratio > 0.5

At retrieval time, `sims[mask] = -inf` before top-K. No index mutation, no re-embed, no `app/*` edits.
Excluded chunks: **2638 / 145197 (1.8%)**.

## MEASURED

### Relevance + garbage @ top-8  (baseline = no mask, filtered = mask applied)

| company | n | mean_base_relevance | mean_filt_relevance | Δ | mean_noise_baseline_top8 |
|---|---:|---:|---:|---:|---:|
| Danone | 8 | 0.680 | 0.695 | +0.016 | 0.00 |
| Enel | 8 | 0.445 | 0.438 | -0.008 | 0.00 |
| TotalEnergies | 7 | 0.464 | 0.464 | +0.000 | 0.00 |
| Volkswagen | 8 | 0.141 | 0.141 | +0.000 | 0.00 |
| **OVERALL** | **31** | **0.431** | **0.433** | **+0.002** | **0.00** |

### "Is the question answerable from these passages?" distribution

| condition | yes | partial | no | judge_error | other |
|---|---:|---:|---:|---:|---:|
| baseline | 6 | 24 | 1 | 0 | 0 |
| filtered | 7 | 23 | 1 | 0 | 0 |

### Qualitative swaps (top-N rows by garbage-in-baseline-top-8)

#### Swap 1 — Danone / `ragas_esg_03_01`  (noise in baseline top-8: **0/8**)

- **Q**: What ESG target, ambition, or commitment does Danone disclose in 2024?
- **Baseline top-8 noise chunks (0)**:
- **Filtered top-1** (replacement for the highest-ranked noise chunk):
  - `2022_urd_d3c5cbdd_pdf_t0975` kind=`table_fact` cos=0.712 → "For Danone's urd, EDP in 2022 is 54%."

#### Swap 2 — Danone / `ragas_esg_17_01`  (noise in baseline top-8: **0/8**)

- **Q**: What emissions-related disclosure does Danone report in 2025?
- **Baseline top-8 noise chunks (0)**:
- **Filtered top-1** (replacement for the highest-ranked noise chunk):
  - `2023_urd_0bb6500a_pdf_n0051` kind=`narrative` cos=0.671 → 'and the bonds are listed on carbon label;Euronext Paris; On February 22, 2023, Danone reframed its sustainability journey, On December 20, 2023, Danone published its Climate Transition ■■ through the '

#### Swap 3 — Danone / `ragas_esg_17_02`  (noise in baseline top-8: **0/8**)

- **Q**: What emissions-related disclosure does Danone report in 2025?
- **Baseline top-8 noise chunks (0)**:
- **Filtered top-1** (replacement for the highest-ranked noise chunk):
  - `2023_urd_0bb6500a_pdf_n0051` kind=`narrative` cos=0.671 → 'and the bonds are listed on carbon label;Euronext Paris; On February 22, 2023, Danone reframed its sustainability journey, On December 20, 2023, Danone published its Climate Transition ■■ through the '

#### Swap 4 — Danone / `ragas_esg_17_03`  (noise in baseline top-8: **0/8**)

- **Q**: What ESG target, ambition, or commitment does Danone disclose in 2025?
- **Baseline top-8 noise chunks (0)**:
- **Filtered top-1** (replacement for the highest-ranked noise chunk):
  - `2023_climate_report_fa2980c4_pdf_n0013` kind=`narrative` cos=0.720 → 'DANONE CLIMATE TRANSITION PLAN - WHAT - 6. SETTING OUR REDUCTION TARGETS 15 6. Setting Near-term targets: We aim at reducing our absolute emissions by 34.7% by 2030, compared to 2020 baseline: 2020 to'

## VERDICT

- Baseline relevance 0.431 and filter Δ +0.002. Filter effect is modest; noise count in baseline top-8 averaged 0.00/8. Not a strong signal in either direction.

_Manifest: outputs/improvement_iter1.run_manifest.json_