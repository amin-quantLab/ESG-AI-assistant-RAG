# Pre-Lap-2 retrieval weakness map

- Eval companies: **10** (alias map: ENGIE→Engie, Schneider→Schneider Electric; excluded no-corpus: Hermès, Iberdrola, Unknown)
- Rows: **49** (cap 5/company, first-by-id)
- Judge: `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (Q + passages only, NO gold, NO answer)
- Embedding: `BAAI/bge-m3`  noise-masked corpus: 2638 chunks (1.8%)

## Per-company table  (sorted ascending by dense_rel)

| company | idx_name | n | dense_rel | hybrid_rel | Δ | cov_yes% | cov_no% | tablefact% | CLASS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Volkswagen | Volkswagen | 5 | 0.138 | 0.250 | +0.112 | 0% | 80% | 92% | **COVERAGE_GAP** |
| Airbus | Airbus | 5 | 0.188 | 0.338 | +0.150 | 0% | 80% | 5% | **COVERAGE_GAP** |
| L'Oréal | L'Oréal | 4 | 0.375 | 0.422 | +0.047 | 0% | 75% | 12% | **COVERAGE_GAP** |
| TotalEnergies | TotalEnergies | 5 | 0.375 | 0.487 | +0.112 | 20% | 40% | 20% | **COVERAGE_GAP** |
| Schneider | Schneider Electric | 5 | 0.450 | 0.600 | +0.150 | 0% | 20% | 55% | **COVERAGE_GAP** |
| Enel | Enel | 5 | 0.500 | 0.438 | -0.062 | 20% | 60% | 35% | **COVERAGE_GAP** |
| BNP Paribas | BNP Paribas | 5 | 0.525 | 0.400 | -0.125 | 20% | 40% | 25% | **COVERAGE_GAP** |
| ENGIE | Engie | 5 | 0.550 | 0.525 | -0.025 | 0% | 80% | 52% | **COVERAGE_GAP** |
| Danone | Danone | 5 | 0.637 | 0.425 | -0.212 | 0% | 80% | 15% | **GOOD** |
| Siemens | Siemens | 5 | 0.662 | 0.575 | -0.088 | 40% | 20% | 32% | **GOOD** |

## Overall

- Corpus-wide mean dense relevance: **0.441**
- Corpus-wide mean hybrid relevance: **0.446**
- Mean hybrid Δ (hybrid − dense): **+0.005**

Companies per class:

| class | n |
|---|---:|
| **COVERAGE_GAP** | 8 |
| **GOOD** | 2 |

## Worst-3 forensics  (2 example rows per company)

### Volkswagen  →  index `Volkswagen`  (dense_rel=0.138, class=**COVERAGE_GAP**)

**Row `ragas_esg_12_01`** — What emissions-related disclosure does Volkswagen report in 2024?

- dense_rel=0.125 (partial) hybrid_rel=0.375 (yes) tablefact_top8=8/8
- judge note (dense): _The passage only mentions CO1, 2 emissions without specifying the type or context of the emissions-related disclosure._
- coverage verdict (gold-span → nearest Volkswagen chunk): **no**
- coverage reason: _The passage does not mention any specific emissions-related disclosures by Volkswagen in 2024._
- coverage nearest chunk `2024_sustainability_report_91620cfe_pdf_n0125` (kind=narrative, cos_to_gold=0.647):

  > Sustainability Report EU Taxonomy OPERATING EXPENDITURE 2024 1 All percentages relate to the Group’s total operating expenditure. 357

**Dense top-8** (cos | kind | preview):
  1. `2024_sustainability_report_91620cfe_pdf_t2531` cos=0.715 kind=table_fact → "For Volkswagen's sustainability_report, CO1, 2 in 2024 is 0.0."
  2. `2024_sustainability_report_91620cfe_pdf_t2872` cos=0.709 kind=table_fact → "For Volkswagen's sustainability_report, Total wastewater discharge1 in 2024 is 13.9."
  3. `2024_sustainability_report_91620cfe_pdf_t2080` cos=0.707 kind=table_fact → "For Volkswagen's sustainability_report, Total energy consumption in 2024 is 19.0."
  4. `2024_sustainability_report_91620cfe_pdf_t2090` cos=0.702 kind=table_fact → "For Volkswagen's sustainability_report, Natural gas in 2024 is 8.7."
  5. `2024_sustainability_report_91620cfe_pdf_t2086` cos=0.692 kind=table_fact → "For Volkswagen's sustainability_report, Coal and coal products in 2024 is 0.8."
  6. `2024_sustainability_report_91620cfe_pdf_t2088` cos=0.690 kind=table_fact → "For Volkswagen's sustainability_report, Crude oil and petroleum products in 2024 is 1.1."
  7. `2024_sustainability_report_91620cfe_pdf_t2083` cos=0.688 kind=table_fact → "For Volkswagen's sustainability_report, with operational control in 2024 is 24.6."
  8. `2024_sustainability_report_91620cfe_pdf_t2868` cos=0.682 kind=table_fact → "For Volkswagen's sustainability_report, Total water withdrawals in 2024 is 21.2."

**Row `ragas_esg_12_02`** — What ESG target, ambition, or commitment does Volkswagen disclose in 2024?

- dense_rel=0.188 (partial) hybrid_rel=0.188 (partial) tablefact_top8=7/8
- judge note (dense): _The passages mention a decarbonization index but lack specific ESG targets, ambitions, or commitments._
- coverage verdict (gold-span → nearest Volkswagen chunk): **no**
- coverage reason: _The passage discusses supplier environmental management systems but does not mention any ESG target, ambition, or commitment for 2024._
- coverage nearest chunk `2022_sustainability_report_c273a201_pdf_t1878` (kind=table_fact, cos_to_gold=0.676):

  > In Volkswagen's table on page 112, row 'Revenue-based direct suppliers in scope with certified environmental management system pursuant to ISO 14001 or EMAS validation or letter of commitment' has column 1: in %, column 2: 85, column 3: 78, column 4:

**Dense top-8** (cos | kind | preview):
  1. `2024_sustainability_report_91620cfe_pdf_t2531` cos=0.706 kind=table_fact → "For Volkswagen's sustainability_report, CO1, 2 in 2024 is 0.0."
  2. `2024_sustainability_report_91620cfe_pdf_t2090` cos=0.691 kind=table_fact → "For Volkswagen's sustainability_report, Natural gas in 2024 is 8.7."
  3. `2024_sustainability_report_91620cfe_pdf_t2083` cos=0.679 kind=table_fact → "For Volkswagen's sustainability_report, with operational control in 2024 is 24.6."
  4. `2024_sustainability_report_91620cfe_pdf_t2080` cos=0.679 kind=table_fact → "For Volkswagen's sustainability_report, Total energy consumption in 2024 is 19.0."
  5. `2024_sustainability_report_91620cfe_pdf_t2086` cos=0.678 kind=table_fact → "For Volkswagen's sustainability_report, Coal and coal products in 2024 is 0.8."
  6. `volkswagen_ag_2023_earnings_call_mar_13__n0012` cos=0.677 kind=narrative → 'Oliver Blume [Executives]: Thanks, Arno, for your insights. These results form a strong basis for a year 2024, which is expected to be demanding in ma'
  7. `2024_sustainability_report_91620cfe_pdf_t1759` cos=0.674 kind=table_fact → "For Volkswagen's sustainability_report, Decarbonization index* │GRI 305- in 2024 is 48.0."
  8. `2024_sustainability_report_91620cfe_pdf_t2088` cos=0.669 kind=table_fact → "For Volkswagen's sustainability_report, Crude oil and petroleum products in 2024 is 1.1."

### Airbus  →  index `Airbus`  (dense_rel=0.188, class=**COVERAGE_GAP**)

**Row `ragas_esg_01_01`** — What does Airbus disclose about diversity, inclusion, or gender representation in 2025?

- dense_rel=0.188 (partial) hybrid_rel=0.188 (partial) tablefact_top8=1/8
- judge note (dense): _Passage 5 provides general information about diversity and inclusion but lacks specific details about 2025._
- coverage verdict (gold-span → nearest Airbus chunk): **partial**
- coverage reason: _The passage mentions diversity and inclusion but does not provide specific details about Airbus's goals or disclosures for 2025._
- coverage nearest chunk `2024_sustainability_report_44a3d536_pdf_n0023` (kind=narrative, cos_to_gold=0.675):

  > To pursue the highest standards, Airbus has adopted the ISO 45001 standard for occupational health and safety management systems. The Airbus Occupational Health & Safety Policy provides Airbus employees with a single top-level reference for occupatio

**Dense top-8** (cos | kind | preview):
  1. `2024_esg_databook_3ea4de53_pdf_n0001` cos=0.643 kind=narrative → "Airbus ESG Datasheet 2025 on FY 2024 Issue #2 - July 2025 This document contains quantitative information related to Airbus' ESG performance. It shall"
  2. `airbus_se_nine_months_2025_earnings_call_n0002` cos=0.639 kind=narrative → "Helene Le Gorgeu [Executives]: Thank you, Sharon, and good evening, ladies and gentlemen. This is the Airbus' Nine-Months 2025 Earnings Release Confer"
  3. `airbus_se_2024_earnings_call_feb_20_2025_n0012` cos=0.629 kind=narrative → 'Guillaume Faury [Executives]: Thank you, Thomas. So Page 16, and as the basis for the 2025 guidance, the company assumes no additional disruptions to '
  4. `airbus_se_2023_earnings_call_feb_15_2024_n0002` cos=0.627 kind=narrative → 'Helene Le Gorgeu [Executives]: Thank you, Sharon, and good morning, ladies and gentlemen. This is the Airbus Full Year 2023 Results Release Conference'
  5. `2024_sustainability_report_44a3d536_pdf_n0023` cos=0.622 kind=narrative → 'To pursue the highest standards, Airbus has adopted the ISO 45001 standard for occupational health and safety management systems. The Airbus Occupatio'
  6. `airbus_se_2022_earnings_call_feb_16_2023_n0002` cos=0.622 kind=narrative → 'Helene Le Gorgeu [Executives]: Thank you, Melanie. Good morning, ladies and gentlemen. This is the Airbus Full Year 2022 Results Release Conference Ca'
  7. `airbus_se_q1_2025_earnings_call_apr_30_2_n0002` cos=0.618 kind=narrative → 'Helene Le Gorgeu [Executives]: Thank you, Sharon, and good evening, ladies and gentlemen. This is the Airbus Q1 2025 Results Release Conference Call. '
  8. `2024_sustainability_report_44a3d536_pdf_t0064` cos=0.617 kind=table_fact → "In Airbus's table on page 3, row 'In 2025, we are working to he' has ioneer sustainable: lp our industry, Our ambition: increasingly, is for our: need"

**Row `ragas_esg_16_01`** — What emissions-related disclosure does Airbus report in 2025?

- dense_rel=0.188 (partial) hybrid_rel=0.438 (partial) tablefact_top8=0/8
- judge note (dense): _Passage 4 mentions GHG emissions but lacks specific details about the 2025 emissions-related disclosure._
- coverage verdict (gold-span → nearest Airbus chunk): **no**
- coverage reason: _The passage does not specify any emissions-related disclosure for 2025, only mentioning past actions and future announcements._
- coverage nearest chunk `airbus_se_2022_earnings_call_feb_16_2023_n0011` (kind=narrative, cos_to_gold=0.711):

  > Guillaume Faury [Executives]: comes to sustainability, as highlighted at our last Airbus summit, we endeavor to set the sustainability agenda for the aerospace sector. This is our priority. That's my priority. Our focus is now to transition together 

**Dense top-8** (cos | kind | preview):
  1. `2024_esg_databook_3ea4de53_pdf_n0001` cos=0.637 kind=narrative → "Airbus ESG Datasheet 2025 on FY 2024 Issue #2 - July 2025 This document contains quantitative information related to Airbus' ESG performance. It shall"
  2. `airbus_se_2024_earnings_call_feb_20_2025_n0012` cos=0.635 kind=narrative → 'Guillaume Faury [Executives]: Thank you, Thomas. So Page 16, and as the basis for the 2025 guidance, the company assumes no additional disruptions to '
  3. `airbus_se_nine_months_2025_earnings_call_n0002` cos=0.630 kind=narrative → "Helene Le Gorgeu [Executives]: Thank you, Sharon, and good evening, ladies and gentlemen. This is the Airbus' Nine-Months 2025 Earnings Release Confer"
  4. `2024_esg_databook_3ea4de53_pdf_n0008` cos=0.617 kind=narrative → "used in the methodology, it provides a relevant view of the sources of GHG emissions in the Company's supply chain and enables comparison of the vario"
  5. `airbus_se_q1_2023_earnings_call_may_03_2_n0013` cos=0.605 kind=narrative → 'Guillaume Faury [Executives]: Yes. Thank you, Daniela. So when it comes to the operating environment and the supply chain situations as expected, unfo'
  6. `airbus_se_q1_2025_earnings_call_apr_30_2_n0002` cos=0.604 kind=narrative → 'Helene Le Gorgeu [Executives]: Thank you, Sharon, and good evening, ladies and gentlemen. This is the Airbus Q1 2025 Results Release Conference Call. '
  7. `airbus_se_2023_earnings_call_feb_15_2024_n0002` cos=0.603 kind=narrative → 'Helene Le Gorgeu [Executives]: Thank you, Sharon, and good morning, ladies and gentlemen. This is the Airbus Full Year 2023 Results Release Conference'
  8. `airbus_se_q1_2025_earnings_call_apr_30_2_n0010` cos=0.596 kind=narrative → 'Guillaume Faury [Executives]: Thank you, Thomas. So on to our guidance, which remains unchanged. As the basis of its 2025 guidance, the company exclud'

### L'Oréal  →  index `L'Oréal`  (dense_rel=0.375, class=**COVERAGE_GAP**)

**Row `ragas_esg_08_01`** — What emissions-related disclosure does L'Oréal report in 2024?

- dense_rel=0.188 (partial) hybrid_rel=0.312 (partial) tablefact_top8=0/8
- judge note (dense): _Passage 2 provides emissions-related disclosures but lacks specific details on the 2024 reporting framework or methodology._
- coverage verdict (gold-span → nearest L'Oréal chunk): **partial**
- coverage reason: _The passage mentions greenhouse gas emissions as part of an environmental display system but does not specify the exact emissions-related disclosure reported by L'Oréal in 2024._
- coverage nearest chunk `2024_urd_daccf63c_pdf_n0204` (kind=narrative, cos_to_gold=0.745):

  > example), and by leveraging eco-friendly processes that prevent upstream pollution. L’Oréal has specific training programmes for raising employees' awareness about issues related to the climate, water, biodiversity, and resources. These programmes ha

**Dense top-8** (cos | kind | preview):
  1. `2024_urd_daccf63c_pdf_n0054` cos=0.707 kind=narrative → "1Presentation of the Group – Integrated Report 2024 Financial Results and Corporate Social Responsibility commitments ◼ 2024 results L'Oréal is aiming"
  2. `2024_urd_daccf63c_pdf_n0237` cos=0.698 kind=narrative → "Sustainability Report 4 Climate: Mitigation and Adaptation (E1) 4.2.5 Climate outcomes 4.2.5.1 L'Oréal's objectives in relation to climate change Targ"
  3. `2024_urd_daccf63c_pdf_n0053` cos=0.670 kind=narrative → "1Presentation of the Group – Integrated Report 2024 Financial Results and Corporate Social Responsibility commitments 1.4.2L'Oréal for the Future prog"
  4. `2024_urd_daccf63c_pdf_n0260` cos=0.649 kind=narrative → 'Sustainability Report 4 Biodiversity and ecosystems (E4) 4.4.4 Outcomes related to water resources 4.4.4.1 CSRD disclosure requirements relating to wa'
  5. `2024_urd_daccf63c_pdf_n0055` cos=0.647 kind=narrative → '52 L’ORÉAL — UNIVERSAL REGISTRATION DOCUMENT 2024'
  6. `2024_urd_daccf63c_pdf_n0030` cos=0.646 kind=narrative → '1 Presentation of the Group – Integrated Report Value-creating model 1.3.2 Value chain CSRD (1) Value chain Research, Innovation Factories and Design '
  7. `2024_urd_daccf63c_pdf_n0204` cos=0.644 kind=narrative → "example), and by leveraging eco-friendly processes that prevent upstream pollution. L’Oréal has specific training programmes for raising employees' aw"
  8. `2024_urd_daccf63c_pdf_n0232` cos=0.643 kind=narrative → '4Sustainability Report L’ORÉAL — UNIVERSAL REGISTRATION DOCUMENT 2024204'

**Row `ragas_esg_08_02`** — What ESG target, ambition, or commitment does L'Oréal disclose in 2024?

- dense_rel=0.250 (partial) hybrid_rel=0.312 (partial) tablefact_top8=2/8
- judge note (dense): _The passages mention climate and pollution targets but lack specific ESG commitments for 2024._
- coverage verdict (gold-span → nearest L'Oréal chunk): **no**
- coverage reason: _The passage discusses waste and recycling metrics but does not mention any ESG targets, ambitions, or commitments for 2024._
- coverage nearest chunk `2024_urd_daccf63c_pdf_n0271` (kind=narrative, cos_to_gold=0.780):

  > Sustainability Report 4 Own workforce (S1) 4.6.4.2 Outcomes related to resource outflows (E5-5) Key performance indicator 2024 outcomes Percentage of recyclable content in packaging 53% Total amount of non-recycled waste 50,462 tonnes Percentage of n

**Dense top-8** (cos | kind | preview):
  1. `2024_urd_daccf63c_pdf_n0054` cos=0.723 kind=narrative → "1Presentation of the Group – Integrated Report 2024 Financial Results and Corporate Social Responsibility commitments ◼ 2024 results L'Oréal is aiming"
  2. `2024_urd_daccf63c_pdf_n0237` cos=0.710 kind=narrative → "Sustainability Report 4 Climate: Mitigation and Adaptation (E1) 4.2.5 Climate outcomes 4.2.5.1 L'Oréal's objectives in relation to climate change Targ"
  3. `2024_urd_daccf63c_pdf_n0053` cos=0.709 kind=narrative → "1Presentation of the Group – Integrated Report 2024 Financial Results and Corporate Social Responsibility commitments 1.4.2L'Oréal for the Future prog"
  4. `2024_urd_daccf63c_pdf_n0253` cos=0.670 kind=narrative → 'Sustainability Report 4 Water resources: consumption and withdrawals (E3) 4.3.4 Pollution-related outcomes 4.3.4.1 Air pollution-related outcomes (E2-'
  5. `l_or_al_s_a_2024_earnings_call_feb_07_20_n0021` cos=0.662 kind=narrative → "Nicolas Hieronimus [Executives]: Thank you, Alexis, for your confidence and determination. So good morning, everyone. It's an important day today beca"
  6. `2024_urd_daccf63c_pdf_n0030` cos=0.658 kind=narrative → '1 Presentation of the Group – Integrated Report Value-creating model 1.3.2 Value chain CSRD (1) Value chain Research, Innovation Factories and Design '
  7. `2024_urd_daccf63c_pdf_t4725` cos=0.657 kind=table_fact → "For L'Oréal's urd, 442,436 in 2024 is -4%."
  8. `2024_urd_daccf63c_pdf_t4738` cos=0.656 kind=table_fact → "For L'Oréal's urd, 180,988 in 2024 is 35%."

## Per-company prescription

| company | class | prescription |
|---|---|---|
| Volkswagen | COVERAGE_GAP | Coverage-limited: the answer text itself is not in the company's documents in the index (semantic-nearest chunk fails 1-on-1 judgement). Flag those rows as data-bound and add documents OR remove the rows. |
| Airbus | COVERAGE_GAP | Coverage-limited: the answer text itself is not in the company's documents in the index (semantic-nearest chunk fails 1-on-1 judgement). Flag those rows as data-bound and add documents OR remove the rows. |
| L'Oréal | COVERAGE_GAP | Coverage-limited: the answer text itself is not in the company's documents in the index (semantic-nearest chunk fails 1-on-1 judgement). Flag those rows as data-bound and add documents OR remove the rows. |
| TotalEnergies | COVERAGE_GAP | Coverage-limited: the answer text itself is not in the company's documents in the index (semantic-nearest chunk fails 1-on-1 judgement). Flag those rows as data-bound and add documents OR remove the rows. |
| Schneider | COVERAGE_GAP | Coverage-limited: the answer text itself is not in the company's documents in the index (semantic-nearest chunk fails 1-on-1 judgement). Flag those rows as data-bound and add documents OR remove the rows. |
| Enel | COVERAGE_GAP | Coverage-limited: the answer text itself is not in the company's documents in the index (semantic-nearest chunk fails 1-on-1 judgement). Flag those rows as data-bound and add documents OR remove the rows. |
| BNP Paribas | COVERAGE_GAP | Coverage-limited: the answer text itself is not in the company's documents in the index (semantic-nearest chunk fails 1-on-1 judgement). Flag those rows as data-bound and add documents OR remove the rows. |
| ENGIE | COVERAGE_GAP | Coverage-limited: the answer text itself is not in the company's documents in the index (semantic-nearest chunk fails 1-on-1 judgement). Flag those rows as data-bound and add documents OR remove the rows. |
| Danone | GOOD | No retrieval fix needed; ship as-is. |
| Siemens | GOOD | No retrieval fix needed; ship as-is. |

## Lap-2 readiness verdict

- **Ready (GOOD on dense)**: Danone, Siemens
- **Ready IF we ship hybrid**: —
- **Pre-Lap-2 fix needed (re-embed pilot — start here)**: —
- **Coverage-limited (data ceiling, not model failure)**: Airbus, BNP Paribas, ENGIE, Enel, L'Oréal, Schneider, TotalEnergies, Volkswagen
- **Manual inspection**: —

Lap-2 results for the coverage-limited companies must be read as a documents-side ceiling — adding documents or removing the rows is the correct response, not changing the retriever.

_Manifest: outputs/retrieval_weakness_map.run_manifest.json_