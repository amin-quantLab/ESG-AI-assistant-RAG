# RAGAS ESG Evaluation Dataset v2: Retrieval Metrics + Cross-Document Comparisons

This version extends the original ESG RAGAS dataset with explicit retrieval-evaluation labels and cross-document comparison questions.

## What changed

- Added `gold_primary_chunk_id` and `gold_primary_corpus_id` to each question.
- Added `gold_relevant_chunk_ids` and `gold_relevant_corpus_ids`.
- Added graded relevance judgments in `qrels.csv`:
  - `3`: direct answer chunk
  - `2`: strong supporting context, if present
  - `1`: adjacent/weak section context
  - `0`: hard negative, available in `qrels_with_hard_negatives.csv`
- Added `hard_negatives.csv` and `hard_negative_corpus_ids` per question.
- Added `retrieval_metrics_template.py` to compute Precision@k, Recall@k, HitRate@k, MRR@k, and NDCG@k.
- Added 24 questions that require retrieving evidence across different companies and different report years.

## Dataset statistics

```json
{
  "total_questions": 132,
  "original_single_document_questions": 108,
  "new_cross_document_comparison_questions": 24,
  "sampled_documents": 30,
  "eval_corpus_chunks": 4192,
  "qrels_positive_rows": 262,
  "hard_negative_rows": 684
}
```

## Main files

- `ragas_esg_eval_dataset_v2.csv` / `.jsonl`: full QA dataset.
- `cross_document_comparison_questions.csv` / `.jsonl`: new cross-company, cross-year comparison subset.
- `qrels.csv`: positive graded relevance judgments for retrieval evaluation.
- `qrels_with_hard_negatives.csv`: qrels plus hard negatives with relevance grade 0.
- `hard_negatives.csv`: hard negative chunks chosen by same/similar topic, company, sector, or ESG terminology.
- `eval_corpus_sampled_30_docs.csv`: the evaluation corpus for the 30 sampled documents; `corpus_id` is the stable retrieval key.
- `retrieval_metrics_template.py`: metrics script.
- `example_oracle_retrieval_results.csv`: sanity-check retrieval results where gold chunks are ranked first.

## Recommended retriever evaluation workflow

1. Index `eval_corpus_sampled_30_docs.csv` in your retriever, using `text` as the chunk text and `corpus_id` as the stable document/chunk identifier.
2. Run the retriever for every `question_id` from `ragas_esg_eval_dataset_v2.csv`.
3. Export retrieval output as CSV with columns:

```csv
question_id,corpus_id,rank,score
```

4. Run:

```bash
python retrieval_metrics_template.py   --qrels qrels.csv   --retrieval-results your_retrieval_results.csv   --k 1 3 5 10   --strict-min-grade 2
```

Use `--strict-min-grade 3` if you only want direct-answer chunks to count as relevant. Use `qrels_with_hard_negatives.csv` when you want to explicitly inspect whether known hard negatives are retrieved.

## Report-style coverage of sampled documents

```json
{
  "short_doc": 5,
  "integrated_or_annual_report": 6,
  "table_heavy_doc": 5,
  "clean_esg_or_sustainability_report": 7,
  "other_report_style": 2,
  "long_doc": 5
}
```

## Company coverage of sampled documents

```json
{
  "Airbus": 2,
  "BNP Paribas": 2,
  "Danone": 4,
  "ENGIE": 3,
  "Enel": 4,
  "Hermès": 1,
  "Iberdrola": 1,
  "L'Oréal": 1,
  "Schneider": 2,
  "Siemens": 3,
  "TotalEnergies": 2,
  "Volkswagen": 4,
  "Unknown": 1
}
```

## Cross-document design

The cross-document questions were intentionally constructed to be hard retrieval cases:

- each question requires at least two cited chunks;
- cited chunks come from different companies;
- cited chunks come from different report years where report year metadata is available;
- topics include emissions targets, water targets, biodiversity, women in management, sustainable finance, human rights, supply-chain due diligence, health and safety, circularity, and supplier engagement.

The answer should be considered correct only if it uses all required cited chunks; this is why the cross-document rows have multiple grade-3 qrels.
