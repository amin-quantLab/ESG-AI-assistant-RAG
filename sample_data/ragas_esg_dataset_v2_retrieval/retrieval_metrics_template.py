#!/usr/bin/env python3
"""
Retrieval metrics template for the ESG RAG/RAGAS evaluation dataset.

Expected retrieval results CSV columns:
    question_id, corpus_id, rank
Optional columns:
    score, source_file, chunk_id

Example:
    python retrieval_metrics_template.py \
        --qrels qrels.csv \
        --retrieval-results my_retrieval_results.csv \
        --k 1 3 5 10 \
        --strict-min-grade 2

Notes:
- qrels.csv uses graded relevance:
    3 = direct answer chunk
    2 = strong supporting context, if present
    1 = adjacent/weak section context
    0 = hard negative, only present in qrels_with_hard_negatives.csv
- Precision/Recall/HitRate/MRR use `strict_min_grade` to decide what counts as relevant.
- NDCG uses the actual graded relevance values.
"""
import argparse
import math
from collections import defaultdict
import pandas as pd


def dcg(grades):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def load_qrels(path):
    qrels_df = pd.read_csv(path)
    qrels_df["relevance_grade"] = qrels_df["relevance_grade"].astype(float)
    qrels = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        qrels[str(row["question_id"])][str(row["corpus_id"])] = float(row["relevance_grade"])
    return qrels


def load_results(path):
    res = pd.read_csv(path)
    required = {"question_id", "corpus_id"}
    missing = required - set(res.columns)
    if missing:
        raise ValueError(f"retrieval results missing required columns: {missing}")
    if "rank" not in res.columns:
        if "score" in res.columns:
            res = res.sort_values(["question_id", "score"], ascending=[True, False]).copy()
            res["rank"] = res.groupby("question_id").cumcount() + 1
        else:
            res = res.copy()
            res["rank"] = res.groupby("question_id").cumcount() + 1
    res["question_id"] = res["question_id"].astype(str)
    res["corpus_id"] = res["corpus_id"].astype(str)
    res = res.sort_values(["question_id", "rank"])
    results = defaultdict(list)
    for _, row in res.iterrows():
        results[row["question_id"]].append(str(row["corpus_id"]))
    return results


def metrics_at_k(qrels, results, k, strict_min_grade=2):
    rows = []
    for qid, relmap in qrels.items():
        retrieved = results.get(qid, [])[:k]
        strict_relevant = {cid for cid, grade in relmap.items() if grade >= strict_min_grade}
        if not strict_relevant:
            continue
        retrieved_relevant = [cid for cid in retrieved if cid in strict_relevant]
        precision = len(retrieved_relevant) / k
        recall = len(set(retrieved_relevant)) / len(strict_relevant)
        hit_rate = 1.0 if retrieved_relevant else 0.0
        rr = 0.0
        for idx, cid in enumerate(retrieved, start=1):
            if cid in strict_relevant:
                rr = 1.0 / idx
                break
        grades = [relmap.get(cid, 0.0) for cid in retrieved]
        ideal_grades = sorted([g for g in relmap.values() if g > 0], reverse=True)[:k]
        ndcg = dcg(grades) / dcg(ideal_grades) if ideal_grades and dcg(ideal_grades) > 0 else 0.0
        rows.append({
            "question_id": qid,
            f"precision@{k}": precision,
            f"recall@{k}": recall,
            f"hit_rate@{k}": hit_rate,
            f"mrr@{k}": rr,
            f"ndcg@{k}": ndcg,
            "num_strict_relevant": len(strict_relevant),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", default="qrels.csv")
    parser.add_argument("--retrieval-results", required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--strict-min-grade", type=float, default=2.0)
    parser.add_argument("--per-question-out", default="retrieval_metrics_per_question.csv")
    parser.add_argument("--summary-out", default="retrieval_metrics_summary.csv")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    results = load_results(args.retrieval_results)

    per_k = []
    summary_rows = []
    for k in args.k:
        dfk = metrics_at_k(qrels, results, k, strict_min_grade=args.strict_min_grade)
        per_k.append(dfk)
        metric_cols = [c for c in dfk.columns if c not in {"question_id", "num_strict_relevant"}]
        row = {"k": k, "question_count": len(dfk)}
        row.update({c: dfk[c].mean() for c in metric_cols})
        summary_rows.append(row)

    # Merge per-question metrics across k.
    merged = None
    for df in per_k:
        cols = [c for c in df.columns if c != "num_strict_relevant"]
        df = df[cols]
        merged = df if merged is None else merged.merge(df, on="question_id", how="outer")
    summary = pd.DataFrame(summary_rows)
    merged.to_csv(args.per_question_out, index=False)
    summary.to_csv(args.summary_out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
