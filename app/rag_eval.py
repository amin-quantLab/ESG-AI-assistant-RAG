"""Retrieval evaluation helpers for the local ESG RAG pipeline."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from app.rag import DEFAULT_BASE_URL, DEFAULT_INDEX_DIR, build_index, retrieve_chunks
from app.utils import OUTPUT_DIR

DEFAULT_EVAL_DATASET = Path(__file__).resolve().parent.parent / "sample_data" / "rag_evaluation_dataset.csv"
DEFAULT_EVAL_OUTPUT = OUTPUT_DIR / "rag_eval_results.json"
DEFAULT_GRID_OUTPUT_JSON = OUTPUT_DIR / "rag_eval_grid_results.json"
DEFAULT_GRID_OUTPUT_CSV = OUTPUT_DIR / "rag_eval_grid_results.csv"
DEFAULT_BEST_CONFIG = OUTPUT_DIR / "best_rag_config.json"


def _normalize_match_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _load_rows(dataset_path: Path) -> list[dict[str, str]]:
    with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_expected_contexts(row: dict[str, str]) -> list[str]:
    contexts_field = row.get("contexts", "").strip()
    if contexts_field:
        try:
            parsed = json.loads(contexts_field)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

    context = row.get("context", "").strip()
    return [context] if context else []


def get_ground_truth_answer(row: dict[str, str]) -> str:
    return (row.get("ground_truth_answer") or row.get("ground_truth") or "").strip()


def get_topic_label(row: dict[str, str]) -> str:
    topic = (row.get("topic") or "").strip()
    if topic:
        return topic

    topic_tags = (row.get("topic_tags") or "").strip()
    if not topic_tags:
        return ""
    try:
        parsed = json.loads(topic_tags)
        if isinstance(parsed, list):
            return ", ".join(str(item) for item in parsed[:3])
    except json.JSONDecodeError:
        pass
    return topic_tags


def _answer_variants_for_scoring(answer: str) -> list[str]:
    normalized = (answer or "").strip()
    if not normalized:
        return []

    variants: list[str] = [normalized]

    no_citations = re.sub(r"\[[^\]]+\]", "", normalized)
    no_citations = re.sub(r"\s+", " ", no_citations).strip()
    if no_citations:
        variants.append(no_citations)

    before_limitations = re.split(r"\*\*Limitations\*\*|Limitations", normalized, maxsplit=1)[0].strip()
    if before_limitations:
        variants.append(before_limitations)

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", before_limitations or normalized) if part.strip()]
    if paragraphs:
        variants.append(paragraphs[0])
    if len(paragraphs) >= 2:
        variants.append("\n\n".join(paragraphs[:2]))

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant and variant not in seen:
            deduped.append(variant)
            seen.add(variant)
    return deduped


def _infer_company(index_dir: Path, rows: list[dict[str, str]]) -> str | None:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    pdf_blob = " ".join(manifest.get("pdfs", [])).lower()
    companies = {row["company"] for row in rows if row.get("company")}
    matches = [company for company in companies if company.lower().replace(" ", "") in pdf_blob.replace(" ", "")]
    if len(matches) == 1:
        return matches[0]

    slug_matches = []
    for company in companies:
        slug = re.sub(r"[^a-z0-9]+", "", company.lower())
        if slug and slug in re.sub(r"[^a-z0-9]+", "", pdf_blob):
            slug_matches.append(company)
    if len(slug_matches) == 1:
        return slug_matches[0]
    return None


def _score_chunk_against_snippet(snippet: str, chunk_text: str) -> float:
    normalized_snippet = _normalize_match_text(snippet)
    normalized_chunk = _normalize_match_text(chunk_text)
    if not normalized_snippet or not normalized_chunk:
        return 0.0
    if normalized_snippet in normalized_chunk:
        return 100.0
    return max(
        float(fuzz.partial_ratio(normalized_snippet, normalized_chunk)),
        float(fuzz.token_set_ratio(normalized_snippet, normalized_chunk)),
    )


def _score_retrieval(expected_contexts: list[str], retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    best_score = 0.0
    best_chunk_id = None
    best_snippet = None

    for snippet in expected_contexts:
        for chunk in retrieved_chunks:
            score = _score_chunk_against_snippet(snippet, chunk["text"])
            if score > best_score:
                best_score = score
                best_chunk_id = chunk["chunk_id"]
                best_snippet = snippet

    status = classify_match_score(best_score)

    return {
        "status": status,
        "best_match_score": round(best_score, 2),
        "best_chunk_id": best_chunk_id,
        "matched_context": best_snippet,
    }


def classify_match_score(score: float) -> str:
    if score >= 90:
        return "hit"
    if score >= 70:
        return "partial"
    return "miss"


def score_answer_text(answer: str, ground_truth: str, expected_contexts: list[str]) -> dict[str, Any]:
    variants = _answer_variants_for_scoring(answer)
    if not variants:
        return {"status": "miss", "score": 0.0}

    targets = [ground_truth.strip(), *[context.strip() for context in expected_contexts if context.strip()]]
    best_score = 0.0
    best_target = None
    best_variant = None

    for variant in variants:
        for target in targets:
            score = max(
                float(fuzz.partial_ratio(variant.lower(), target.lower())),
                float(fuzz.token_set_ratio(variant.lower(), target.lower())),
            )
            if score > best_score:
                best_score = score
                best_target = target
                best_variant = variant

    return {
        "status": classify_match_score(best_score),
        "score": round(best_score, 2),
        "matched_target": best_target,
        "matched_answer_variant": best_variant,
    }


def compute_eval_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "hit": 0, "partial": 0, "miss": 0,
            "hit_rate": 0.0, "partial_or_hit_rate": 0.0,
            "average_best_match_score": 0.0, "average_top_retrieval_score": 0.0,
        }

    counts = Counter(result.get("status", "miss") for result in results)
    hit = counts.get("hit", 0)
    partial = counts.get("partial", 0)
    miss = counts.get("miss", 0)

    best_scores = [float(result.get("best_match_score", 0)) for result in results]
    top_scores = [float(result["top_score"]) for result in results if result.get("top_score") is not None]

    return {
        "hit": hit,
        "partial": partial,
        "miss": miss,
        "hit_rate": round(hit / total, 4) if total else 0.0,
        "partial_or_hit_rate": round((hit + partial) / total, 4) if total else 0.0,
        "average_best_match_score": round(sum(best_scores) / len(best_scores), 2) if best_scores else 0.0,
        "average_top_retrieval_score": round(sum(top_scores) / len(top_scores), 4) if top_scores else 0.0,
    }


def evaluate_retrieval(
    *,
    index_dir: Path,
    dataset_path: Path = DEFAULT_EVAL_DATASET,
    company: str | None = None,
    top_k: int = 5,
    embedding_model: str | None = None,
    retrieval_architecture: str = "dense",
    search_breadth: int | None = None,
    base_url: str,
) -> dict[str, Any]:
    rows = _load_rows(dataset_path)
    selected_company = company or _infer_company(index_dir, rows)
    if not selected_company:
        raise RuntimeError("Could not infer which company to evaluate. Pass --company explicitly.")

    company_rows = [row for row in rows if row.get("company") == selected_company]
    if not company_rows:
        raise RuntimeError(f"No evaluation rows found for company '{selected_company}'.")

    results: list[dict[str, Any]] = []
    for row in company_rows:
        question = row["question"]
        expected_contexts = _parse_expected_contexts(row)
        retrieved_chunks, embedding_model = retrieve_chunks(
            index_dir=index_dir,
            question=question,
            top_k=top_k,
            embedding_model=embedding_model,
            retrieval_architecture=retrieval_architecture,
            search_breadth=search_breadth,
            base_url=base_url,
        )
        scoring = _score_retrieval(expected_contexts, retrieved_chunks)
        results.append(
            {
                "company": selected_company,
                "question": question,
                "topic": get_topic_label(row),
                "ground_truth": get_ground_truth_answer(row),
                "status": scoring["status"],
                "best_match_score": scoring["best_match_score"],
                "best_chunk_id": scoring["best_chunk_id"],
                "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
                "top_score": round(float(retrieved_chunks[0]["score"]), 4) if retrieved_chunks else None,
                "expected_context": scoring["matched_context"],
            }
        )

    extended = compute_eval_summary(results)
    counts = Counter(result["status"] for result in results)
    topic_counts = Counter((result["topic"], result["status"]) for result in results)
    by_topic: dict[str, dict[str, int]] = {}
    for (topic, status), count in topic_counts.items():
        by_topic.setdefault(topic, {"hit": 0, "partial": 0, "miss": 0})
        by_topic[topic][status] = count

    return {
        "company": selected_company,
        "question_count": len(results),
        "top_k": top_k,
        "retrieval_architecture": retrieval_architecture,
        "search_breadth": max(top_k, search_breadth or top_k),
        "summary": extended,
        "by_topic": by_topic,
        "results": results,
        "embedding_model": embedding_model,
    }


def write_evaluation_outputs(payload: dict[str, Any], output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    rows = payload["results"]
    headers = [
        "company",
        "question",
        "topic",
        "status",
        "best_match_score",
        "best_chunk_id",
        "top_score",
        "ground_truth",
        "retrieved_chunk_ids",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    header: (" | ".join(row["retrieved_chunk_ids"]) if header == "retrieved_chunk_ids" else row.get(header, ""))
                    for header in headers
                }
            )
    return output_path, csv_path


def run_rag_eval(args: Any) -> int:
    retrieval_mode = getattr(args, "retrieval_mode", None) or getattr(args, "retrieval_architecture", "dense")
    candidate_k = getattr(args, "candidate_k", None) or args.search_breadth
    payload = evaluate_retrieval(
        index_dir=args.index_dir,
        dataset_path=args.dataset,
        company=args.company,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        retrieval_architecture=retrieval_mode,
        search_breadth=candidate_k,
        base_url=args.base_url,
    )
    json_path, csv_path = write_evaluation_outputs(payload, args.output)

    summary = payload["summary"]
    print(
        f"Evaluated retrieval for {payload['company']} on {payload['question_count']} questions "
        f"(top_k={payload['top_k']}, breadth={payload['search_breadth']}, "
        f"architecture={payload['retrieval_architecture']}, embedding_model={payload['embedding_model']})."
    )
    print(
        f"Hits: {summary['hit']} | Partial: {summary['partial']} | "
        f"Misses: {summary['miss']} | Hit rate: {summary['hit_rate']:.1%} "
        f"| Partial+Hit rate: {summary['partial_or_hit_rate']:.1%}"
    )
    print(
        f"Avg best match score: {summary['average_best_match_score']:.2f} "
        f"| Avg top retrieval score: {summary['average_top_retrieval_score']:.4f}"
    )
    print(f"Saved evaluation to {json_path} and {csv_path}")

    failures = [row for row in payload["results"] if row["status"] != "hit"]
    if failures:
        print("\nQuestions needing attention:")
        for row in failures:
            print(
                f"- [{row['status']}] score={row['best_match_score']:.2f} | "
                f"{row['topic']} | {row['question']}"
            )
    return 0


def _pick_pdf_for_company(company: str, sample_dir: Path) -> Path:
    best_path = None
    best_score = -1.0
    pdf_paths = sorted(sample_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found under {sample_dir}.")

    company_slug = re.sub(r"[^a-z0-9]+", " ", company.lower()).strip()
    for pdf_path in pdf_paths:
        stem = pdf_path.stem.replace("_", " ").replace("-", " ")
        score = max(
            float(fuzz.partial_ratio(company_slug, stem.lower())),
            float(fuzz.token_set_ratio(company_slug, stem.lower())),
        )
        if score > best_score:
            best_score = score
            best_path = pdf_path
    if best_path is None:
        raise RuntimeError(f"Could not match a sample PDF to company '{company}'.")
    return best_path


def _configuration_key(config: dict[str, Any]) -> tuple:
    return (
        config["target_tokens"],
        config["min_tokens"],
        config["max_tokens"],
        config["overlap_tokens"],
        config["top_k"],
        config["retrieval_mode"],
    )


def _generate_grid_configs(
    chunk_target_tokens: list[int],
    chunk_min_tokens: list[int],
    chunk_max_tokens: list[int],
    chunk_overlap_tokens: list[int],
    top_k_values: list[int],
    retrieval_modes: list[str],
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for target in chunk_target_tokens:
        for min_tok in chunk_min_tokens:
            for max_tok in chunk_max_tokens:
                for overlap in chunk_overlap_tokens:
                    if not (min_tok <= target <= max_tok):
                        continue
                    if overlap >= max_tok:
                        continue
                    for tk in top_k_values:
                        for mode in retrieval_modes:
                            configs.append(
                                {
                                    "target_tokens": target,
                                    "min_tokens": min_tok,
                                    "max_tokens": max_tok,
                                    "overlap_tokens": overlap,
                                    "top_k": tk,
                                    "retrieval_mode": mode,
                                }
                            )
    return configs


def run_rag_eval_grid(args: Any) -> int:
    from app.rag import require_api_key, AlbertClient

    require_api_key()
    dataset_path = args.dataset
    company = args.company
    if not company:
        raise RuntimeError("--company is required for rag-eval-grid.")

    rows = _load_rows(dataset_path)
    company_rows = [row for row in rows if row.get("company") == company]
    if not company_rows:
        raise RuntimeError(f"No evaluation rows found for company '{company}'.")

    sample_dir = args.sample_dir
    pdf_path = _pick_pdf_for_company(company, sample_dir)

    client = AlbertClient(api_key=require_api_key(), base_url=args.base_url)
    embedding_model = args.embedding_model or client.get_embedding_model(preferred="bge-m3")

    index_root = args.index_root
    index_root.mkdir(parents=True, exist_ok=True)
    built_cache: set[tuple] = set()

    configs = _generate_grid_configs(
        chunk_target_tokens=args.chunk_targets,
        chunk_min_tokens=args.chunk_mins,
        chunk_max_tokens=args.chunk_maxs,
        chunk_overlap_tokens=args.chunk_overlaps,
        top_k_values=args.top_ks,
        retrieval_modes=args.retrieval_modes,
    )

    print(f"Grid search: {len(configs)} valid configurations across {len(company_rows)} questions")
    grid_results: list[dict[str, Any]] = []

    for idx, config in enumerate(configs, start=1):
        cache_key = (
            config["target_tokens"],
            config["min_tokens"],
            config["max_tokens"],
            config["overlap_tokens"],
        )
        index_dir = index_root / f"idx_{cache_key[0]}_{cache_key[1]}_{cache_key[2]}_{cache_key[3]}"

        if cache_key not in built_cache:
            build_index(
                pdf_paths=[pdf_path],
                index_dir=index_dir,
                target_tokens=config["target_tokens"],
                min_tokens=config["min_tokens"],
                max_tokens=config["max_tokens"],
                overlap_tokens=config["overlap_tokens"],
                section_aware=args.section_aware,
                batch_size=args.batch_size,
                embedding_model=embedding_model,
                dry_run=False,
                base_url=args.base_url,
            )
            built_cache.add(cache_key)

        payload = evaluate_retrieval(
            index_dir=index_dir,
            dataset_path=dataset_path,
            company=company,
            top_k=config["top_k"],
            embedding_model=embedding_model,
            retrieval_architecture=config["retrieval_mode"],
            search_breadth=max(config["top_k"], args.candidate_k),
            base_url=args.base_url,
        )

        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        avg_chunk_tokens = round(manifest.get("chunk_count", 0) and (
            sum(c["token_count"] for c in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))) / manifest["chunk_count"]
        ) or 0, 1)

        summary = payload["summary"]
        grid_results.append(
            {
                "config": config,
                "index_dir": str(index_dir),
                "chunk_count": manifest.get("chunk_count", 0),
                "avg_chunk_tokens": avg_chunk_tokens,
                "hit": summary["hit"],
                "partial": summary["partial"],
                "miss": summary["miss"],
                "hit_rate": summary["hit_rate"],
                "partial_or_hit_rate": summary["partial_or_hit_rate"],
                "average_best_match_score": summary["average_best_match_score"],
                "average_top_retrieval_score": summary["average_top_retrieval_score"],
                "results": payload["results"],
            }
        )

        print(
            f"  [{idx}/{len(configs)}] "
            f"t={config['target_tokens']} min={config['min_tokens']} max={config['max_tokens']} "
            f"ov={config['overlap_tokens']} top_k={config['top_k']} mode={config['retrieval_mode']} "
            f"-> hit_rate={summary['hit_rate']:.1%} partial+hit={summary['partial_or_hit_rate']:.1%} "
            f"best_score={summary['average_best_match_score']:.1f}"
        )

    grid_results.sort(
        key=lambda r: (
            r["hit_rate"],
            r["partial_or_hit_rate"],
            r["average_best_match_score"],
            -r["avg_chunk_tokens"],
        ),
        reverse=True,
    )

    print("\nTop 10 configurations:")
    for rank, result in enumerate(grid_results[:10], start=1):
        c = result["config"]
        print(
            f"  #{rank} hit_rate={result['hit_rate']:.1%} "
            f"partial+hit={result['partial_or_hit_rate']:.1%} "
            f"best_score={result['average_best_match_score']:.1f} "
            f"top_ret_score={result['average_top_retrieval_score']:.4f} "
            f"chunks={result['chunk_count']} avg_tok={result['avg_chunk_tokens']:.0f} "
            f"| t={c['target_tokens']} min={c['min_tokens']} max={c['max_tokens']} "
            f"ov={c['overlap_tokens']} top_k={c['top_k']} mode={c['retrieval_mode']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "company": company,
        "dataset": str(dataset_path),
        "embedding_model": embedding_model,
        "grid_parameters": {
            "chunk_target_tokens": args.chunk_targets,
            "chunk_min_tokens": args.chunk_mins,
            "chunk_max_tokens": args.chunk_maxs,
            "chunk_overlap_tokens": args.chunk_overlaps,
            "top_k_values": args.top_ks,
            "retrieval_modes": args.retrieval_modes,
        },
        "total_configurations": len(grid_results),
        "results": grid_results,
    }
    DEFAULT_GRID_OUTPUT_JSON.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_headers = [
        "rank", "target_tokens", "min_tokens", "max_tokens", "overlap_tokens",
        "top_k", "retrieval_mode", "chunk_count", "avg_chunk_tokens",
        "hit", "partial", "miss", "hit_rate", "partial_or_hit_rate",
        "average_best_match_score", "average_top_retrieval_score",
    ]
    with DEFAULT_GRID_OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_headers)
        writer.writeheader()
        for rank, result in enumerate(grid_results, start=1):
            c = result["config"]
            writer.writerow(
                {
                    "rank": rank,
                    "target_tokens": c["target_tokens"],
                    "min_tokens": c["min_tokens"],
                    "max_tokens": c["max_tokens"],
                    "overlap_tokens": c["overlap_tokens"],
                    "top_k": c["top_k"],
                    "retrieval_mode": c["retrieval_mode"],
                    "chunk_count": result["chunk_count"],
                    "avg_chunk_tokens": result["avg_chunk_tokens"],
                    "hit": result["hit"],
                    "partial": result["partial"],
                    "miss": result["miss"],
                    "hit_rate": result["hit_rate"],
                    "partial_or_hit_rate": result["partial_or_hit_rate"],
                    "average_best_match_score": result["average_best_match_score"],
                    "average_top_retrieval_score": result["average_top_retrieval_score"],
                }
            )

    if grid_results:
        best = grid_results[0]
        best_config = {
            "company": company,
            "embedding_model": embedding_model,
            "chunk_config": {
                "target_tokens": best["config"]["target_tokens"],
                "min_tokens": best["config"]["min_tokens"],
                "max_tokens": best["config"]["max_tokens"],
                "overlap_tokens": best["config"]["overlap_tokens"],
            },
            "retrieval_config": {
                "top_k": best["config"]["top_k"],
                "retrieval_mode": best["config"]["retrieval_mode"],
            },
            "metrics": {
                "hit_rate": best["hit_rate"],
                "partial_or_hit_rate": best["partial_or_hit_rate"],
                "average_best_match_score": best["average_best_match_score"],
                "average_top_retrieval_score": best["average_top_retrieval_score"],
            },
        }
        DEFAULT_BEST_CONFIG.write_text(json.dumps(best_config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nBest configuration saved to {DEFAULT_BEST_CONFIG}")

    print(f"\nGrid results saved to {DEFAULT_GRID_OUTPUT_JSON} and {DEFAULT_GRID_OUTPUT_CSV}")
    return 0
