"""Iterative tuning workflow for the local ESG RAG pipeline."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from app.rag import (
    DEFAULT_BASE_URL,
    AlbertClient,
    VALID_PROMPT_STYLES,
    answer_question,
    build_index,
    gather_pdf_paths,
    require_api_key,
    retrieve_chunks,
)
from app.rag_eval import (
    DEFAULT_EVAL_DATASET,
    _load_rows,
    _parse_expected_contexts,
    _score_retrieval,
    get_ground_truth_answer,
    get_topic_label,
    score_answer_text,
)
from app.utils import OUTPUT_DIR

DEFAULT_TUNE_OUTPUT = OUTPUT_DIR / "rag_tune_results.json"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "item"


def _pick_pdf_for_company(company: str, pdf_paths: list[Path]) -> Path:
    best_path = None
    best_score = -1.0
    for pdf_path in pdf_paths:
        stem = pdf_path.stem.replace("_", " ").replace("-", " ")
        score = max(
            float(fuzz.token_set_ratio(company.lower(), stem.lower())),
            float(fuzz.partial_ratio(company.lower(), stem.lower())),
        )
        if score > best_score:
            best_score = score
            best_path = pdf_path
    if best_path is None:
        raise RuntimeError(f"Could not match a sample PDF to company '{company}'.")
    return best_path


def _discover_pdf_paths(sample_dir: Path) -> list[Path]:
    discovered = {path.resolve() for path in sample_dir.rglob("*.pdf")}
    discovered.update(gather_pdf_paths(None))
    return sorted(discovered)


def _resolve_pdf_catalog(rows: list[dict[str, str]], pdf_paths: list[Path]) -> dict[str, Path]:
    by_name: dict[str, list[Path]] = {}
    for pdf_path in pdf_paths:
        by_name.setdefault(pdf_path.name, []).append(pdf_path)

    resolved: dict[str, Path] = {}
    source_files = sorted({row.get("source_file", "").strip() for row in rows if row.get("source_file", "").strip()})
    missing: list[str] = []
    for source_file in source_files:
        matches = by_name.get(source_file, [])
        if len(matches) == 1:
            resolved[source_file] = matches[0]
        elif matches:
            # Prefer paths that look like curated/raw database content.
            resolved[source_file] = sorted(matches, key=lambda path: ("/sample_data/" not in str(path), len(str(path))))[0]
        else:
            missing.append(source_file)

    if missing:
        missing_preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Could not resolve {len(missing)} dataset source files to local PDFs. First missing: {missing_preview}"
        )
    return resolved


def _derive_chunk_bounds(target_tokens: int) -> tuple[int, int]:
    min_tokens = max(120, int(round(target_tokens * 0.72)))
    max_tokens = max(target_tokens + 40, int(round(target_tokens * 1.22)))
    if min_tokens >= target_tokens:
        min_tokens = max(100, target_tokens - 40)
    if max_tokens <= target_tokens:
        max_tokens = target_tokens + 40
    return min_tokens, max_tokens


def _build_chunk_config(target_tokens: int, overlap_tokens: int) -> dict[str, int]:
    min_tokens, max_tokens = _derive_chunk_bounds(target_tokens)
    return {
        "target_tokens": target_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
    }


def _chunk_config_label(chunk_config: dict[str, int]) -> str:
    return (
        f"target={chunk_config['target_tokens']}, min={chunk_config['min_tokens']}, "
        f"max={chunk_config['max_tokens']}, overlap={chunk_config['overlap_tokens']}"
    )


def _summarize_stage_result(result: dict[str, Any]) -> str:
    summary = result["summary"]
    parts = [
        f"retrieval={summary['avg_retrieval_score']:.2f}",
        f"retrieval_hit_rate={summary['retrieval_hit_rate']:.1%}",
    ]
    if summary["avg_answer_score"] is not None:
        parts.append(f"answer={summary['avg_answer_score']:.2f}")
        parts.append(f"answer_hit_rate={summary['answer_hit_rate']:.1%}")
    parts.append(f"composite={summary['composite_score']:.2f}")
    return " | ".join(parts)


def _refine_grid(best_value: int, *, step: int, minimum: int) -> list[int]:
    return sorted({max(minimum, best_value - step), best_value, best_value + step})


def _build_or_reuse_index(
    *,
    company: str,
    pdf_path: Path,
    chunk_config: dict[str, int],
    index_root: Path,
    built_cache: set[tuple[str, int, int]],
    embedding_model: str,
    batch_size: int,
    base_url: str,
) -> Path:
    cache_key = (
        f"{company}:{embedding_model}:{index_root}",
        chunk_config["target_tokens"],
        chunk_config["overlap_tokens"],
    )
    index_dir = index_root / _slugify(company) / (
        f"{_slugify(embedding_model)}_t{chunk_config['target_tokens']}_ov{chunk_config['overlap_tokens']}"
    )
    if cache_key not in built_cache:
        build_index(
            pdf_paths=[pdf_path],
            index_dir=index_dir,
            target_tokens=chunk_config["target_tokens"],
            min_tokens=chunk_config["min_tokens"],
            max_tokens=chunk_config["max_tokens"],
            overlap_tokens=chunk_config["overlap_tokens"],
            batch_size=batch_size,
            embedding_model=embedding_model,
            dry_run=False,
            base_url=base_url,
        )
        built_cache.add(cache_key)
    return index_dir


def evaluate_configuration(
    *,
    source_rows: dict[str, list[dict[str, str]]],
    source_pdfs: dict[str, Path],
    chunk_config: dict[str, int],
    top_k: int,
    embedding_model: str,
    retrieval_architecture: str,
    search_breadth: int,
    text_model: str,
    temperature: float,
    prompt_style: str,
    index_root: Path,
    built_cache: set[tuple[str, int, int]],
    batch_size: int,
    base_url: str,
    generate_answers: bool,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for source_file, rows in source_rows.items():
        company = rows[0].get("company", "Unknown")
        pdf_path = source_pdfs[source_file]
        index_dir = _build_or_reuse_index(
            company=source_file,
            pdf_path=pdf_path,
            chunk_config=chunk_config,
            index_root=index_root,
            built_cache=built_cache,
            embedding_model=embedding_model,
            batch_size=batch_size,
            base_url=base_url,
        )

        for row in rows:
            expected_contexts = _parse_expected_contexts(row)
            retrieved_chunks, resolved_embedding_model = retrieve_chunks(
                index_dir=index_dir,
                question=row["question"],
                top_k=top_k,
                embedding_model=embedding_model,
                retrieval_architecture=retrieval_architecture,
                search_breadth=search_breadth,
                base_url=base_url,
            )
            retrieval_scoring = _score_retrieval(expected_contexts, retrieved_chunks)

            answer = None
            answer_scoring = None
            resolved_text_model = text_model
            if generate_answers:
                answer, resolved_text_model = answer_question(
                    question=row["question"],
                    retrieved_chunks=retrieved_chunks,
                    text_model=text_model,
                    temperature=temperature,
                    prompt_style=prompt_style,
                    base_url=base_url,
                )
                answer_scoring = score_answer_text(answer, get_ground_truth_answer(row), expected_contexts)

            results.append(
                {
                    "company": company,
                    "source_file": source_file,
                    "question": row["question"],
                    "topic": get_topic_label(row),
                    "ground_truth": get_ground_truth_answer(row),
                    "retrieval_status": retrieval_scoring["status"],
                    "retrieval_score": retrieval_scoring["best_match_score"],
                    "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
                    "best_chunk_id": retrieval_scoring["best_chunk_id"],
                    "top_score": round(float(retrieved_chunks[0]["score"]), 4) if retrieved_chunks else None,
                    "answer_status": answer_scoring["status"] if answer_scoring else None,
                    "answer_score": answer_scoring["score"] if answer_scoring else None,
                    "answer": answer,
                    "embedding_model": resolved_embedding_model,
                    "text_model": resolved_text_model,
                    "prompt_style": prompt_style,
                }
            )

    retrieval_scores = [float(row["retrieval_score"]) for row in results]
    answer_scores = [float(row["answer_score"]) for row in results if row["answer_score"] is not None]
    retrieval_counts = Counter(row["retrieval_status"] for row in results)
    answer_counts = Counter(row["answer_status"] for row in results if row["answer_status"])

    avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
    avg_answer_score = (sum(answer_scores) / len(answer_scores)) if answer_scores else None
    retrieval_hit_rate = retrieval_counts.get("hit", 0) / len(results) if results else 0.0
    answer_hit_rate = (
        answer_counts.get("hit", 0) / len(answer_scores) if answer_scores else None
    )

    if avg_answer_score is None:
        composite_score = avg_retrieval_score
    else:
        composite_score = (0.45 * avg_retrieval_score) + (0.55 * avg_answer_score)

    by_company: dict[str, dict[str, float]] = {}
    companies = sorted({rows[0].get("company", "Unknown") for rows in source_rows.values() if rows})
    for company in companies:
        company_result_rows = [row for row in results if row["company"] == company]
        company_answer_scores = [float(row["answer_score"]) for row in company_result_rows if row["answer_score"] is not None]
        by_company[company] = {
            "avg_retrieval_score": round(
                sum(float(row["retrieval_score"]) for row in company_result_rows) / len(company_result_rows),
                2,
            ),
            "avg_answer_score": (
                round(sum(company_answer_scores) / len(company_answer_scores), 2) if company_answer_scores else None
            ),
        }

    return {
        "chunk_config": chunk_config,
        "retrieval_architecture": retrieval_architecture,
        "search_breadth": search_breadth,
        "temperature": temperature,
        "prompt_style": prompt_style,
        "summary": {
            "question_count": len(results),
            "avg_retrieval_score": round(avg_retrieval_score, 2),
            "retrieval_hit_rate": round(retrieval_hit_rate, 3),
            "avg_answer_score": round(avg_answer_score, 2) if avg_answer_score is not None else None,
            "answer_hit_rate": round(answer_hit_rate, 3) if answer_hit_rate is not None else None,
            "composite_score": round(composite_score, 2),
        },
        "by_company": by_company,
        "results": results,
    }


def _pick_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("No tuning results were produced for this stage.")
    return max(results, key=lambda item: item["summary"]["composite_score"])


def _stage_candidates_from_chunk_grid(targets: list[int], overlaps: list[int]) -> list[dict[str, int]]:
    return [_build_chunk_config(target, overlap) for target in targets for overlap in overlaps]


def run_rag_tune(args: Any) -> int:
    require_api_key()
    rows = _load_rows(args.dataset)
    companies = sorted({row["company"] for row in rows if row.get("company")})
    if args.company:
        rows = [row for row in rows if row.get("company") == args.company]
        companies = [company for company in companies if company == args.company]
        if not rows:
            raise RuntimeError(f"No rows found for company '{args.company}'.")

    source_rows = {
        source_file: [row for row in rows if row.get("source_file") == source_file]
        for source_file in sorted({row.get("source_file", "").strip() for row in rows if row.get("source_file", "").strip()})
    }
    pdf_paths = _discover_pdf_paths(args.sample_dir)
    if not pdf_paths:
        raise RuntimeError(f"No PDFs were found under {args.sample_dir}.")
    source_pdfs = _resolve_pdf_catalog(rows, pdf_paths)

    client = AlbertClient(api_key=require_api_key(), base_url=args.base_url)
    embedding_models = args.embedding_models or ([args.embedding_model] if args.embedding_model else None)
    if not embedding_models:
        embedding_models = [client.get_embedding_model(preferred="bge-m3")]
    text_model = args.text_model or client.get_text_generation_model()
    index_root = args.index_root
    index_root.mkdir(parents=True, exist_ok=True)
    built_cache: set[tuple[str, int, int]] = set()

    chunk_targets = args.chunk_targets or [320, 420, 520]
    chunk_overlaps = args.chunk_overlaps or [0, 60, 120]
    retrieval_architectures = (
        getattr(args, "retrieval_modes", None)
        or getattr(args, "retrieval_architectures", None)
        or ["dense", "hybrid", "lexical"]
    )
    search_breadths = args.search_breadths or [5, 8, 12]
    temperatures = args.temperatures or [0.0, 0.1, 0.2]
    prompt_styles = args.prompt_styles or ["balanced", "extractive", "audit"]
    prompt_styles = [style for style in prompt_styles if style in VALID_PROMPT_STYLES]

    embedding_results: list[dict[str, Any]] = []
    print(f"Stage 0/5: embedding models across {len(embedding_models)} configurations")
    for index, embedding_model in enumerate(embedding_models, start=1):
        result = evaluate_configuration(
            source_rows=source_rows,
            source_pdfs=source_pdfs,
            chunk_config=_build_chunk_config(chunk_targets[0], chunk_overlaps[0]),
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture="semantic",
            search_breadth=max(args.top_k, search_breadths[0]),
            text_model=text_model,
            temperature=temperatures[0],
            prompt_style=prompt_styles[0],
            index_root=index_root / _slugify(embedding_model),
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=False,
        )
        embedding_results.append(result)
        print(f"  [{index}/{len(embedding_models)}] {embedding_model} -> {_summarize_stage_result(result)}")
    best_embedding_result = _pick_best(embedding_results)
    embedding_model = best_embedding_result["results"][0]["embedding_model"] if best_embedding_result["results"] else embedding_models[0]

    stages: list[dict[str, Any]] = []
    stages.append({"name": "embedding_model", "candidates": embedding_results, "best": best_embedding_result})

    coarse_chunk_candidates = _stage_candidates_from_chunk_grid(chunk_targets, chunk_overlaps)
    coarse_results: list[dict[str, Any]] = []
    print(f"Stage 1/5: coarse chunking across {len(coarse_chunk_candidates)} configurations")
    for index, chunk_config in enumerate(coarse_chunk_candidates, start=1):
        result = evaluate_configuration(
            source_rows=source_rows,
            source_pdfs=source_pdfs,
            chunk_config=chunk_config,
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture="semantic",
            search_breadth=max(args.top_k, search_breadths[0]),
            text_model=text_model,
            temperature=temperatures[0],
            prompt_style=prompt_styles[0],
            index_root=index_root,
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=False,
        )
        coarse_results.append(result)
        print(f"  [{index}/{len(coarse_chunk_candidates)}] {_chunk_config_label(chunk_config)} -> {_summarize_stage_result(result)}")
    best_chunk_result = _pick_best(coarse_results)
    stages.append({"name": "coarse_chunking", "candidates": coarse_results, "best": best_chunk_result})

    retrieval_results: list[dict[str, Any]] = []
    retrieval_candidates = [
        (architecture, breadth)
        for architecture in retrieval_architectures
        for breadth in search_breadths
        if breadth >= args.top_k
    ]
    print(f"Stage 2/5: retrieval architecture and breadth across {len(retrieval_candidates)} configurations")
    for index, (architecture, breadth) in enumerate(retrieval_candidates, start=1):
        result = evaluate_configuration(
            source_rows=source_rows,
            source_pdfs=source_pdfs,
            chunk_config=best_chunk_result["chunk_config"],
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture=architecture,
            search_breadth=breadth,
            text_model=text_model,
            temperature=temperatures[0],
            prompt_style=prompt_styles[0],
            index_root=index_root,
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=False,
        )
        retrieval_results.append(result)
        print(f"  [{index}/{len(retrieval_candidates)}] {architecture}, breadth={breadth} -> {_summarize_stage_result(result)}")
    best_retrieval_result = _pick_best(retrieval_results)
    stages.append({"name": "retrieval_architecture", "candidates": retrieval_results, "best": best_retrieval_result})

    prompt_results: list[dict[str, Any]] = []
    prompt_candidates = [(prompt_style, temperature) for prompt_style in prompt_styles for temperature in temperatures]
    print(f"Stage 3/5: answer prompt style and temperature across {len(prompt_candidates)} configurations")
    for index, (prompt_style, temperature) in enumerate(prompt_candidates, start=1):
        result = evaluate_configuration(
            source_rows=source_rows,
            source_pdfs=source_pdfs,
            chunk_config=best_chunk_result["chunk_config"],
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture=best_retrieval_result["retrieval_architecture"],
            search_breadth=best_retrieval_result["search_breadth"],
            text_model=text_model,
            temperature=temperature,
            prompt_style=prompt_style,
            index_root=index_root,
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=True,
        )
        prompt_results.append(result)
        print(
            f"  [{index}/{len(prompt_candidates)}] prompt={prompt_style}, temperature={temperature:.2f} "
            f"-> {_summarize_stage_result(result)}"
        )
    best_prompt_result = _pick_best(prompt_results)
    stages.append({"name": "prompt_style_temperature", "candidates": prompt_results, "best": best_prompt_result})

    refined_targets = _refine_grid(best_chunk_result["chunk_config"]["target_tokens"], step=60, minimum=180)
    refined_overlaps = _refine_grid(best_chunk_result["chunk_config"]["overlap_tokens"], step=30, minimum=0)
    refined_chunk_candidates = _stage_candidates_from_chunk_grid(refined_targets, refined_overlaps)
    refined_results: list[dict[str, Any]] = []
    print(f"Stage 4/5: refined chunking across {len(refined_chunk_candidates)} configurations")
    for index, chunk_config in enumerate(refined_chunk_candidates, start=1):
        result = evaluate_configuration(
            source_rows=source_rows,
            source_pdfs=source_pdfs,
            chunk_config=chunk_config,
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture=best_retrieval_result["retrieval_architecture"],
            search_breadth=best_retrieval_result["search_breadth"],
            text_model=text_model,
            temperature=best_prompt_result["temperature"],
            prompt_style=best_prompt_result["prompt_style"],
            index_root=index_root,
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=True,
        )
        refined_results.append(result)
        print(f"  [{index}/{len(refined_chunk_candidates)}] {_chunk_config_label(chunk_config)} -> {_summarize_stage_result(result)}")
    best_refined_result = _pick_best(refined_results)
    stages.append({"name": "refined_chunking", "candidates": refined_results, "best": best_refined_result})

    final_retrieval_candidates = [
        (best_retrieval_result["retrieval_architecture"], best_retrieval_result["search_breadth"]),
        ("hybrid", max(best_retrieval_result["search_breadth"], args.top_k)),
        ("semantic_rerank", max(best_retrieval_result["search_breadth"], args.top_k)),
    ]
    final_retrieval_candidates = list(dict.fromkeys(final_retrieval_candidates))
    final_retrieval_results: list[dict[str, Any]] = []
    print(f"Stage 5/5: final retrieval confirmation across {len(final_retrieval_candidates)} configurations")
    for index, (architecture, breadth) in enumerate(final_retrieval_candidates, start=1):
        result = evaluate_configuration(
            source_rows=source_rows,
            source_pdfs=source_pdfs,
            chunk_config=best_refined_result["chunk_config"],
            top_k=args.top_k,
            embedding_model=embedding_model,
            retrieval_architecture=architecture,
            search_breadth=breadth,
            text_model=text_model,
            temperature=best_prompt_result["temperature"],
            prompt_style=best_prompt_result["prompt_style"],
            index_root=index_root,
            built_cache=built_cache,
            batch_size=args.batch_size,
            base_url=args.base_url,
            generate_answers=True,
        )
        final_retrieval_results.append(result)
        print(f"  [{index}/{len(final_retrieval_candidates)}] {architecture}, breadth={breadth} -> {_summarize_stage_result(result)}")
    best_final_retrieval = _pick_best(final_retrieval_results)
    stages.append({"name": "final_retrieval_confirmation", "candidates": final_retrieval_results, "best": best_final_retrieval})

    all_ranked_results = [best_prompt_result, best_refined_result, best_final_retrieval]
    overall_best = _pick_best(all_ranked_results)

    payload = {
        "dataset": str(args.dataset),
        "sample_dir": str(args.sample_dir),
        "companies": companies,
        "embedding_model": embedding_model,
        "embedding_models_tested": embedding_models,
        "text_model": text_model,
        "top_k": args.top_k,
        "stages": stages,
        "best_configuration": overall_best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nBest configuration:")
    print(f"  chunking: {_chunk_config_label(overall_best['chunk_config'])}")
    print(
        f"  retrieval: {overall_best['retrieval_architecture']} "
        f"(breadth={overall_best['search_breadth']})"
    )
    print(f"  prompt_style: {overall_best['prompt_style']}")
    print(f"  temperature: {overall_best['temperature']:.2f}")
    print(f"  metrics: {_summarize_stage_result(overall_best)}")
    print(f"Saved tuning results to {args.output}")
    return 0
