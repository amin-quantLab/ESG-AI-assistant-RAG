"""Qrels-based retrieval evaluation for the RAGAS ESG v2 dataset."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.rag import (
    DEFAULT_BASE_URL,
    AlbertClient,
    ChunkRecord,
    _company_label_from_source,
    _load_chunks,
    _mentioned_company_labels,
    _normalize_retrieval_mode,
    _ranking_dimensions,
    _search_vectors_for_indices,
    _store_index,
    embed_texts,
    require_api_key,
)
from app.reranker import rerank_candidates
from app.utils import OUTPUT_DIR

DEFAULT_RAGAS_V2_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "ragas_esg_dataset_v2_retrieval"
DEFAULT_RAGAS_V2_DATASET = DEFAULT_RAGAS_V2_DIR / "ragas_esg_eval_dataset_v2.csv"
DEFAULT_RAGAS_V2_CORPUS = DEFAULT_RAGAS_V2_DIR / "eval_corpus_sampled_30_docs.csv"
DEFAULT_RAGAS_V2_QRELS = DEFAULT_RAGAS_V2_DIR / "qrels.csv"
DEFAULT_RAGAS_V2_INDEX_DIR = OUTPUT_DIR / "ragas_v2_corpus_index"
DEFAULT_RAGAS_V2_OUTPUT_DIR = OUTPUT_DIR / "ragas_v2_retrieval"
DEFAULT_RANDOM_SEED = 42

ESG_SYNONYM_GROUPS: dict[str, list[str]] = {
    "ghg": ["ghg", "greenhouse gas", "greenhouse gases", "emissions", "co2e", "co2"],
    "scope": ["scope 1", "scope 2", "scope 3", "scope i", "scope ii", "scope iii"],
    "safety": ["ltifr", "trir", "safety", "injury rate", "lost time injury", "recordable incident"],
    "dei": ["dei", "diversity", "inclusion", "gender balance", "women in management"],
    "water": ["water withdrawal", "water consumption", "water discharge", "water"],
    "waste": ["waste", "recycling", "circularity", "circular economy"],
    "finance": ["taxonomy-aligned", "sustainable finance", "green bonds", "taxonomy"],
}

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "emissions": ["emission", "ghg", "co2", "co2e", "carbon", "net zero"],
    "water": ["water", "withdrawal", "discharge", "consumption"],
    "waste": ["waste", "recycling", "circular", "circularity"],
    "safety": ["ltifr", "trir", "injury", "safety"],
    "diversity_inclusion": ["diversity", "inclusion", "dei", "gender", "women"],
    "finance": ["taxonomy", "green bond", "sustainable finance"],
    "biodiversity": ["biodiversity", "nature", "ecosystem"],
    "human_rights": ["human rights", "due diligence", "supply chain"],
    "circularity": ["circular", "recycling", "reuse"],
    "targets": ["target", "ambition", "commitment"],
}

REPORT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "sustainability": ["sustainability"],
    "integrated": ["integrated"],
    "annual": ["annual"],
    "climate": ["climate"],
    "esg": ["esg"],
    "framework": ["framework"],
    "thematic": ["thematic"],
    "urd": ["universal registration", "urd"],
}


def _bool_from_cell(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_json_list(raw: str) -> list[str]:
    value = (raw or "").strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in value.split(";") if part.strip()]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class QueryEntities:
    companies: list[str] = field(default_factory=list)
    years: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    report_types: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    comparison: bool = False


@dataclass
class BM25Index:
    doc_term_counts: list[Counter[str]]
    document_frequency: Counter[str]
    document_lengths: list[int]
    avg_length: float
    k1: float = 1.2
    b: float = 0.75


def _infer_report_type(text: str) -> str:
    lowered = text.lower()
    for report_type, keywords in REPORT_TYPE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return report_type
    return ""


def _infer_topic_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    for tag, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(tag)
    return tags


def _build_company_aliases(rows: list[dict[str, str]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for row in rows:
        company = (row.get("company") or "").strip()
        if not company:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", company.lower()).strip()
        if normalized:
            aliases[normalized] = company
        aliases[company.lower()] = company
    return aliases


def _extract_query_entities(
    *,
    question: str,
    row: dict[str, str],
    company_aliases: dict[str, str],
) -> QueryEntities:
    normalized_question = _normalize_text(question)
    companies = _parse_json_list(row.get("comparison_companies", ""))
    if not companies:
        companies = []
        for alias, label in company_aliases.items():
            if re.search(rf"\b{re.escape(alias)}\b", normalized_question):
                companies.append(label)
    if not companies and row.get("company"):
        companies = [row["company"].strip()]

    years = sorted(set(re.findall(r"\b20\d{2}\b", normalized_question)))
    comparison_years = _parse_json_list(row.get("comparison_years", ""))
    if comparison_years:
        years = [str(year) for year in comparison_years if str(year).strip()]

    topics = _parse_json_list(row.get("topic_tags", ""))
    if not topics:
        topics = _infer_topic_tags(normalized_question)

    metrics: list[str] = []
    for group_name, synonyms in ESG_SYNONYM_GROUPS.items():
        if any(term in normalized_question for term in synonyms):
            metrics.append(group_name)

    report_types = [
        report_type
        for report_type, keywords in REPORT_TYPE_KEYWORDS.items()
        if any(keyword in normalized_question for keyword in keywords)
    ]
    sectors = [row["sector"].strip()] if row.get("sector") else []

    comparison = bool(
        _bool_from_cell(row.get("is_cross_document"))
        or re.search(r"\bcompare|versus|vs\.|difference|between\b", normalized_question)
    )
    return QueryEntities(
        companies=companies,
        years=years,
        topics=topics,
        metrics=metrics,
        report_types=report_types,
        sectors=sectors,
        comparison=comparison,
    )


def _expand_query_with_synonyms(question: str, entities: QueryEntities) -> str:
    normalized = _normalize_text(question)
    expanded_terms: list[str] = []
    for synonyms in ESG_SYNONYM_GROUPS.values():
        if any(term in normalized for term in synonyms):
            expanded_terms.extend(synonyms)
    for topic in entities.topics:
        expanded_terms.extend(TOPIC_KEYWORDS.get(topic, []))
    for report_type in entities.report_types:
        expanded_terms.extend(REPORT_TYPE_KEYWORDS.get(report_type, []))
    expanded_terms = sorted({term for term in expanded_terms if term})
    if not expanded_terms:
        return question
    return f"{question}\nSynonyms: {'; '.join(expanded_terms)}"


def _build_bm25_index(rows: list[dict[str, str]]) -> BM25Index:
    doc_term_counts: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []

    for row in rows:
        metadata_parts = [
            row.get("company", ""),
            row.get("display_title", ""),
            row.get("source_file", ""),
            row.get("report_year", ""),
            row.get("sector", ""),
            row.get("section_title", ""),
            row.get("esg_pillar", ""),
        ]
        topic_tags = _parse_json_list(row.get("topic_tags", ""))
        metadata_parts.extend(topic_tags)
        metadata = " ".join(part for part in metadata_parts if part)
        text = (row.get("text") or "").strip()
        tokens = Counter(_tokenize(f"{metadata} {text}"))
        doc_term_counts.append(tokens)
        doc_length = sum(tokens.values())
        document_lengths.append(doc_length)
        for term in tokens:
            document_frequency[term] += 1

    avg_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
    return BM25Index(
        doc_term_counts=doc_term_counts,
        document_frequency=document_frequency,
        document_lengths=document_lengths,
        avg_length=avg_length,
    )


def _bm25_scores(
    query: str,
    bm25_index: BM25Index,
    candidate_indices: list[int] | None = None,
) -> dict[int, float]:
    query_terms = Counter(_tokenize(query))
    if not query_terms:
        return {}
    selected_indices = candidate_indices or list(range(len(bm25_index.doc_term_counts)))
    scores: dict[int, float] = {}
    doc_count = len(bm25_index.doc_term_counts)
    for index in selected_indices:
        term_counts = bm25_index.doc_term_counts[index]
        if not term_counts:
            continue
        doc_length = bm25_index.document_lengths[index]
        score = 0.0
        for term, query_count in query_terms.items():
            term_frequency = term_counts.get(term, 0)
            if term_frequency <= 0:
                continue
            doc_frequency = bm25_index.document_frequency.get(term, 0)
            idf = math.log(1.0 + ((doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5)))
            length_penalty = 1.0 - bm25_index.b + bm25_index.b * (doc_length / max(bm25_index.avg_length, 1.0))
            numerator = term_frequency * (bm25_index.k1 + 1.0)
            denominator = term_frequency + (bm25_index.k1 * length_penalty)
            score += idf * (numerator / max(denominator, 1e-6)) * (0.5 + 0.5 * min(query_count, 3))
        if score > 0:
            scores[index] = score
    return scores


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    min_score = min(scores.values())
    if math.isclose(max_score, min_score):
        return {index: 1.0 for index in scores}
    span = max_score - min_score
    return {index: (score - min_score) / span for index, score in scores.items()}


def _score_metadata_boost(
    entry: dict[str, Any],
    entities: QueryEntities,
) -> float:
    score = 0.0
    company = str(entry.get("company", "")).strip()
    report_year = str(entry.get("report_year", "")).strip()
    sector = str(entry.get("sector", "")).strip()
    report_type = str(entry.get("report_type", "")).strip()
    section_title = str(entry.get("section_title", "")).lower()
    topic_tags = set(entry.get("topic_tags") or [])

    if entities.companies:
        if company in entities.companies:
            score += 0.35
        else:
            score -= 0.1
    if entities.years:
        if report_year in entities.years:
            score += 0.2
        elif report_year:
            score -= 0.05
    if entities.sectors and sector in entities.sectors:
        score += 0.08
    if entities.report_types and report_type in entities.report_types:
        score += 0.08
    if entities.topics:
        overlap = topic_tags & set(entities.topics)
        if overlap:
            score += 0.12 * (len(overlap) / max(len(set(entities.topics)), 1))
    if entities.topics and section_title:
        if any(topic.replace("_", " ") in section_title for topic in entities.topics):
            score += 0.06
    return score


def _build_adjacent_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_doc: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        source_file = row.get("source_file", "")
        corpus_id = row.get("corpus_id", "")
        try:
            chunk_index = int(float(row.get("chunk_index") or 0))
        except ValueError:
            chunk_index = 0
        if source_file and corpus_id and chunk_index:
            by_doc[source_file].append((chunk_index, corpus_id))

    adjacent: dict[str, dict[str, str]] = {}
    for source_file, items in by_doc.items():
        items.sort(key=lambda item: item[0])
        for idx, (_chunk_index, corpus_id) in enumerate(items):
            prev_id = items[idx - 1][1] if idx > 0 else ""
            next_id = items[idx + 1][1] if idx + 1 < len(items) else ""
            adjacent[corpus_id] = {"prev": prev_id, "next": next_id}
    return adjacent


@dataclass
class CorpusState:
    rows: list[dict[str, str]]
    metadata_by_id: dict[str, dict[str, Any]]
    index_by_id: dict[str, int]
    adjacent_map: dict[str, dict[str, str]]
    bm25_index: BM25Index
    company_aliases: dict[str, str]


def _load_corpus_state(corpus_path: Path, chunks: list[ChunkRecord]) -> CorpusState:
    rows = _read_csv_rows(corpus_path)
    metadata_by_id: dict[str, dict[str, Any]] = {}
    index_by_id: dict[str, int] = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    for row in rows:
        corpus_id = row.get("corpus_id", "")
        if not corpus_id:
            continue
        report_type = _infer_report_type(row.get("display_title", "") or row.get("source_file", ""))
        topic_tags = _parse_json_list(row.get("topic_tags", ""))
        if not topic_tags:
            topic_tags = _infer_topic_tags(row.get("text", "") or "")
        metadata_by_id[corpus_id] = {
            "corpus_id": corpus_id,
            "source_file": row.get("source_file", ""),
            "display_title": row.get("display_title", ""),
            "company": row.get("company", ""),
            "sector": row.get("sector", ""),
            "report_year": str(row.get("report_year", "")).replace(".0", ""),
            "report_style_bucket": row.get("report_style_bucket", ""),
            "page_start": row.get("page_start", ""),
            "page_end": row.get("page_end", ""),
            "chunk_id": row.get("chunk_id", ""),
            "chunk_index": row.get("chunk_index", ""),
            "section_title": row.get("section_title", ""),
            "esg_pillar": row.get("esg_pillar", ""),
            "contains_table": _bool_from_cell(row.get("contains_table")),
            "contains_targets": _bool_from_cell(row.get("contains_targets")),
            "report_type": report_type,
            "topic_tags": topic_tags,
        }

    adjacent_map = _build_adjacent_map(rows)
    bm25_index = _build_bm25_index(rows)
    company_aliases = _build_company_aliases(rows)
    return CorpusState(
        rows=rows,
        metadata_by_id=metadata_by_id,
        index_by_id=index_by_id,
        adjacent_map=adjacent_map,
        bm25_index=bm25_index,
        company_aliases=company_aliases,
    )


def _augment_corpus_text(row: dict[str, str]) -> str:
    topic_tags = _parse_json_list(row.get("topic_tags", ""))
    if not topic_tags:
        topic_tags = _infer_topic_tags(row.get("text", "") or "")
    parts = [
        f"Company: {row.get('company', '').strip()}",
        f"Document: {row.get('display_title', '').strip() or row.get('source_file', '').strip()}",
        f"Report year: {row.get('report_year', '').strip()}",
        f"Sector: {row.get('sector', '').strip()}",
        f"Section: {row.get('section_title', '').strip()}",
        f"ESG pillar: {row.get('esg_pillar', '').strip()}",
        f"Topics: {', '.join(topic_tags)}" if topic_tags else "",
    ]
    metadata = ". ".join(part for part in parts if part and not part.endswith(": "))
    text = (row.get("text") or "").strip()
    return f"{metadata}.\n{text}" if metadata else text


def build_ragas_v2_corpus_index(
    *,
    corpus_path: Path = DEFAULT_RAGAS_V2_CORPUS,
    index_dir: Path = DEFAULT_RAGAS_V2_INDEX_DIR,
    embedding_model: str | None = None,
    batch_size: int = 32,
    base_url: str = DEFAULT_BASE_URL,
    force: bool = False,
) -> dict[str, Any]:
    if not force and (index_dir / "manifest.json").exists() and (index_dir / "vectors.npy").exists():
        return json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))

    rows = _read_csv_rows(corpus_path)
    if not rows:
        raise RuntimeError(f"No corpus rows found in {corpus_path}.")

    index_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[ChunkRecord] = []
    texts_to_embed: list[str] = []
    corpus_lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        corpus_id = row["corpus_id"]
        source_file = row.get("source_file", "")
        company = row.get("company", "Unknown").strip() or "Unknown"
        source_path = f"/ragas_v2_eval_corpus/{company}/{source_file}"
        text = (row.get("text") or "").strip()
        chunk = ChunkRecord(
            chunk_id=corpus_id,
            source_file=source_file,
            source_path=source_path,
            page_start=int(float(row.get("page_start") or 0)),
            page_end=int(float(row.get("page_end") or 0)),
            token_count=len(text.split()),
            text=text,
            contextual_summary=row.get("display_title", ""),
            esg_pillar=row.get("esg_pillar", ""),
            section_title=row.get("section_title", ""),
            report_year=str(row.get("report_year", "")).replace(".0", ""),
            contains_table=_bool_from_cell(row.get("contains_table")),
            contains_targets=_bool_from_cell(row.get("contains_targets")),
        )
        chunks.append(chunk)
        texts_to_embed.append(_augment_corpus_text(row))
        corpus_lookup[corpus_id] = {
            "source_file": source_file,
            "company": company,
            "display_title": row.get("display_title", ""),
            "sector": row.get("sector", ""),
            "report_year": row.get("report_year", ""),
            "report_style_bucket": row.get("report_style_bucket", ""),
            "page_start": row.get("page_start", ""),
            "page_end": row.get("page_end", ""),
            "chunk_index": row.get("chunk_index", ""),
            "section_title": row.get("section_title", ""),
            "esg_pillar": row.get("esg_pillar", ""),
            "contains_table": row.get("contains_table", ""),
            "contains_targets": row.get("contains_targets", ""),
            "report_type": _infer_report_type(row.get("display_title", "") or source_file),
            "topic_tags": _parse_json_list(row.get("topic_tags", "")),
        }

    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (index_dir / "corpus_lookup.json").write_text(
        json.dumps(corpus_lookup, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    selected_embedding_model = embedding_model or client.get_embedding_model(preferred="bge-m3")
    vectors = embed_texts(
        client,
        texts_to_embed,
        embedding_model=selected_embedding_model,
        batch_size=batch_size,
    )
    vector_backend = _store_index(index_dir, vectors)
    manifest = {
        "built_at": datetime.now(UTC).isoformat(),
        "corpus_path": str(corpus_path),
        "chunk_count": len(chunks),
        "embedding_model": selected_embedding_model,
        "embedding_dimension": int(vectors.shape[1]),
        "vector_backend": vector_backend,
        "index_kind": "ragas_v2_eval_corpus",
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _read_csv_rows(path):
        qrels[str(row["question_id"])][str(row["corpus_id"])] = float(row["relevance_grade"])
    return qrels


def _dcg(grades: list[float]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _metrics_at_k(
    *,
    qrels: dict[str, dict[str, float]],
    results: dict[str, list[str]],
    k: int,
    strict_min_grade: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_id, relevance_map in qrels.items():
        strict_relevant = {cid for cid, grade in relevance_map.items() if grade >= strict_min_grade}
        if not strict_relevant:
            continue
        retrieved = results.get(question_id, [])[:k]
        retrieved_relevant = [cid for cid in retrieved if cid in strict_relevant]
        reciprocal_rank = 0.0
        for index, corpus_id in enumerate(retrieved, start=1):
            if corpus_id in strict_relevant:
                reciprocal_rank = 1.0 / index
                break
        grades = [relevance_map.get(corpus_id, 0.0) for corpus_id in retrieved]
        ideal_grades = sorted([grade for grade in relevance_map.values() if grade > 0], reverse=True)[:k]
        ideal_dcg = _dcg(ideal_grades)
        rows.append(
            {
                "question_id": question_id,
                f"precision@{k}": len(retrieved_relevant) / k,
                f"recall@{k}": len(set(retrieved_relevant)) / len(strict_relevant),
                f"hit_rate@{k}": 1.0 if retrieved_relevant else 0.0,
                f"mrr@{k}": reciprocal_rank,
                f"ndcg@{k}": _dcg(grades) / ideal_dcg if ideal_dcg else 0.0,
                "num_strict_relevant": len(strict_relevant),
            }
        )
    return rows


def compute_qrels_metrics(
    *,
    qrels: dict[str, dict[str, float]],
    results: dict[str, list[str]],
    k_values: list[int],
    strict_min_grade: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_question_by_id: dict[str, dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []
    for k in k_values:
        rows = _metrics_at_k(qrels=qrels, results=results, k=k, strict_min_grade=strict_min_grade)
        if not rows:
            continue
        metric_keys = [key for key in rows[0] if key not in {"question_id", "num_strict_relevant"}]
        summary_row = {"k": k, "question_count": len(rows)}
        summary_row.update({key: sum(float(row[key]) for row in rows) / len(rows) for key in metric_keys})
        summary.append(summary_row)
        for row in rows:
            target = per_question_by_id.setdefault(row["question_id"], {"question_id": row["question_id"]})
            target.update({key: value for key, value in row.items() if key != "num_strict_relevant"})
    return list(per_question_by_id.values()), summary


def _build_document_qrels(
    qrels: dict[str, dict[str, float]],
    metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    doc_qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for question_id, relmap in qrels.items():
        for corpus_id, grade in relmap.items():
            metadata = metadata_by_id.get(corpus_id, {})
            source_file = metadata.get("source_file") or ""
            if not source_file:
                continue
            doc_qrels[question_id][source_file] = max(doc_qrels[question_id].get(source_file, 0.0), grade)
    return doc_qrels


def _build_document_results(
    results_by_question: dict[str, list[str]],
    metadata_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    doc_results: dict[str, list[str]] = {}
    for question_id, corpus_ids in results_by_question.items():
        seen: set[str] = set()
        ordered_docs: list[str] = []
        for corpus_id in corpus_ids:
            metadata = metadata_by_id.get(corpus_id, {})
            source_file = metadata.get("source_file") or ""
            if source_file and source_file not in seen:
                seen.add(source_file)
                ordered_docs.append(source_file)
        doc_results[question_id] = ordered_docs
    return doc_results


def compute_subset_metrics(
    *,
    question_rows: list[dict[str, str]],
    qrels: dict[str, dict[str, float]],
    results_by_question: dict[str, list[str]],
    k_values: list[int],
    strict_min_grade: float,
) -> list[dict[str, Any]]:
    subsets: dict[str, Any] = {
        "all": lambda row: True,
        "single_document": lambda row: not _bool_from_cell(row.get("is_cross_document")),
        "cross_document": lambda row: _bool_from_cell(row.get("is_cross_document")),
        "table_based": lambda row: _bool_from_cell(row.get("contains_table")) or "table" in (row.get("question_type") or ""),
        "numeric": lambda row: "numeric" in (row.get("question_type") or "") or "metric" in (row.get("question_type") or ""),
        "hard_retrieval": lambda row: "hard" in (row.get("retrieval_difficulty") or "").lower(),
    }
    rows: list[dict[str, Any]] = []
    rows_by_id = {row["question_id"]: row for row in question_rows}
    for subset_name, predicate in subsets.items():
        subset_ids = {qid for qid, row in rows_by_id.items() if predicate(row)}
        subset_qrels = {qid: relmap for qid, relmap in qrels.items() if qid in subset_ids}
        subset_results = {qid: results_by_question.get(qid, []) for qid in subset_ids}
        _, summary = compute_qrels_metrics(
            qrels=subset_qrels,
            results=subset_results,
            k_values=k_values,
            strict_min_grade=strict_min_grade,
        )
        for summary_row in summary:
            rows.append({"subset": subset_name, **summary_row})
    return rows


def build_error_analysis_rows(
    *,
    question_rows: list[dict[str, str]],
    results_by_question: dict[str, list[str]],
    corpus_state: CorpusState,
    qrels: dict[str, dict[str, float]],
    strict_min_grade: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_lookup = {row["question_id"]: row for row in question_rows}

    for question_id, retrieved in results_by_question.items():
        row = row_lookup.get(question_id, {})
        question = row.get("question", "")
        entities = _extract_query_entities(question=question, row=row, company_aliases=corpus_state.company_aliases)

        relevance_map = qrels.get(question_id, {})
        gold_ids = {cid for cid, grade in relevance_map.items() if grade >= strict_min_grade}
        gold_metadata = [corpus_state.metadata_by_id.get(cid, {}) for cid in gold_ids]
        gold_companies = {meta.get("company") for meta in gold_metadata if meta.get("company")}
        gold_years = {str(meta.get("report_year")) for meta in gold_metadata if meta.get("report_year")}
        gold_docs = {meta.get("source_file") for meta in gold_metadata if meta.get("source_file")}

        top_id = retrieved[0] if retrieved else ""
        top_meta = corpus_state.metadata_by_id.get(top_id, {}) if top_id else {}
        top_company = top_meta.get("company", "")
        top_year = str(top_meta.get("report_year", ""))
        top_doc = top_meta.get("source_file", "")

        retrieved_companies = {corpus_state.metadata_by_id.get(cid, {}).get("company") for cid in retrieved}
        retrieved_companies = {company for company in retrieved_companies if company}
        retrieved_tables = any(corpus_state.metadata_by_id.get(cid, {}).get("contains_table") for cid in retrieved)

        error_types: list[str] = []
        if gold_companies and top_company and top_company not in gold_companies:
            error_types.append("wrong_company")
        if (entities.years or gold_years) and top_year and top_year not in gold_years:
            error_types.append("wrong_year")
        if top_doc and gold_docs and top_doc in gold_docs and top_id not in gold_ids:
            error_types.append("right_document_wrong_section")
        if (_bool_from_cell(row.get("contains_table")) or any(meta.get("contains_table") for meta in gold_metadata)) and not retrieved_tables:
            error_types.append("table_heavy_failure")
        if _bool_from_cell(row.get("is_cross_document")):
            expected_companies = _parse_json_list(row.get("comparison_companies", ""))
            if expected_companies:
                missing = set(expected_companies) - retrieved_companies
                if missing:
                    error_types.append("multi_hop_comparison_miss")
            elif len(retrieved_companies) < 2:
                error_types.append("multi_hop_comparison_miss")
        if gold_ids:
            for rank, corpus_id in enumerate(retrieved, start=1):
                if corpus_id in gold_ids and rank > 5:
                    error_types.append("ranking_failure")
                    break
            if not any(cid in gold_ids for cid in retrieved):
                error_types.append("ranking_failure")

        rows.append(
            {
                "question_id": question_id,
                "question": question,
                "top_corpus_id": top_id,
                "top_source_file": top_doc,
                "top_company": top_company,
                "top_report_year": top_year,
                "gold_corpus_ids": ";".join(sorted(gold_ids)),
                "gold_companies": ";".join(sorted(gold_companies)),
                "gold_years": ";".join(sorted(gold_years)),
                "error_types": ";".join(sorted(set(error_types))),
            }
        )
    return rows


def _comparison_companies_from_row(row: dict[str, str]) -> list[str]:
    raw = (row.get("comparison_companies") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in raw.split(";") if part.strip()]
    return _mentioned_company_labels(row.get("question", ""))


def _chunk_to_result(chunk: ChunkRecord, score: float, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    company_label = _company_label_from_source(chunk.source_file, chunk.source_path)
    return {
        "score": score,
        "chunk_id": chunk.chunk_id,
        "company_label": company_label,
        "company": metadata.get("company") or company_label,
        "sector": metadata.get("sector", ""),
        "report_type": metadata.get("report_type", ""),
        "topic_tags": metadata.get("topic_tags", []),
        "source_file": chunk.source_file,
        "source_path": chunk.source_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "token_count": chunk.token_count,
        "text": chunk.text,
        "contextual_summary": chunk.contextual_summary,
        "esg_pillar": chunk.esg_pillar,
        "section_title": chunk.section_title,
        "report_year": chunk.report_year,
        "contains_table": chunk.contains_table,
        "contains_targets": chunk.contains_targets,
    }


def _decompose_comparison_queries(
    *,
    question: str,
    entities: QueryEntities,
    comparison_dimension: str | None = None,
) -> list[str]:
    if not entities.comparison or not entities.companies:
        return []
    years = entities.years or []
    topic_hint = " ".join(entities.topics) if entities.topics else ""
    dimension_hint = comparison_dimension or ""
    subqueries: list[str] = []
    if years:
        for company in entities.companies:
            for year in years:
                subqueries.append(
                    " ".join(part for part in [question, company, str(year), topic_hint, dimension_hint] if part)
                )
    else:
        for company in entities.companies:
            subqueries.append(
                " ".join(part for part in [question, company, topic_hint, dimension_hint] if part)
            )
    return subqueries


def _decompose_comparison_subqueries(
    *,
    question: str,
    entities: QueryEntities,
    comparison_dimension: str | None = None,
) -> list[dict[str, str]]:
    if not entities.comparison or not entities.companies:
        return []
    years = entities.years or []
    topic_hint = " ".join(entities.topics) if entities.topics else ""
    dimension_hint = comparison_dimension or ""
    subqueries: list[dict[str, str]] = []

    if years:
        for company in entities.companies:
            for year in years:
                subqueries.append(
                    {
                        "query": " ".join(
                            part for part in [question, company, str(year), topic_hint, dimension_hint] if part
                        ),
                        "company": company,
                        "year": str(year),
                    }
                )
    else:
        for company in entities.companies:
            subqueries.append(
                {
                    "query": " ".join(part for part in [question, company, topic_hint, dimension_hint] if part),
                    "company": company,
                    "year": "",
                }
            )
    return subqueries


def _build_query_variants(
    *,
    row: dict[str, str],
    entities: QueryEntities,
    include_dimensions: bool = True,
    include_decomposition: bool = True,
) -> list[str]:
    question = row["question"]
    variants: list[str] = [question]
    variants.append(_expand_query_with_synonyms(question, entities))

    if include_decomposition:
        comparison_dimension = row.get("comparison_dimension") or None
        variants.extend(
            _decompose_comparison_queries(
                question=question,
                entities=entities,
                comparison_dimension=comparison_dimension,
            )
        )

    if include_dimensions and (entities.comparison or _bool_from_cell(row.get("is_cross_document"))):
        variants.extend(f"{question}\n{dimension_query}" for _name, dimension_query in _ranking_dimensions(question))

    deduped: list[str] = []
    seen: set[str] = set()
    for text in variants:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def _prepare_query_vectors(
    *,
    question_rows: list[dict[str, str]],
    client: AlbertClient,
    embedding_model: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    query_texts: list[str] = []
    seen: set[str] = set()
    company_aliases = _build_company_aliases(question_rows)
    for row in question_rows:
        entities = _extract_query_entities(question=row["question"], row=row, company_aliases=company_aliases)
        candidates = _build_query_variants(row=row, entities=entities)
        for query_text in candidates:
            if query_text not in seen:
                seen.add(query_text)
                query_texts.append(query_text)
    vectors = embed_texts(client, query_texts, embedding_model=embedding_model, batch_size=batch_size)
    return {query_text: vectors[index] for index, query_text in enumerate(query_texts)}


def _ensure_query_vector(
    query_text: str,
    query_vectors: dict[str, np.ndarray],
    client: AlbertClient,
    embedding_model: str,
) -> np.ndarray:
    if query_text in query_vectors:
        return query_vectors[query_text]
    vector = embed_texts(client, [query_text], embedding_model=embedding_model, batch_size=1)[0]
    query_vectors[query_text] = vector
    return vector


def _filter_candidate_indices(
    *,
    chunks: list[ChunkRecord],
    corpus_state: CorpusState,
    entities: QueryEntities,
    forced_company: str | None = None,
    forced_year: str | None = None,
) -> list[int]:
    indices: list[int] = []
    for index, chunk in enumerate(chunks):
        metadata = corpus_state.metadata_by_id.get(chunk.chunk_id, {})
        company = str(metadata.get("company", "")).strip()
        report_year = str(metadata.get("report_year", "")).strip()
        sector = str(metadata.get("sector", "")).strip()
        report_type = str(metadata.get("report_type", "")).strip()

        if forced_company and company and company != forced_company:
            continue
        if forced_year and report_year and report_year != forced_year:
            continue
        if entities.companies and company and company not in entities.companies and forced_company is None:
            continue
        if entities.years and report_year and report_year not in entities.years and forced_year is None:
            continue
        if entities.sectors and sector and sector not in entities.sectors:
            continue
        if entities.report_types and report_type and report_type not in entities.report_types:
            continue
        indices.append(index)
    return indices


def _rank_hybrid_candidates(
    *,
    query_text: str,
    query_vector: np.ndarray,
    chunks: list[ChunkRecord],
    vectors: np.ndarray,
    bm25_index: BM25Index,
    corpus_state: CorpusState,
    entities: QueryEntities,
    candidate_indices: list[int],
    retrieval_mode: str,
    candidate_k: int,
    dense_weight: float,
    bm25_weight: float,
    metadata_weight: float,
) -> list[tuple[int, float]]:
    breadth = min(max(candidate_k, 1), len(candidate_indices))
    semantic_matches = _search_vectors_for_indices(query_vector, vectors, candidate_indices, breadth)
    semantic_scores = {index: score for index, score in semantic_matches}
    lexical_scores = _bm25_scores(query_text, bm25_index, candidate_indices)

    dense_norm = _normalize_scores(semantic_scores)
    lexical_norm = _normalize_scores(lexical_scores)
    combined: dict[int, float] = {}

    for index in candidate_indices:
        score = 0.0
        if retrieval_mode == "dense":
            score = dense_norm.get(index, 0.0)
        elif retrieval_mode == "lexical":
            score = lexical_norm.get(index, 0.0)
        elif retrieval_mode == "semantic_rerank":
            score = (0.8 * dense_norm.get(index, 0.0)) + (0.2 * lexical_norm.get(index, 0.0))
        else:
            score = (dense_weight * dense_norm.get(index, 0.0)) + (bm25_weight * lexical_norm.get(index, 0.0))

        metadata = corpus_state.metadata_by_id.get(chunks[index].chunk_id, {})
        score += metadata_weight * _score_metadata_boost(metadata, entities)
        combined[index] = score

    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    return ranked[:candidate_k]


def _expand_adjacent_chunks(
    *,
    ranked: list[dict[str, Any]],
    corpus_state: CorpusState,
    chunks: list[ChunkRecord],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return ranked
    selected: dict[str, dict[str, Any]] = {item["chunk_id"]: item for item in ranked}
    for item in ranked:
        neighbors = corpus_state.adjacent_map.get(item["chunk_id"], {})
        for key in ("prev", "next"):
            neighbor_id = neighbors.get(key) or ""
            if neighbor_id and neighbor_id not in selected:
                index = corpus_state.index_by_id.get(neighbor_id)
                if index is None:
                    continue
                neighbor_chunk = chunks[index]
                metadata = corpus_state.metadata_by_id.get(neighbor_id, {})
                boosted = _chunk_to_result(neighbor_chunk, float(item["score"]) * 0.85, metadata)
                boosted["retrieval_method"] = "adjacent"
                selected[neighbor_id] = boosted
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    return list(selected.values())


def _retrieve_for_eval_row_cached(
    *,
    row: dict[str, str],
    chunks: list[ChunkRecord],
    vectors: np.ndarray,
    query_vectors: dict[str, np.ndarray],
    top_k: int,
    retrieval_mode: str,
    search_breadth: int,
    comparison_mode: str,
    corpus_state: CorpusState,
    reranker: str,
    dense_weight: float,
    bm25_weight: float,
    metadata_weight: float,
    adjacent_window: int,
    decompose_comparisons: bool,
    expand_synonyms: bool,
    client: AlbertClient,
    embedding_model: str,
    text_model: str | None,
) -> list[dict[str, Any]]:
    question = row["question"]
    entities = _extract_query_entities(question=question, row=row, company_aliases=corpus_state.company_aliases)
    if comparison_mode == "never":
        entities.comparison = False
    elif comparison_mode == "always":
        entities.comparison = True

    subqueries: list[dict[str, str]] = []
    if decompose_comparisons and entities.comparison:
        subqueries = _decompose_comparison_subqueries(
            question=question,
            entities=entities,
            comparison_dimension=row.get("comparison_dimension") or None,
        )

    if not subqueries:
        subqueries = [{"query": question, "company": "", "year": ""}]

    selected: dict[str, dict[str, Any]] = {}
    for subquery in subqueries:
        query_text = subquery["query"]
        if expand_synonyms:
            query_text = _expand_query_with_synonyms(query_text, entities)
        forced_company = subquery.get("company") or None
        forced_year = subquery.get("year") or None
        candidate_indices = _filter_candidate_indices(
            chunks=chunks,
            corpus_state=corpus_state,
            entities=entities,
            forced_company=forced_company,
            forced_year=forced_year,
        )
        if not candidate_indices:
            candidate_indices = list(range(len(chunks)))

        query_vector = _ensure_query_vector(query_text, query_vectors, client, embedding_model)
        ranked = _rank_hybrid_candidates(
            query_text=query_text,
            query_vector=query_vector,
            chunks=chunks,
            vectors=vectors,
            bm25_index=corpus_state.bm25_index,
            corpus_state=corpus_state,
            entities=entities,
            candidate_indices=candidate_indices,
            retrieval_mode=retrieval_mode,
            candidate_k=search_breadth,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            metadata_weight=metadata_weight,
        )
        for chunk_index, score in ranked:
            chunk = chunks[chunk_index]
            metadata = corpus_state.metadata_by_id.get(chunk.chunk_id, {})
            existing = selected.get(chunk.chunk_id)
            if existing:
                existing["score"] = max(float(existing["score"]), float(score))
            else:
                selected[chunk.chunk_id] = _chunk_to_result(chunk, score, metadata)

    candidates = sorted(selected.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
    candidates = candidates[: max(search_breadth, top_k)]

    if reranker and reranker != "none":
        candidates = rerank_candidates(
            question=question,
            candidates=candidates,
            top_k=top_k,
            reranker=reranker,
            client=client,
            embedding_model=embedding_model,
            text_model=text_model,
        )
        for item in candidates:
            item["retrieval_method"] = f"{retrieval_mode}+rerank:{reranker}"
    else:
        candidates = candidates[:top_k]
        for item in candidates:
            item["retrieval_method"] = retrieval_mode

    if adjacent_window > 0:
        expanded = _expand_adjacent_chunks(
            ranked=candidates,
            corpus_state=corpus_state,
            chunks=chunks,
            limit=min(len(chunks), top_k + adjacent_window * 2),
        )
        candidates = sorted(expanded, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:top_k]

    return candidates


def _retrieve_for_eval_row(
    *,
    row: dict[str, str],
    index_dir: Path,
    top_k: int,
    embedding_model: str | None,
    retrieval_mode: str,
    search_breadth: int,
    base_url: str,
    comparison_mode: str,
) -> tuple[list[dict[str, Any]], str | None]:
    client = AlbertClient(api_key=require_api_key(), base_url=base_url)
    resolved_model = embedding_model or client.get_embedding_model(preferred="bge-m3")
    chunks = _load_chunks(index_dir)
    vectors = np.load(index_dir / "vectors.npy")
    corpus_state = _load_corpus_state(DEFAULT_RAGAS_V2_CORPUS, chunks)
    query_vectors: dict[str, np.ndarray] = {}
    retrieved = _retrieve_for_eval_row_cached(
        row=row,
        chunks=chunks,
        vectors=vectors,
        query_vectors=query_vectors,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        search_breadth=search_breadth,
        comparison_mode=comparison_mode,
        corpus_state=corpus_state,
        reranker="none",
        dense_weight=0.6,
        bm25_weight=0.4,
        metadata_weight=0.2,
        adjacent_window=0,
        decompose_comparisons=True,
        expand_synonyms=True,
        client=client,
        embedding_model=resolved_model,
        text_model=None,
    )
    return retrieved, resolved_model


def evaluate_ragas_v2_retrieval(
    *,
    index_dir: Path = DEFAULT_RAGAS_V2_INDEX_DIR,
    dataset_path: Path = DEFAULT_RAGAS_V2_DATASET,
    qrels_path: Path = DEFAULT_RAGAS_V2_QRELS,
    corpus_path: Path = DEFAULT_RAGAS_V2_CORPUS,
    top_k: int = 10,
    retrieval_mode: str = "semantic_rerank",
    search_breadth: int = 100,
    embedding_model: str | None = None,
    strict_min_grade: float = 2.0,
    k_values: list[int] | None = None,
    comparison_mode: str = "auto",
    base_url: str = DEFAULT_BASE_URL,
    batch_size: int = 32,
    cached_queries: bool = True,
    preloaded_chunks: list[ChunkRecord] | None = None,
    preloaded_vectors: np.ndarray | None = None,
    precomputed_query_vectors: dict[str, np.ndarray] | None = None,
    resolved_embedding_model: str | None = None,
    reranker: str = "none",
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
    metadata_weight: float = 0.2,
    adjacent_window: int = 0,
    decompose_comparisons: bool = True,
    expand_synonyms: bool = True,
    text_model: str | None = None,
) -> dict[str, Any]:
    _normalize_retrieval_mode(retrieval_mode)
    k_values = k_values or [1, 3, 5, 10, 20, 50]
    random.seed(DEFAULT_RANDOM_SEED)
    np.random.seed(DEFAULT_RANDOM_SEED)
    question_rows = _read_csv_rows(dataset_path)
    qrels = _load_qrels(qrels_path)
    results_by_question: dict[str, list[str]] = {}
    retrieval_rows: list[dict[str, Any]] = []
    resolved_embedding_model = resolved_embedding_model or embedding_model
    chunks: list[ChunkRecord] | None = preloaded_chunks
    vectors: np.ndarray | None = preloaded_vectors
    query_vectors: dict[str, np.ndarray] | None = precomputed_query_vectors

    corpus_state: CorpusState | None = None
    if chunks is None:
        chunks = _load_chunks(index_dir)
    if corpus_state is None and chunks is not None:
        corpus_state = _load_corpus_state(corpus_path, chunks)

    if cached_queries:
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        resolved_embedding_model = resolved_embedding_model or manifest.get("embedding_model")
        if not resolved_embedding_model:
            client = AlbertClient(api_key=require_api_key(), base_url=base_url)
            resolved_embedding_model = client.get_embedding_model(preferred="bge-m3")
        if vectors is None:
            vectors = np.load(index_dir / "vectors.npy")
        if query_vectors is None:
            client = AlbertClient(api_key=require_api_key(), base_url=base_url)
            query_vectors = _prepare_query_vectors(
                question_rows=question_rows,
                client=client,
                embedding_model=resolved_embedding_model,
                batch_size=batch_size,
            )

    for row in question_rows:
        question_id = row["question_id"]
        if cached_queries and chunks is not None and vectors is not None and query_vectors is not None and corpus_state is not None:
            client = AlbertClient(api_key=require_api_key(), base_url=base_url)
            retrieved = _retrieve_for_eval_row_cached(
                row=row,
                chunks=chunks,
                vectors=vectors,
                query_vectors=query_vectors,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                search_breadth=search_breadth,
                comparison_mode=comparison_mode,
                corpus_state=corpus_state,
                reranker=reranker,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
                metadata_weight=metadata_weight,
                adjacent_window=adjacent_window,
                decompose_comparisons=decompose_comparisons,
                expand_synonyms=expand_synonyms,
                client=client,
                embedding_model=resolved_embedding_model,
                text_model=text_model,
            )
        else:
            retrieved, resolved_embedding_model = _retrieve_for_eval_row(
                row=row,
                index_dir=index_dir,
                top_k=top_k,
                embedding_model=resolved_embedding_model,
                retrieval_mode=retrieval_mode,
                search_breadth=search_breadth,
                base_url=base_url,
                comparison_mode=comparison_mode,
            )
        corpus_ids = [str(chunk["chunk_id"]) for chunk in retrieved]
        results_by_question[question_id] = corpus_ids
        for rank, chunk in enumerate(retrieved, start=1):
            retrieval_rows.append(
                {
                    "question_id": question_id,
                    "corpus_id": str(chunk["chunk_id"]),
                    "rank": rank,
                    "score": float(chunk.get("score", 0.0)),
                    "retrieval_method": chunk.get("retrieval_method", retrieval_mode),
                    "source_file": chunk.get("source_file", ""),
                    "page_start": chunk.get("page_start", ""),
                    "page_end": chunk.get("page_end", ""),
                }
            )

    per_question, summary = compute_qrels_metrics(
        qrels=qrels,
        results=results_by_question,
        k_values=k_values,
        strict_min_grade=strict_min_grade,
    )
    subset_summary = []
    doc_per_question: list[dict[str, Any]] = []
    doc_summary: list[dict[str, Any]] = []
    error_analysis: list[dict[str, Any]] = []

    if corpus_state is not None:
        subset_summary = compute_subset_metrics(
            question_rows=question_rows,
            qrels=qrels,
            results_by_question=results_by_question,
            k_values=k_values,
            strict_min_grade=strict_min_grade,
        )
        doc_qrels = _build_document_qrels(qrels, corpus_state.metadata_by_id)
        doc_results = _build_document_results(results_by_question, corpus_state.metadata_by_id)
        doc_per_question, doc_summary = compute_qrels_metrics(
            qrels=doc_qrels,
            results=doc_results,
            k_values=k_values,
            strict_min_grade=strict_min_grade,
        )
        error_analysis = build_error_analysis_rows(
            question_rows=question_rows,
            results_by_question=results_by_question,
            corpus_state=corpus_state,
            qrels=qrels,
            strict_min_grade=strict_min_grade,
        )
    return {
        "dataset": str(dataset_path),
        "qrels": str(qrels_path),
        "index_dir": str(index_dir),
        "question_count": len(question_rows),
        "retrieval_mode": retrieval_mode,
        "search_breadth": search_breadth,
        "top_k": top_k,
        "strict_min_grade": strict_min_grade,
        "comparison_mode": comparison_mode,
        "reranker": reranker,
        "dense_weight": dense_weight,
        "bm25_weight": bm25_weight,
        "metadata_weight": metadata_weight,
        "adjacent_window": adjacent_window,
        "decompose_comparisons": decompose_comparisons,
        "expand_synonyms": expand_synonyms,
        "embedding_model": resolved_embedding_model,
        "retrieval_rows": retrieval_rows,
        "per_question": per_question,
        "summary": summary,
        "subset_summary": subset_summary,
        "doc_per_question": doc_per_question,
        "doc_summary": doc_summary,
        "error_analysis": error_analysis,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ragas_v2_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {key: value for key, value in payload.items() if key != "retrieval_rows"}
    (output_dir / "ragas_v2_retrieval_eval.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(output_dir / "retrieval_results.csv", payload["retrieval_rows"])
    _write_csv(output_dir / "metrics_per_question.csv", payload["per_question"])
    _write_csv(output_dir / "metrics_summary.csv", payload["summary"])
    _write_csv(output_dir / "subset_metrics_summary.csv", payload.get("subset_summary", []))
    _write_csv(output_dir / "doc_metrics_per_question.csv", payload.get("doc_per_question", []))
    _write_csv(output_dir / "doc_metrics_summary.csv", payload.get("doc_summary", []))
    _write_csv(output_dir / "error_analysis.csv", payload.get("error_analysis", []))

    (output_dir / "metrics_summary.json").write_text(
        json.dumps(payload.get("summary", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "subset_metrics_summary.json").write_text(
        json.dumps(payload.get("subset_summary", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "doc_metrics_summary.json").write_text(
        json.dumps(payload.get("doc_summary", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_ragas_v2_build_index(args: Any) -> int:
    manifest = build_ragas_v2_corpus_index(
        corpus_path=args.corpus,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        base_url=args.base_url,
        force=args.force,
    )
    print(
        f"Built RAGAS v2 corpus index with {manifest['chunk_count']} chunks "
        f"using {manifest['embedding_model']} at {args.index_dir}."
    )
    return 0


def run_ragas_v2_retrieval_eval(args: Any) -> int:
    if not (args.index_dir / "manifest.json").exists():
        build_ragas_v2_corpus_index(
            corpus_path=args.corpus,
            index_dir=args.index_dir,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            base_url=args.base_url,
        )
    payload = evaluate_ragas_v2_retrieval(
        index_dir=args.index_dir,
        dataset_path=args.dataset,
        qrels_path=args.qrels,
        corpus_path=args.corpus,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        search_breadth=args.candidate_k,
        embedding_model=args.embedding_model,
        strict_min_grade=args.strict_min_grade,
        k_values=args.k,
        comparison_mode=args.comparison_mode,
        base_url=args.base_url,
        batch_size=args.batch_size,
        reranker=args.reranker,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        metadata_weight=args.metadata_weight,
        adjacent_window=args.adjacent_window,
        decompose_comparisons=not args.no_decompose,
        expand_synonyms=not args.no_synonyms,
        text_model=getattr(args, "text_model", None),
    )
    write_ragas_v2_outputs(payload, args.output_dir)
    print(
        f"RAGAS v2 retrieval eval complete: {payload['question_count']} questions | "
        f"mode={payload['retrieval_mode']} breadth={payload['search_breadth']} comparison={payload['comparison_mode']}"
    )
    for row in payload["summary"]:
        k = row["k"]
        print(
            f"@{k}: hit={row[f'hit_rate@{k}']:.1%} recall={row[f'recall@{k}']:.1%} "
            f"mrr={row[f'mrr@{k}']:.3f} ndcg={row[f'ndcg@{k}']:.3f}"
        )
    print(f"Saved outputs to {args.output_dir}")
    return 0


def run_ragas_v2_retrieval_grid(args: Any) -> int:
    if not (args.index_dir / "manifest.json").exists():
        build_ragas_v2_corpus_index(
            corpus_path=args.corpus,
            index_dir=args.index_dir,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            base_url=args.base_url,
        )
    question_rows = _read_csv_rows(args.dataset)
    manifest = json.loads((args.index_dir / "manifest.json").read_text(encoding="utf-8"))
    resolved_embedding_model = args.embedding_model or manifest.get("embedding_model")
    if not resolved_embedding_model:
        client = AlbertClient(api_key=require_api_key(), base_url=args.base_url)
        resolved_embedding_model = client.get_embedding_model(preferred="bge-m3")
    client = AlbertClient(api_key=require_api_key(), base_url=args.base_url)
    chunks = _load_chunks(args.index_dir)
    vectors = np.load(args.index_dir / "vectors.npy")
    query_vectors = _prepare_query_vectors(
        question_rows=question_rows,
        client=client,
        embedding_model=resolved_embedding_model,
        batch_size=args.batch_size,
    )

    candidates: list[dict[str, Any]] = []
    for retrieval_mode in args.retrieval_modes:
        for candidate_k in args.candidate_ks:
            for comparison_mode in args.comparison_modes:
                payload = evaluate_ragas_v2_retrieval(
                    index_dir=args.index_dir,
                    dataset_path=args.dataset,
                    qrels_path=args.qrels,
                    corpus_path=args.corpus,
                    top_k=args.top_k,
                    retrieval_mode=retrieval_mode,
                    search_breadth=candidate_k,
                    embedding_model=args.embedding_model,
                    strict_min_grade=args.strict_min_grade,
                    k_values=args.k,
                    comparison_mode=comparison_mode,
                    base_url=args.base_url,
                    batch_size=args.batch_size,
                    preloaded_chunks=chunks,
                    preloaded_vectors=vectors,
                    precomputed_query_vectors=query_vectors,
                    resolved_embedding_model=resolved_embedding_model,
                )
                summary_at_10 = next((row for row in payload["summary"] if row["k"] == 10), payload["summary"][-1])
                result = {
                    "retrieval_mode": retrieval_mode,
                    "candidate_k": candidate_k,
                    "comparison_mode": comparison_mode,
                    "summary": payload["summary"],
                    "score": summary_at_10.get("ndcg@10", 0.0) + summary_at_10.get("recall@10", 0.0),
                }
                candidates.append(result)
                print(
                    f"{retrieval_mode} breadth={candidate_k} comparison={comparison_mode}: "
                    f"recall@10={summary_at_10.get('recall@10', 0.0):.1%} "
                    f"ndcg@10={summary_at_10.get('ndcg@10', 0.0):.3f}",
                    flush=True,
                )
    best = max(candidates, key=lambda item: item["score"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ragas_v2_retrieval_grid.json").write_text(
        json.dumps({"candidates": candidates, "best": best}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nBest retrieval configuration:")
    print(
        f"  mode={best['retrieval_mode']} breadth={best['candidate_k']} "
        f"comparison={best['comparison_mode']}"
    )
    print(f"Saved grid results to {args.output_dir / 'ragas_v2_retrieval_grid.json'}")
    return 0
