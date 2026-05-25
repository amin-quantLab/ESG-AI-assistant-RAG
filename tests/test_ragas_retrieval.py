from app.ragas_retrieval import (
    _bm25_scores,
    _build_bm25_index,
    _expand_query_with_synonyms,
    _extract_query_entities,
    _parse_json_list,
    QueryEntities,
)


def test_parse_json_list_handles_json_and_delimited():
    assert _parse_json_list('["a", "b"]') == ["a", "b"]
    assert _parse_json_list("a; b") == ["a", "b"]


def test_expand_query_with_synonyms_adds_terms():
    entities = QueryEntities(topics=["emissions"], metrics=["ghg"])
    expanded = _expand_query_with_synonyms("What are the GHG targets?", entities)
    assert "greenhouse gas" in expanded
    assert "co2e" in expanded


def test_extract_query_entities_uses_row_company_and_year():
    row = {
        "question": "What does ENGIE disclose in 2023?",
        "company": "ENGIE",
        "sector": "utilities_energy",
        "topic_tags": "[\"emissions\"]",
        "is_cross_document": "false",
    }
    entities = _extract_query_entities(question=row["question"], row=row, company_aliases={"engie": "ENGIE"})
    assert "ENGIE" in entities.companies
    assert "2023" in entities.years
    assert "emissions" in entities.topics


def test_bm25_scores_returns_match():
    rows = [
        {"text": "greenhouse gas emissions scope 1 scope 2", "company": "ENGIE"},
        {"text": "water withdrawal and discharge", "company": "Danone"},
    ]
    index = _build_bm25_index(rows)
    scores = _bm25_scores("ghg emissions", index, candidate_indices=[0, 1])
    assert scores.get(0, 0.0) > 0.0
    assert scores.get(1, 0.0) == 0.0
