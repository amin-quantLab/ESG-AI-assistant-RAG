from pathlib import Path
from urllib.parse import urlencode

from app.rag import DEFAULT_SAMPLE_PDF, parse_pdf_path_lines
from app.web import _render_markdown, create_app


def test_parse_pdf_path_lines_uses_sample_when_blank():
    paths = parse_pdf_path_lines("")
    assert DEFAULT_SAMPLE_PDF.resolve() in paths
    assert any("esg_scraper/data/pdfs" in str(path) for path in paths)


def test_web_homepage_renders_successfully():
    app = create_app(index_dir=Path("/tmp/nonexistent-index"))
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(
        app(
            {
                "PATH_INFO": "/",
                "REQUEST_METHOD": "GET",
                "wsgi.input": __import__("io").BytesIO(b""),
                "CONTENT_LENGTH": "0",
            },
            start_response,
        )
    ).decode("utf-8")

    assert captured["status"] == "200 OK"
    assert "ESG RAG Studio" in body
    assert "Build Index" in body


def test_render_markdown_formats_sections_and_tables():
    html = _render_markdown(
        """**Key takeaway**
- Danone has validated 2030 climate targets.

**Detailed answer**
| Metric | Target |
|---|---|
| Scope 1 & 2 | -46.3% |
"""
    )

    assert "<h3>Key takeaway</h3>" in html
    assert "<li>Danone has validated 2030 climate targets.</li>" in html
    assert "<table>" in html
    assert "<th>Metric</th>" in html
    assert "<td>Scope 1 &amp; 2</td>" in html


def test_web_ask_keeps_question_and_renders_markdown(monkeypatch):
    def fake_retrieve_chunks(**kwargs):
        return (
            [
                {
                    "chunk_id": "chunk-0001",
                    "source_file": "report.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.8,
                    "text": "Target evidence.",
                }
            ],
            "embedding-model",
        )

    def fake_answer_question(**kwargs):
        return (
            "**Key takeaway**\n- The target is disclosed.\n\n**Uncertainty**\n- Scope is partial.",
            "text-model",
        )

    monkeypatch.setattr("app.web.retrieve_chunks", fake_retrieve_chunks)
    monkeypatch.setattr("app.web.answer_question", fake_answer_question)

    app = create_app(index_dir=Path("/tmp/nonexistent-index"))
    captured = {}
    body_bytes = urlencode({"question": "What are Danone's climate targets?"}).encode("utf-8")

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(
        app(
            {
                "PATH_INFO": "/ask",
                "REQUEST_METHOD": "POST",
                "wsgi.input": __import__("io").BytesIO(body_bytes),
                "CONTENT_LENGTH": str(len(body_bytes)),
            },
            start_response,
        )
    ).decode("utf-8")

    assert captured["status"] == "200 OK"
    assert "What are Danone&#x27;s climate targets?" in body
    assert "<h3>Key takeaway</h3>" in body
    assert "<li>The target is disclosed.</li>" in body


def test_web_ask_uses_company_ranking_retrieval_for_ranking_questions(monkeypatch):
    calls = {}

    def fake_retrieve_company_ranking_chunks(**kwargs):
        calls["ranking"] = kwargs
        return (
            [
                {
                    "chunk_id": "chunk-0001",
                    "company_label": "Danone",
                    "source_file": "danone.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.8,
                    "text": "Danone evidence.",
                },
                {
                    "chunk_id": "chunk-0002",
                    "company_label": "Enel",
                    "source_file": "enel.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.7,
                    "text": "Enel evidence.",
                },
            ],
            "embedding-model",
            {"mode": "company_ranking"},
        )

    def fake_generate_company_ranking_answer(**kwargs):
        calls["ranking_answer"] = kwargs
        return "**Overall ESG Performance**\n\n| Rank | Company |\n|---|---|\n| 1 | Danone |"

    monkeypatch.setattr("app.web.retrieve_company_ranking_chunks", fake_retrieve_company_ranking_chunks)
    monkeypatch.setattr("app.web.generate_company_ranking_answer", fake_generate_company_ranking_answer)

    app = create_app(index_dir=Path("/tmp/nonexistent-index"))
    body_bytes = urlencode(
        {"question": "Rank companies by ESG performance in three different ways: achievements, targets, and improvements."}
    ).encode("utf-8")

    def start_response(status, headers):
        pass

    body = b"".join(
        app(
            {
                "PATH_INFO": "/ask",
                "REQUEST_METHOD": "POST",
                "wsgi.input": __import__("io").BytesIO(body_bytes),
                "CONTENT_LENGTH": str(len(body_bytes)),
            },
            start_response,
        )
    ).decode("utf-8")

    assert calls["ranking"]["per_company_k"] == 1
    assert calls["ranking"]["max_companies"] == 20
    assert calls["ranking_answer"]["question"].startswith("Rank companies")
    assert "deterministic-esg-ranker" in body
    assert "Overall ESG Performance" in body


def test_web_ask_uses_named_company_comparison_path(monkeypatch):
    calls = {}

    def fake_retrieve_company_ranking_chunks(**kwargs):
        calls["retrieval"] = kwargs
        return (
            [
                {
                    "chunk_id": "chunk-0001",
                    "company_label": "Engie",
                    "source_file": "engie.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.8,
                    "text": "Engie target evidence.",
                },
                {
                    "chunk_id": "chunk-0002",
                    "company_label": "LVMH",
                    "source_file": "lvmh.pdf",
                    "page_start": 2,
                    "page_end": 2,
                    "score": 0.7,
                    "text": "LVMH target evidence.",
                },
            ],
            "embedding-model",
            {"mode": "company_ranking"},
        )

    def fake_generate_company_comparison_answer(**kwargs):
        calls["comparison_answer"] = kwargs
        return "**Company Comparison**\n\n| Company | Evidence |\n|---|---|\n| Engie | Evidence |"

    monkeypatch.setattr("app.web.retrieve_company_ranking_chunks", fake_retrieve_company_ranking_chunks)
    monkeypatch.setattr("app.web.generate_company_comparison_answer", fake_generate_company_comparison_answer)

    app = create_app(index_dir=Path("/tmp/nonexistent-index"))
    body_bytes = urlencode({"question": "Compare Engie and LVMH in their ESG ambitions"}).encode("utf-8")

    def start_response(status, headers):
        pass

    body = b"".join(
        app(
            {
                "PATH_INFO": "/ask",
                "REQUEST_METHOD": "POST",
                "wsgi.input": __import__("io").BytesIO(body_bytes),
                "CONTENT_LENGTH": str(len(body_bytes)),
            },
            start_response,
        )
    ).decode("utf-8")

    assert calls["retrieval"]["target_companies"] == ["Engie", "LVMH"]
    assert calls["retrieval"]["per_company_k"] == 3
    assert calls["comparison_answer"]["companies"] == ["Engie", "LVMH"]
    assert "deterministic-esg-comparator" in body
    assert "Company Comparison" in body


def test_web_ask_uses_named_company_comparison_path_for_acronym_plural_prompt(monkeypatch):
    calls = {}

    def fake_retrieve_company_ranking_chunks(**kwargs):
        calls["retrieval"] = kwargs
        return (
            [
                {
                    "chunk_id": "chunk-0001",
                    "company_label": "BNP Paribas",
                    "source_file": "bnp.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.8,
                    "text": "BNP evidence.",
                },
                {
                    "chunk_id": "chunk-0002",
                    "company_label": "Enel",
                    "source_file": "enel.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.7,
                    "text": "Enel evidence.",
                },
            ],
            "embedding-model",
            {"mode": "company_ranking"},
        )

    def fake_generate_company_comparison_answer(**kwargs):
        calls["comparison_answer"] = kwargs
        return "**Company Comparison**\n\n| Company | Evidence |\n|---|---|\n| BNP Paribas | Evidence |"

    monkeypatch.setattr("app.web.retrieve_company_ranking_chunks", fake_retrieve_company_ranking_chunks)
    monkeypatch.setattr("app.web.generate_company_comparison_answer", fake_generate_company_comparison_answer)

    app = create_app(index_dir=Path("/tmp/nonexistent-index"))
    body_bytes = urlencode(
        {"question": "Compare the evolution of BNPs environmental commitments to those of ENEL"}
    ).encode("utf-8")

    def start_response(status, headers):
        pass

    body = b"".join(
        app(
            {
                "PATH_INFO": "/ask",
                "REQUEST_METHOD": "POST",
                "wsgi.input": __import__("io").BytesIO(body_bytes),
                "CONTENT_LENGTH": str(len(body_bytes)),
            },
            start_response,
        )
    ).decode("utf-8")

    assert calls["retrieval"]["target_companies"] == ["BNP Paribas", "Enel"]
    assert calls["comparison_answer"]["companies"] == ["BNP Paribas", "Enel"]
    assert "deterministic-esg-comparator" in body


def test_web_ask_uses_named_company_comparison_path_for_parallel_list_prompt(monkeypatch):
    calls = {}

    def fake_retrieve_company_ranking_chunks(**kwargs):
        calls["retrieval"] = kwargs
        return (
            [
                {
                    "chunk_id": "chunk-0001",
                    "company_label": "BNP Paribas",
                    "source_file": "bnp.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.8,
                    "text": "BNP environmental goals evidence.",
                },
                {
                    "chunk_id": "chunk-0002",
                    "company_label": "Enel",
                    "source_file": "enel.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "score": 0.7,
                    "text": "Enel environmental goals evidence.",
                },
            ],
            "embedding-model",
            {"mode": "company_ranking"},
        )

    def fake_generate_company_comparison_answer(**kwargs):
        calls["comparison_answer"] = kwargs
        return "**Company Comparison**\n\n**Goals and Commitments**\n| Company | Evidence |\n|---|---|\n| BNP Paribas | Evidence |"

    monkeypatch.setattr("app.web.retrieve_company_ranking_chunks", fake_retrieve_company_ranking_chunks)
    monkeypatch.setattr("app.web.generate_company_comparison_answer", fake_generate_company_comparison_answer)

    app = create_app(index_dir=Path("/tmp/nonexistent-index"))
    body_bytes = urlencode(
        {"question": "List BNP environmental goals. List ENEL environmental goals"}
    ).encode("utf-8")

    def start_response(status, headers):
        pass

    body = b"".join(
        app(
            {
                "PATH_INFO": "/ask",
                "REQUEST_METHOD": "POST",
                "wsgi.input": __import__("io").BytesIO(body_bytes),
                "CONTENT_LENGTH": str(len(body_bytes)),
            },
            start_response,
        )
    ).decode("utf-8")

    assert calls["retrieval"]["target_companies"] == ["BNP Paribas", "Enel"]
    assert calls["comparison_answer"]["companies"] == ["BNP Paribas", "Enel"]
    assert "deterministic-esg-comparator" in body
