"""Pre-Lap-2 retrieval weakness map — find -> cause -> fix-test -> prescription, all evaluable companies.

For each (company, row): three judge calls.
  1. DENSE   = raw cosine top-8, repaired-relevance judge.
  2. COVERAGE = embed gold span, nearest company-restricted chunk (noise-masked), judge it 1-on-1.
  3. HYBRID  = retrieve_chunks(architecture="hybrid"), repaired-relevance judge.

PIPELINE FLOW
  Stage 0 capture_provenance()
  Stage 1 load_inputs()         + alias map + EVAL_COMPANIES derivation
  Stage 2 build_noise_mask()
  Stage 3 measure_rows()        per (company,row) — 3 judge calls each
  Stage 4 aggregate_classify()  per-company weakness class
  Stage 5 report()              md + csv (incl. worst-3 forensics + prescriptions)
  Stage 6 write_manifest()

NO edits to app/*. NO git writes. NO pip/env changes. NO deletions. NO re-embedding.
Secrets via setdefault from ..\\esg_scraper\\.env and never printed.

Run from eval/:  PYTHONPATH=.. python retrieval_weakness_map.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =============================== CONFIG ===============================
CONFIG: dict[str, Any] = {
    "INDEX_DIR":           "../outputs/rag_index",
    "DATASET":             "../sample_data/ragas_esg_eval_dataset.csv",
    "ALIAS_MAP":           {"ENGIE": "Engie", "Schneider": "Schneider Electric"},
    "EXCLUDE_COMPANIES":   {"Hermès", "Iberdrola", "Unknown"},
    "ROWS_PER_COMPANY":    5,                       # capped: min(available, 5)
    "TOP_K":               8,
    "CANDIDATE_K":         100,
    "JUDGE":               "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "PASSAGE_CHAR_CAP":    800,
    # Noise-mask thresholds (same as iter1)
    "MIN_TOKENS":          5,
    "MIN_WORDLIKE":        3,
    "MAX_NONALNUM_RATIO":  0.5,
    # Classification thresholds
    "GOOD_THRESHOLD":      0.60,
    "HYBRID_HELP_DELTA":   0.10,
    "COVERAGE_GAP_YES_RATE": 0.40,
    "TABLEFACT_DOMINANT":  0.60,
}

BUDGET = {"chat_calls": 0, "rerank_calls": 0, "embedding_calls": 0,
          "embedding_strings": 0, "judge_calls": 0}

READ_ONLY_GIT_COMMANDS_USED: list[str] = []
WORDLIKE_RE = re.compile(r"[A-Za-z]{3,}")


# =============================== ENV ===============================
def load_env() -> None:
    """Load .env via setdefault from the first matching candidate path. Works whether
    the script is run from the repo root (Git_clone/) or from eval/. Never prints secrets."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / ".." / "esg_scraper" / ".env",   # eval/ -> Script/esg_scraper/.env
        here / ".." / "esg_scraper" / ".env",          # legacy: repo root cwd
        Path(".env"),
    ]
    for env_path in candidates:
        if env_path.is_file():
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


load_env()
if not os.environ.get("ALBERT_API_KEY"):
    sys.stderr.write("ALBERT_API_KEY missing. Aborting.\n")
    sys.exit(1)

from app.rag import AlbertClient, DEFAULT_BASE_URL, retrieve_chunks      # noqa: E402
from app.rag_eval import _parse_expected_contexts                         # noqa: E402

ALBERT_BASE = os.environ.get("ALBERT_BASE_URL") or DEFAULT_BASE_URL

_orig_chat = AlbertClient.chat_completion
_orig_embed = AlbertClient.create_embeddings


def _chat_wrapper(self, model, messages, *args, **kwargs):
    BUDGET["chat_calls"] += 1
    return _orig_chat(self, model, messages, *args, **kwargs)


def _embed_wrapper(self, model, inputs):
    BUDGET["embedding_calls"] += 1
    BUDGET["embedding_strings"] += len(inputs)
    return _orig_embed(self, model, inputs)


AlbertClient.chat_completion = _chat_wrapper
AlbertClient.create_embeddings = _embed_wrapper


# =============================== HELPERS ===============================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_record(args: list[str]) -> str:
    READ_ONLY_GIT_COMMANDS_USED.append("git " + " ".join(args))
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=30, check=False)
        return r.stdout.strip()
    except Exception as exc:
        return f"<git error: {type(exc).__name__}: {exc}>"


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {"numpy": getattr(np, "__version__", "?")}
    for name in ("rapidfuzz", "thefuzz", "requests", "faiss"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "<not installed>"
    return out


def truncate(s: str, n: int) -> str:
    if not s:
        return ""
    return " ".join(s.split())[:n]


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(raw[start: end + 1])


# =============================== STAGE 0 ===============================
def capture_provenance() -> dict[str, Any]:
    head = git_record(["rev-parse", "HEAD"])
    porcelain = git_record(["status", "--porcelain"])
    diff_stat = git_record(["diff", "--stat", "--", "app/*.py"])
    app_hashes: dict[str, str] = {}
    for fname in ("app/rag.py", "app/ragas_eval.py", "app/cli.py"):
        p = Path(fname)
        app_hashes[fname] = sha256_file(p) if p.is_file() else "<missing>"
    return {"git_head": head, "git_status_porcelain": porcelain,
            "git_diff_stat_app": diff_stat, "code_sha256": app_hashes}


# =============================== STAGE 1 ===============================
def load_inputs() -> dict[str, Any]:
    idx_dir = Path(CONFIG["INDEX_DIR"])
    manifest_path = idx_dir / "manifest.json"
    chunks_path = idx_dir / "chunks.json"
    vectors_path = idx_dir / "vectors.npy"
    dataset_path = Path(CONFIG["DATASET"])

    print("[Stage 1] loading manifest, chunks, vectors, dataset")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    vectors_mmap = np.load(vectors_path, mmap_mode="r")
    if not (len(chunks) == vectors_mmap.shape[0] == 145197):
        raise SystemExit("STOP: length mismatch")
    if vectors_mmap.dtype != np.float32 or vectors_mmap.shape[1] != 1024:
        raise SystemExit("STOP: vectors dtype/shape mismatch")
    print(f"[Stage 1] alignment OK: 145197 chunks, vectors (145197, 1024) float32")

    print("[Stage 1] materializing full vector matrix into RAM (~600 MB)...")
    full_vecs = np.asarray(vectors_mmap, dtype=np.float32)
    fnorms = np.linalg.norm(full_vecs, axis=1)
    if not np.allclose(fnorms, 1.0, atol=1e-3):
        full_vecs /= fnorms[:, None]

    # Build company -> indices map
    company_to_indices: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        company_to_indices.setdefault(c.get("company", ""), []).append(i)

    # Load dataset and derive EVAL_COMPANIES
    with dataset_path.open("r", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))
    from collections import Counter
    ds_counter = Counter(r.get("company", "") for r in all_rows)
    alias = CONFIG["ALIAS_MAP"]
    exclude = CONFIG["EXCLUDE_COMPANIES"]
    eval_companies: list[dict[str, Any]] = []
    for ds_co in sorted(ds_counter):
        if ds_co in exclude:
            continue
        idx_co = alias.get(ds_co, ds_co)
        if company_to_indices.get(idx_co):
            eval_companies.append({
                "dataset_name": ds_co,
                "index_name": idx_co,
                "n_dataset_rows": ds_counter[ds_co],
                "n_corpus_chunks": len(company_to_indices[idx_co]),
            })
    print(f"[Stage 1] EVAL_COMPANIES (n={len(eval_companies)}):")
    for ec in eval_companies:
        print(f"    ds={ec['dataset_name']:<14} -> idx={ec['index_name']:<22} "
              f"ds_rows={ec['n_dataset_rows']:<3} corpus_chunks={ec['n_corpus_chunks']}")

    # Sample rows per company
    selected_rows: list[dict[str, Any]] = []
    for ec in eval_companies:
        cr = [r for r in all_rows if r.get("company") == ec["dataset_name"]]
        cr.sort(key=lambda r: r.get("id", ""))
        picked = cr[: CONFIG["ROWS_PER_COMPANY"]]
        for r in picked:
            r["_index_company"] = ec["index_name"]
            r["_dataset_company"] = ec["dataset_name"]
        selected_rows.extend(picked)
    print(f"[Stage 1] sampled {len(selected_rows)} rows across {len(eval_companies)} companies")

    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "chunks": chunks,
        "vectors": full_vecs,
        "vectors_shape": list(full_vecs.shape),
        "vectors_dtype": str(full_vecs.dtype),
        "company_to_indices": company_to_indices,
        "selected_rows": selected_rows,
        "eval_companies": eval_companies,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_row_count": len(all_rows),
    }


# =============================== STAGE 2 ===============================
def build_noise_mask(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(chunks)
    mask = np.zeros(n, dtype=bool)
    rule_too_short = np.zeros(n, dtype=bool)
    rule_replacement = np.zeros(n, dtype=bool)
    rule_too_few_wordlike = np.zeros(n, dtype=bool)
    rule_nonalnum_heavy = np.zeros(n, dtype=bool)

    for i, c in enumerate(chunks):
        text = c.get("text", "") or ""
        if not text:
            mask[i] = True
            rule_too_short[i] = True
            continue
        tokens = text.split()
        if len(tokens) < CONFIG["MIN_TOKENS"]:
            rule_too_short[i] = True
        if "�" in text:
            rule_replacement[i] = True
        wordlike = sum(1 for t in tokens if WORDLIKE_RE.search(t))
        if wordlike < CONFIG["MIN_WORDLIKE"]:
            rule_too_few_wordlike[i] = True
        total = len(text)
        if total > 0:
            non_alnum = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
            if (non_alnum / total) > CONFIG["MAX_NONALNUM_RATIO"]:
                rule_nonalnum_heavy[i] = True
        mask[i] = (rule_too_short[i] or rule_replacement[i] or
                   rule_too_few_wordlike[i] or rule_nonalnum_heavy[i])

    counts = {
        "too_short_lt5": int(rule_too_short.sum()),
        "replacement_char": int(rule_replacement.sum()),
        "too_few_wordlike_lt3": int(rule_too_few_wordlike.sum()),
        "nonalnum_ratio_gt_0_5": int(rule_nonalnum_heavy.sum()),
        "total_unique_noise": int(mask.sum()),
    }
    counts["total_pct"] = counts["total_unique_noise"] / n
    print(f"[Stage 2] noise mask: {counts['total_unique_noise']} chunks "
          f"({counts['total_pct']:.1%} of corpus)")
    return {"mask": mask, "counts": counts}


# =============================== JUDGE PROMPTS ===============================
JUDGE_INSTRUCTION_TOPK = (
    "You are a retrieval-quality judge. For each numbered passage, decide whether it helps "
    "answer the question:\n"
    " - \"relevant\": directly provides information that answers the question\n"
    " - \"partially_relevant\": related to the question but missing key details\n"
    " - \"not_relevant\": unrelated, contentless, or noise\n"
    "Then judge whether the question can be answered using these passages as a whole:\n"
    " - \"yes\" (fully), \"partial\", or \"no\".\n"
    "If answerable != \"yes\", add a one-sentence \"why_not\" explaining what's missing or wrong.\n"
    "Output STRICT JSON ONLY:\n"
    "{\"passages\":[{\"index\":<int>,\"verdict\":\"relevant|partially_relevant|not_relevant\"}],"
    "\"answerable\":\"yes|partial|no\",\"why_not\":\"<one sentence or empty>\"}\n"
    "No markdown, no commentary."
)


JUDGE_INSTRUCTION_COVERAGE = (
    "You are evaluating a single retrieved passage against a question. Decide whether THIS "
    "passage alone answers the question:\n"
    " - \"yes\": the passage clearly contains the answer\n"
    " - \"partial\": the passage is related but missing key details\n"
    " - \"no\": the passage does not answer the question\n"
    "Add a one-sentence \"reason\".\n"
    "Output STRICT JSON ONLY:\n"
    "{\"verdict\":\"yes|partial|no\",\"reason\":\"<one sentence>\"}\n"
    "No markdown."
)


def _build_topk_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    cap = CONFIG["PASSAGE_CHAR_CAP"]
    blocks = []
    for i, p in enumerate(passages, start=1):
        text = p.get("text", "") or ""
        if len(text) > cap:
            text = text[:cap] + "…"
        blocks.append(f"[{i}] (kind={p.get('kind','?')}) {text}")
    return f"{JUDGE_INSTRUCTION_TOPK}\n\nQuestion: {question}\n\nPassages:\n" + "\n\n".join(blocks)


def _build_coverage_prompt(question: str, chunk: dict[str, Any]) -> str:
    cap = CONFIG["PASSAGE_CHAR_CAP"]
    text = chunk.get("text", "") or ""
    if len(text) > cap:
        text = text[:cap] + "…"
    return (f"{JUDGE_INSTRUCTION_COVERAGE}\n\nQuestion: {question}\n\n"
            f"Passage (kind={chunk.get('kind','?')}, company={chunk.get('company','?')}):\n{text}")


# =============================== STAGE 3 ===============================
def measure_rows(inputs: dict[str, Any], noise_mask: np.ndarray) -> list[dict[str, Any]]:
    chunks = inputs["chunks"]
    vectors = inputs["vectors"]
    rows = inputs["selected_rows"]
    company_to_indices = inputs["company_to_indices"]
    idx_dir = Path(CONFIG["INDEX_DIR"])
    client = AlbertClient(api_key=os.environ["ALBERT_API_KEY"], base_url=ALBERT_BASE)
    embed_model = client.get_embedding_model(preferred="bge-m3")
    top_k = CONFIG["TOP_K"]
    cand_k = CONFIG["CANDIDATE_K"]

    qvec_cache: dict[str, np.ndarray] = {}

    out: list[dict[str, Any]] = []
    n = len(rows)
    for ri, r in enumerate(rows):
        q = r.get("question", "")
        ds_co = r["_dataset_company"]
        idx_co = r["_index_company"]
        rec: dict[str, Any] = {
            "row_id": r.get("id", ""),
            "dataset_company": ds_co,
            "index_company": idx_co,
            "question": q,
            "question_type": r.get("question_type", ""),
        }

        # --- 1. DENSE: question -> top-8 over full corpus ---
        if q in qvec_cache:
            qvec = qvec_cache[q]
        else:
            emb = client.create_embeddings(embed_model, [q])[0]
            qvec = np.asarray(emb, dtype=np.float32)
            qn = np.linalg.norm(qvec)
            if qn > 0:
                qvec = qvec / qn
            qvec_cache[q] = qvec
        sims = vectors @ qvec
        dense_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        dense_idx = dense_idx[np.argsort(-sims[dense_idx], kind="stable")]
        dense_top = []
        for gi in dense_idx:
            c = chunks[int(gi)]
            dense_top.append({"chunk_id": c["chunk_id"], "kind": c.get("chunk_kind", ""),
                              "company": c.get("company", ""), "text": c["text"],
                              "cos": float(sims[int(gi)])})
        dense_judge = _judge_topk(client, q, dense_top)
        rec["dense_top8"] = dense_top
        rec["dense_judge"] = dense_judge
        rec["dense_relevance_score"] = dense_judge.get("relevance_score", 0.0)
        rec["dense_answerable"] = dense_judge.get("answerable", "no")
        rec["dense_n_tablefact"] = sum(1 for p in dense_top if p["kind"] == "table_fact")
        rec["dense_n_narrative"] = sum(1 for p in dense_top if p["kind"] == "narrative")

        # --- 2. COVERAGE: embed gold span -> nearest eligible chunk in this company ---
        gold = [g for g in _parse_expected_contexts(r) if isinstance(g, str) and g.strip()]
        if not gold:
            rec["coverage_verdict"] = "no_gold"
            rec["coverage_reason"] = "No gold context in dataset row"
            rec["coverage_chunk"] = None
            rec["coverage_chunk_cos_to_gold"] = None
        else:
            gold_snippet = gold[0]
            emb_g = client.create_embeddings(embed_model, [gold_snippet])[0]
            gvec = np.asarray(emb_g, dtype=np.float32)
            gn = np.linalg.norm(gvec)
            if gn > 0:
                gvec = gvec / gn
            company_indices = company_to_indices.get(idx_co, [])
            eligible = [i for i in company_indices if not noise_mask[i]]
            if not eligible:
                rec["coverage_verdict"] = "no_eligible"
                rec["coverage_reason"] = f"No eligible {idx_co} chunks (all masked)"
                rec["coverage_chunk"] = None
                rec["coverage_chunk_cos_to_gold"] = None
            else:
                elig_arr = np.array(eligible, dtype=np.int64)
                sub_vecs = vectors[elig_arr]
                sims_g = sub_vecs @ gvec
                best_local = int(np.argmax(sims_g))
                best_idx = int(elig_arr[best_local])
                best_cos = float(sims_g[best_local])
                best_chunk = chunks[best_idx]
                cov_chunk = {"chunk_id": best_chunk["chunk_id"],
                             "kind": best_chunk.get("chunk_kind", ""),
                             "company": best_chunk.get("company", ""),
                             "text": best_chunk["text"], "cos": best_cos}
                cov_judge = _judge_coverage(client, q, cov_chunk)
                rec["coverage_verdict"] = cov_judge.get("verdict", "no")
                rec["coverage_reason"] = cov_judge.get("reason", "")
                rec["coverage_chunk"] = cov_chunk
                rec["coverage_chunk_cos_to_gold"] = best_cos

        # --- 3. HYBRID: retrieve_chunks(hybrid) -> top-8 + judge ---
        try:
            hy_chunks, _ = retrieve_chunks(
                index_dir=idx_dir, question=q, top_k=top_k,
                retrieval_architecture="hybrid", search_breadth=cand_k,
                base_url=ALBERT_BASE,
            )
        except Exception as exc:
            print(f"  [row {ri+1}] hybrid ERROR: {type(exc).__name__}: {exc}")
            hy_chunks = []
        hy_top = []
        for hc in hy_chunks:
            hy_top.append({"chunk_id": hc.get("chunk_id", ""),
                           "kind": hc.get("chunk_kind", ""),
                           "company": hc.get("company_label") or hc.get("company", ""),
                           "text": hc.get("text", ""),
                           "cos": float(hc.get("score", 0.0))})
        if hy_top:
            hy_judge = _judge_topk(client, q, hy_top)
        else:
            hy_judge = {"relevance_score": 0.0, "answerable": "no", "why_not": "hybrid call failed"}
        rec["hybrid_top8"] = hy_top
        rec["hybrid_judge"] = hy_judge
        rec["hybrid_relevance_score"] = hy_judge.get("relevance_score", 0.0)
        rec["hybrid_answerable"] = hy_judge.get("answerable", "no")

        print(f"[Stage 3] row {ri+1:>2}/{n} {ds_co:<14} -> {idx_co:<22} "
              f"dense={rec['dense_relevance_score']:.3f}({rec['dense_answerable']}) "
              f"cov={rec['coverage_verdict']:<7} "
              f"hyb={rec['hybrid_relevance_score']:.3f}({rec['hybrid_answerable']}) "
              f"tablefact={rec['dense_n_tablefact']}/{top_k}")
        out.append(rec)
    print(f"[Stage 3] embedding cache: {len(qvec_cache)} unique questions; "
          f"embeddings used: {BUDGET['embedding_calls']}")
    return out


def _judge_topk(client: AlbertClient, question: str,
                passages: list[dict[str, Any]]) -> dict[str, Any]:
    if not passages:
        return {"relevance_score": 0.0, "answerable": "no", "why_not": "no passages",
                "n_relevant": 0, "n_partial": 0, "n_irrel": 0}
    prompt = _build_topk_prompt(question, passages)
    BUDGET["judge_calls"] += 1
    try:
        raw = client.chat_completion(CONFIG["JUDGE"],
                                     [{"role": "user", "content": prompt}], temperature=0.0)
        parsed = _parse_json_object(raw)
        verdicts = parsed.get("passages", [])
        relevant = sum(1 for v in verdicts if v.get("verdict") == "relevant")
        partial = sum(1 for v in verdicts if v.get("verdict") == "partially_relevant")
        total = len(verdicts) or len(passages)
        score = (relevant + 0.5 * partial) / max(total, 1)
        return {"relevance_score": round(score, 4),
                "answerable": parsed.get("answerable", "no"),
                "why_not": parsed.get("why_not", ""),
                "n_relevant": relevant, "n_partial": partial,
                "n_irrel": max(total - relevant - partial, 0),
                "raw": raw}
    except Exception as exc:
        return {"relevance_score": 0.0, "answerable": "judge_error",
                "why_not": f"{type(exc).__name__}: {exc}",
                "n_relevant": 0, "n_partial": 0, "n_irrel": len(passages)}


def _judge_coverage(client: AlbertClient, question: str,
                    chunk: dict[str, Any]) -> dict[str, Any]:
    prompt = _build_coverage_prompt(question, chunk)
    BUDGET["judge_calls"] += 1
    try:
        raw = client.chat_completion(CONFIG["JUDGE"],
                                     [{"role": "user", "content": prompt}], temperature=0.0)
        parsed = _parse_json_object(raw)
        return {"verdict": parsed.get("verdict", "no"),
                "reason": parsed.get("reason", ""),
                "raw": raw}
    except Exception as exc:
        return {"verdict": "judge_error", "reason": f"{type(exc).__name__}: {exc}"}


# =============================== STAGE 4 ===============================
def aggregate_classify(per_row: list[dict[str, Any]]) -> dict[str, Any]:
    by_company: dict[str, dict[str, Any]] = {}
    for r in per_row:
        ds_co = r["dataset_company"]
        d = by_company.setdefault(ds_co, {
            "dataset_company": ds_co,
            "index_company": r["index_company"],
            "n": 0,
            "dense_rel_sum": 0.0, "hybrid_rel_sum": 0.0,
            "coverage_yes": 0, "coverage_partial": 0, "coverage_no": 0,
            "coverage_no_gold": 0, "coverage_no_eligible": 0, "coverage_judge_error": 0,
            "tablefact_sum": 0,
        })
        d["n"] += 1
        d["dense_rel_sum"] += r["dense_relevance_score"]
        d["hybrid_rel_sum"] += r["hybrid_relevance_score"]
        d["tablefact_sum"] += r["dense_n_tablefact"]
        v = r.get("coverage_verdict", "no")
        if v == "yes":
            d["coverage_yes"] += 1
        elif v == "partial":
            d["coverage_partial"] += 1
        elif v == "no":
            d["coverage_no"] += 1
        elif v == "no_gold":
            d["coverage_no_gold"] += 1
        elif v == "no_eligible":
            d["coverage_no_eligible"] += 1
        else:
            d["coverage_judge_error"] += 1

    for ds_co, d in by_company.items():
        n = max(d["n"], 1)
        d["dense_rel"] = d["dense_rel_sum"] / n
        d["hybrid_rel"] = d["hybrid_rel_sum"] / n
        d["hybrid_delta"] = d["hybrid_rel"] - d["dense_rel"]
        d["coverage_yes_rate"] = d["coverage_yes"] / n
        d["coverage_no_rate"] = d["coverage_no"] / n
        d["tablefact_share"] = d["tablefact_sum"] / (n * CONFIG["TOP_K"])
        # Classify (first match)
        if d["dense_rel"] >= CONFIG["GOOD_THRESHOLD"]:
            d["class"] = "GOOD"
        elif d["coverage_yes_rate"] < CONFIG["COVERAGE_GAP_YES_RATE"]:
            d["class"] = "COVERAGE_GAP"
        elif d["hybrid_delta"] >= CONFIG["HYBRID_HELP_DELTA"]:
            d["class"] = "RETRIEVAL_MISMATCH"
        elif d["tablefact_share"] >= CONFIG["TABLEFACT_DOMINANT"]:
            d["class"] = "SEMANTICALLY_EMPTY_TABLES"
        else:
            d["class"] = "MIXED_UNCLEAR"

    overall = {
        "n_rows": len(per_row),
        "n_companies": len(by_company),
        "mean_dense_rel": sum(d["dense_rel"] * d["n"] for d in by_company.values()) /
                          max(len(per_row), 1),
        "mean_hybrid_rel": sum(d["hybrid_rel"] * d["n"] for d in by_company.values()) /
                           max(len(per_row), 1),
        "mean_hybrid_delta": sum(d["hybrid_delta"] * d["n"] for d in by_company.values()) /
                              max(len(per_row), 1),
        "class_counts": {},
    }
    from collections import Counter
    overall["class_counts"] = dict(Counter(d["class"] for d in by_company.values()))
    return {"by_company": by_company, "overall": overall}


PRESCRIPTIONS = {
    "GOOD":                        "No retrieval fix needed; ship as-is.",
    "RETRIEVAL_MISMATCH":          "Ship hybrid retrieval for this company; "
                                    "the same questions on dense miss but hybrid recovers.",
    "SEMANTICALLY_EMPTY_TABLES":   "Pilot contextual-chunking re-embed FIRST on this company "
                                    "(table_fact chunks dominate top-8 but they've lost column "
                                    "labels and units).",
    "COVERAGE_GAP":                "Coverage-limited: the answer text itself is not in the company's "
                                    "documents in the index (semantic-nearest chunk fails 1-on-1 judgement). "
                                    "Flag those rows as data-bound and add documents OR remove the rows.",
    "MIXED_UNCLEAR":               "Needs manual inspection — none of the four signals dominate.",
}


# =============================== STAGE 5 ===============================
def report(provenance: dict[str, Any], inputs: dict[str, Any],
           noise: dict[str, Any], per_row: list[dict[str, Any]],
           agg: dict[str, Any]) -> dict[str, Any]:
    csv_path = Path("retrieval_weakness_map.csv")
    md_path = Path("retrieval_weakness_map.md")

    # ---- CSV ----
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "section", "dataset_company", "index_company", "row_id", "question_type",
            "dense_relevance", "dense_answerable", "dense_n_tablefact",
            "coverage_verdict", "coverage_reason",
            "hybrid_relevance", "hybrid_answerable",
        ])
        for r in per_row:
            w.writerow([
                "row", r["dataset_company"], r["index_company"], r["row_id"],
                r.get("question_type", ""),
                f"{r['dense_relevance_score']:.4f}", r["dense_answerable"], r["dense_n_tablefact"],
                r["coverage_verdict"], truncate(r.get("coverage_reason", ""), 200),
                f"{r['hybrid_relevance_score']:.4f}", r["hybrid_answerable"],
            ])
        w.writerow([])
        w.writerow(["section", "dataset_company", "index_company", "n",
                    "dense_rel", "hybrid_rel", "hybrid_delta",
                    "coverage_yes_rate", "coverage_no_rate", "tablefact_share", "class"])
        for ds_co, d in sorted(agg["by_company"].items(),
                                key=lambda kv: kv[1]["dense_rel"]):
            w.writerow([
                "company_agg", ds_co, d["index_company"], d["n"],
                f"{d['dense_rel']:.4f}", f"{d['hybrid_rel']:.4f}",
                f"{d['hybrid_delta']:+.4f}",
                f"{d['coverage_yes_rate']:.4f}", f"{d['coverage_no_rate']:.4f}",
                f"{d['tablefact_share']:.4f}", d["class"],
            ])

    # ---- Markdown ----
    lines: list[str] = []
    lines.append("# Pre-Lap-2 retrieval weakness map")
    lines.append("")
    lines.append(f"- Eval companies: **{len(agg['by_company'])}** "
                 f"(alias map: ENGIE→Engie, Schneider→Schneider Electric; "
                 f"excluded no-corpus: Hermès, Iberdrola, Unknown)")
    lines.append(f"- Rows: **{agg['overall']['n_rows']}** "
                 f"(cap {CONFIG['ROWS_PER_COMPANY']}/company, first-by-id)")
    lines.append(f"- Judge: `{CONFIG['JUDGE']}` (Q + passages only, NO gold, NO answer)")
    lines.append(f"- Embedding: `BAAI/bge-m3`  noise-masked corpus: "
                 f"{noise['counts']['total_unique_noise']} chunks ({noise['counts']['total_pct']:.1%})")
    lines.append("")

    # ---- Per-company table (sorted ascending by dense_rel) ----
    lines.append("## Per-company table  (sorted ascending by dense_rel)")
    lines.append("")
    lines.append("| company | idx_name | n | dense_rel | hybrid_rel | Δ | cov_yes% | cov_no% | tablefact% | CLASS |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for ds_co, d in sorted(agg["by_company"].items(),
                            key=lambda kv: kv[1]["dense_rel"]):
        lines.append(f"| {ds_co} | {d['index_company']} | {d['n']} | "
                     f"{d['dense_rel']:.3f} | {d['hybrid_rel']:.3f} | "
                     f"{d['hybrid_delta']:+.3f} | "
                     f"{d['coverage_yes_rate']:.0%} | {d['coverage_no_rate']:.0%} | "
                     f"{d['tablefact_share']:.0%} | **{d['class']}** |")
    lines.append("")

    # ---- Overall ----
    ov = agg["overall"]
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Corpus-wide mean dense relevance: **{ov['mean_dense_rel']:.3f}**")
    lines.append(f"- Corpus-wide mean hybrid relevance: **{ov['mean_hybrid_rel']:.3f}**")
    lines.append(f"- Mean hybrid Δ (hybrid − dense): **{ov['mean_hybrid_delta']:+.3f}**")
    lines.append("")
    lines.append("Companies per class:")
    lines.append("")
    lines.append("| class | n |")
    lines.append("|---|---:|")
    for cls, n in sorted(ov["class_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| **{cls}** | {n} |")
    lines.append("")

    # ---- WORST-3 FORENSICS ----
    worst3 = sorted(agg["by_company"].items(), key=lambda kv: kv[1]["dense_rel"])[:3]
    lines.append("## Worst-3 forensics  (2 example rows per company)")
    lines.append("")
    for ds_co, d in worst3:
        lines.append(f"### {ds_co}  →  index `{d['index_company']}`  "
                     f"(dense_rel={d['dense_rel']:.3f}, class=**{d['class']}**)")
        lines.append("")
        sample_rows = [r for r in per_row if r["dataset_company"] == ds_co][:2]
        for r in sample_rows:
            lines.append(f"**Row `{r['row_id']}`** — {r['question']}")
            lines.append("")
            lines.append(f"- dense_rel={r['dense_relevance_score']:.3f} "
                         f"({r['dense_answerable']}) "
                         f"hybrid_rel={r['hybrid_relevance_score']:.3f} "
                         f"({r['hybrid_answerable']}) "
                         f"tablefact_top8={r['dense_n_tablefact']}/8")
            why = r.get("dense_judge", {}).get("why_not", "")
            if why:
                lines.append(f"- judge note (dense): _{truncate(why, 200)}_")
            lines.append(f"- coverage verdict (gold-span → nearest {r['index_company']} chunk): "
                         f"**{r['coverage_verdict']}**")
            cov_reason = truncate(r.get("coverage_reason", ""), 200)
            if cov_reason:
                lines.append(f"- coverage reason: _{cov_reason}_")
            cov_chunk = r.get("coverage_chunk")
            if cov_chunk:
                lines.append(f"- coverage nearest chunk `{cov_chunk['chunk_id']}` "
                             f"(kind={cov_chunk['kind']}, cos_to_gold="
                             f"{r['coverage_chunk_cos_to_gold']:.3f}):")
                lines.append("")
                lines.append("  > " + truncate(cov_chunk["text"], 250))
            lines.append("")
            lines.append("**Dense top-8** (cos | kind | preview):")
            for j, p in enumerate(r["dense_top8"], start=1):
                lines.append(f"  {j}. `{p['chunk_id']}` cos={p['cos']:.3f} "
                             f"kind={p['kind']} → {truncate(p['text'], 150)!r}")
            lines.append("")

    # ---- Prescriptions ----
    lines.append("## Per-company prescription")
    lines.append("")
    lines.append("| company | class | prescription |")
    lines.append("|---|---|---|")
    for ds_co, d in sorted(agg["by_company"].items(),
                            key=lambda kv: kv[1]["dense_rel"]):
        lines.append(f"| {ds_co} | {d['class']} | {PRESCRIPTIONS[d['class']]} |")
    lines.append("")

    # ---- Lap-2 readiness verdict ----
    ready_dense = [ds for ds, d in agg["by_company"].items() if d["class"] == "GOOD"]
    ready_hybrid = [ds for ds, d in agg["by_company"].items() if d["class"] == "RETRIEVAL_MISMATCH"]
    needs_fix = [ds for ds, d in agg["by_company"].items()
                  if d["class"] == "SEMANTICALLY_EMPTY_TABLES"]
    coverage_limited = [ds for ds, d in agg["by_company"].items() if d["class"] == "COVERAGE_GAP"]
    mixed = [ds for ds, d in agg["by_company"].items() if d["class"] == "MIXED_UNCLEAR"]

    lines.append("## Lap-2 readiness verdict")
    lines.append("")
    lines.append(f"- **Ready (GOOD on dense)**: {', '.join(ready_dense) or '—'}")
    lines.append(f"- **Ready IF we ship hybrid**: {', '.join(ready_hybrid) or '—'}")
    lines.append(f"- **Pre-Lap-2 fix needed (re-embed pilot — start here)**: "
                 f"{', '.join(needs_fix) or '—'}")
    lines.append(f"- **Coverage-limited (data ceiling, not model failure)**: "
                 f"{', '.join(coverage_limited) or '—'}")
    lines.append(f"- **Manual inspection**: {', '.join(mixed) or '—'}")
    lines.append("")
    lines.append("Lap-2 results for the coverage-limited companies must be read as a documents-side "
                 "ceiling — adding documents or removing the rows is the correct response, not changing "
                 "the retriever.")
    lines.append("")
    lines.append(f"_Manifest: retrieval_weakness_map.run_manifest.json_")
    md = "\n".join(lines)
    md_path.write_text(md, encoding="utf-8")
    print()
    print(md)

    return {
        "lap2_ready_dense": ready_dense,
        "lap2_ready_hybrid": ready_hybrid,
        "lap2_needs_fix": needs_fix,
        "lap2_coverage_limited": coverage_limited,
        "lap2_mixed": mixed,
    }


# =============================== STAGE 6 ===============================
def write_manifest(provenance: dict[str, Any], inputs: dict[str, Any],
                   noise: dict[str, Any], per_row: list[dict[str, Any]],
                   agg: dict[str, Any], verdict: dict[str, Any]) -> None:
    # Strip heavy fields from per_row before writing
    slim_rows = []
    for r in per_row:
        slim_rows.append({
            "row_id": r["row_id"], "dataset_company": r["dataset_company"],
            "index_company": r["index_company"], "question_type": r.get("question_type", ""),
            "dense_relevance_score": r["dense_relevance_score"],
            "dense_answerable": r["dense_answerable"],
            "dense_n_tablefact": r["dense_n_tablefact"],
            "dense_n_narrative": r["dense_n_narrative"],
            "dense_why_not": r.get("dense_judge", {}).get("why_not", ""),
            "coverage_verdict": r["coverage_verdict"],
            "coverage_reason": r.get("coverage_reason", ""),
            "coverage_chunk_id": (r["coverage_chunk"] or {}).get("chunk_id") if r.get("coverage_chunk") else None,
            "coverage_chunk_kind": (r["coverage_chunk"] or {}).get("kind") if r.get("coverage_chunk") else None,
            "coverage_chunk_cos_to_gold": r.get("coverage_chunk_cos_to_gold"),
            "hybrid_relevance_score": r["hybrid_relevance_score"],
            "hybrid_answerable": r["hybrid_answerable"],
            "hybrid_why_not": r.get("hybrid_judge", {}).get("why_not", ""),
        })
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "code_state": {
            "git_head": provenance["git_head"],
            "git_status_porcelain": provenance["git_status_porcelain"],
            "git_diff_stat_app": provenance["git_diff_stat_app"],
            "code_sha256": provenance["code_sha256"],
        },
        "inputs": {
            "index_dir": CONFIG["INDEX_DIR"],
            "manifest_sha256": inputs["manifest_sha256"],
            "vectors_shape": inputs["vectors_shape"],
            "vectors_dtype": inputs["vectors_dtype"],
            "chunk_ids_file_present": False,
            "chunk_ids_source": "chunks.json[i].chunk_id",
            "alignment": "positional (len-verified)",
            "dataset_path": inputs["dataset_path"],
            "dataset_sha256": inputs["dataset_sha256"],
            "dataset_row_count": inputs["dataset_row_count"],
        },
        "selection": {
            "alias_map": CONFIG["ALIAS_MAP"],
            "excluded_companies": sorted(CONFIG["EXCLUDE_COMPANIES"]),
            "rows_per_company": CONFIG["ROWS_PER_COMPANY"],
            "eval_companies": inputs["eval_companies"],
        },
        "thresholds": {
            "GOOD_THRESHOLD": CONFIG["GOOD_THRESHOLD"],
            "HYBRID_HELP_DELTA": CONFIG["HYBRID_HELP_DELTA"],
            "COVERAGE_GAP_YES_RATE": CONFIG["COVERAGE_GAP_YES_RATE"],
            "TABLEFACT_DOMINANT": CONFIG["TABLEFACT_DOMINANT"],
        },
        "noise_mask_stats": noise["counts"],
        "judge_model": CONFIG["JUDGE"],
        "per_company": agg["by_company"],
        "overall": agg["overall"],
        "lap2_verdict": verdict,
        "prescriptions": PRESCRIPTIONS,
        "per_row": slim_rows,
        "budget": BUDGET,
        "determinism_notes": (
            "Embedding cache + raw cosine + stable argsort are deterministic; "
            "hybrid retrieval is the app's deterministic path; the Mistral judge is "
            "called at temperature=0 but service-dependent."
        ),
        "read_only_git_commands_used": READ_ONLY_GIT_COMMANDS_USED,
    }
    Path("retrieval_weakness_map.run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================== MAIN ===============================
BANNER = """\
====================================================================
PIPELINE FLOW — Pre-Lap-2 retrieval weakness map
  Stage 0 capture_provenance()
  Stage 1 load_inputs()        + alias + EVAL_COMPANIES
  Stage 2 build_noise_mask()
  Stage 3 measure_rows()       dense / coverage / hybrid (3 judge calls/row)
  Stage 4 aggregate_classify() per-company class
  Stage 5 report()             md + csv (worst-3, prescriptions, verdict)
  Stage 6 write_manifest()
===================================================================="""


def main() -> int:
    t0 = time.time()
    print(BANNER)
    print()

    print("[Stage 0] capturing provenance")
    provenance = capture_provenance()

    print()
    print("[Stage 1] loading inputs + deriving EVAL_COMPANIES")
    inputs = load_inputs()

    print()
    print("[Stage 2] building noise mask")
    noise = build_noise_mask(inputs["chunks"])

    print()
    print("[Stage 3] measuring rows (3 judge calls each)")
    per_row = measure_rows(inputs, noise["mask"])

    print()
    print("[Stage 4] aggregating + classifying")
    agg = aggregate_classify(per_row)
    print(f"[Stage 4] class counts: {agg['overall']['class_counts']}")

    print()
    print("[Stage 5] writing report")
    verdict = report(provenance, inputs, noise, per_row, agg)

    print()
    print("[Stage 6] writing manifest")
    write_manifest(provenance, inputs, noise, per_row, agg, verdict)

    print()
    print("=" * 78)
    print("BUDGET SPENT")
    for k, v in BUDGET.items():
        print(f"  {k:22s} = {v}")
    print(f"  total elapsed         = {time.time() - t0:.1f}s")
    print("=" * 78)
    print()
    print("Read-only git commands used:")
    for cmd in READ_ONLY_GIT_COMMANDS_USED:
        print(f"  {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
