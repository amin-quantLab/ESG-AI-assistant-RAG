"""Retrieval Sweep + Batched Rerank — read-only diagnostic, extends the report card.

PIPELINE FLOW
  Stage 0  capture_provenance()        -> manifest fields (code state, env, rerank endpoint, row-set cross-check)
  Stage 1  load_inputs()               -> index arrays + rows + alignment + row-set assert
  Stage 2  reachability()              -> reuse logic (expect 100%); per-company counts
  Stage 3a pool_recall_sweep()         -> raw full-corpus cosine; rank-of-first-gold-hit; recall@K (FREE)
  Stage 3b batched_ce()                -> dense_ce + narrative_ce @ CE_POOL_K with adaptive batched rerank
  Stage 4  score()                     -> CE top-8 hit/partial
  Stage 5  aggregate_and_report()      -> retrieval_sweep_rerank.md + .csv
  Stage 6  write_manifest()            -> retrieval_sweep_rerank.run_manifest.json

NO judge calls. NO generation. NO edits to app/*. NO git writes. NO pip/env changes.
Read-only git inspection (rev-parse, status --porcelain, diff --stat) only for code-state fingerprinting.
Secrets loaded via setdefault from ..\\esg_scraper\\.env and never printed.

Run from eval/:  PYTHONPATH=.. python retrieval_sweep_rerank.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =============================== CONFIG ===============================
CONFIG: dict[str, Any] = {
    "INDEX_DIR":          "../outputs/rag_index",
    "DATASET":            "../sample_data/ragas_esg_eval_dataset.csv",
    "COMPANIES":          ["Danone", "Enel", "TotalEnergies", "Volkswagen"],
    "ROWS_PER_COMPANY":   8,
    "TOP_K":              8,
    "K_SWEEP":            [100, 250, 500, 1000],
    "CE_POOL_K":          100,
    "RERANK_MODEL":       "BAAI/bge-reranker-v2-m3",
    "RERANK_BATCH_SIZES": [20, 10, 5],
    "DOC_CHAR_CAP":       2000,
    # No earlier-manifest cross-check by default in the committed eval/ tree (the
    # report_card iteration that produced "outputs/retrieval_report.run_manifest.json"
    # was the broken-rerank precursor and is not committed). Leave None; load_inputs
    # already handles the absent-prior case gracefully.
    "PRIOR_MANIFEST":     None,
}

# Inferred at runtime
RERANK_ENDPOINT_PATH = "/rerank"  # AlbertClient prepends base_url (which already ends in /v1)
HIT_THRESHOLD = 90.0
PARTIAL_THRESHOLD = 70.0
HIST_BUCKETS = [(1, 8), (9, 100), (101, 250), (251, 500), (501, 1000), (1001, 10**9)]

BUDGET = {"chat_calls": 0, "rerank_calls": 0, "embedding_calls": 0,
          "embedding_strings": 0, "judge_calls": 0, "http_413_total": 0}

READ_ONLY_GIT_COMMANDS_USED: list[str] = []


# =============================== ENV LOAD ===============================
def load_env() -> None:
    """Load .env via setdefault. Works whether the script is run from the repo root
    (Git_clone/) or from eval/. Never prints secrets."""
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
    sys.stderr.write("ALBERT_API_KEY missing — see .env loader. Aborting.\n")
    sys.exit(1)

from app.rag import (                       # noqa: E402
    AlbertClient, DEFAULT_BASE_URL, retrieve_chunks,
)
from app.rag_eval import (                  # noqa: E402
    _parse_expected_contexts, _score_retrieval, _score_chunk_against_snippet,
)

ALBERT_BASE = os.environ.get("ALBERT_BASE_URL") or DEFAULT_BASE_URL


# Instrument AlbertClient for budget counting
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


class BatchTooLargeError(Exception):
    """A rerank sub-batch returned HTTP 413."""


def _call_rerank(client: AlbertClient, query: str, docs: list[str]) -> dict[str, Any]:
    payload = {"model": CONFIG["RERANK_MODEL"], "query": query, "documents": docs}
    BUDGET["rerank_calls"] += 1
    try:
        r = client._request("POST", RERANK_ENDPOINT_PATH, json=payload)
    except requests.HTTPError as exc:
        resp = getattr(exc, "response", None)
        if resp is not None and resp.status_code == 413:
            BUDGET["http_413_total"] += 1
            raise BatchTooLargeError(f"413 on batch of {len(docs)}") from exc
        raise
    return r.json()


def _rerank_batch_pool(client: AlbertClient, query: str, pool: list[dict[str, Any]],
                       batch_size: int, char_cap: int | None) -> list[tuple[int, float]]:
    """Rerank `pool` in contiguous sub-batches of `batch_size`. Returns [(pool_index, score), ...]
    pre-sort. Raises BatchTooLargeError on the first 413 in any sub-batch."""
    scored: list[tuple[int, float]] = []
    for start in range(0, len(pool), batch_size):
        sub = pool[start:start + batch_size]
        texts = [(d["text"][:char_cap] if char_cap else d["text"]) for d in sub]
        body = _call_rerank(client, query, texts)
        for res in body.get("results", []):
            j = int(res["index"])
            score = float(res["relevance_score"])
            global_idx = sub[j]["pool_index"]
            scored.append((global_idx, score))
    return scored


def batched_rerank_adaptive(client: AlbertClient, query: str, pool: list[dict[str, Any]]
                           ) -> dict[str, Any]:
    """Try batch sizes in CONFIG['RERANK_BATCH_SIZES']; on 413 step down. Last resort: smallest + char_cap.
    Returns {top_k_pool: list[dict], batch_size_used: int|None, truncation_applied: bool,
             ce_failed: bool, attempts: list[dict]}.
    """
    attempts: list[dict[str, Any]] = []
    for size in CONFIG["RERANK_BATCH_SIZES"]:
        try:
            scored = _rerank_batch_pool(client, query, pool, size, char_cap=None)
            sorted_scored = sorted(scored, key=lambda p: -p[1])
            top_indices = [idx for idx, _s in sorted_scored[:CONFIG["TOP_K"]]]
            attempts.append({"batch_size": size, "char_cap": None, "result": "ok"})
            top = []
            for idx in top_indices:
                d = dict(pool[idx])
                d["score"] = next(s for (i, s) in sorted_scored if i == idx)
                d["reranker"] = "bge-reranker-v2-m3"
                top.append(d)
            return {"top_k_pool": top, "batch_size_used": size,
                    "truncation_applied": False, "ce_failed": False, "attempts": attempts}
        except BatchTooLargeError as exc:
            attempts.append({"batch_size": size, "char_cap": None, "result": f"413: {exc}"})
            continue

    # Last resort: smallest size + char_cap
    smallest = CONFIG["RERANK_BATCH_SIZES"][-1]
    try:
        scored = _rerank_batch_pool(client, query, pool, smallest,
                                    char_cap=CONFIG["DOC_CHAR_CAP"])
        sorted_scored = sorted(scored, key=lambda p: -p[1])
        top_indices = [idx for idx, _s in sorted_scored[:CONFIG["TOP_K"]]]
        attempts.append({"batch_size": smallest, "char_cap": CONFIG["DOC_CHAR_CAP"], "result": "ok"})
        top = []
        for idx in top_indices:
            d = dict(pool[idx])
            d["score"] = next(s for (i, s) in sorted_scored if i == idx)
            d["reranker"] = "bge-reranker-v2-m3"
            top.append(d)
        return {"top_k_pool": top, "batch_size_used": smallest,
                "truncation_applied": True, "ce_failed": False, "attempts": attempts}
    except BatchTooLargeError as exc:
        attempts.append({"batch_size": smallest, "char_cap": CONFIG["DOC_CHAR_CAP"],
                         "result": f"413 (truncated): {exc}"})
        return {"top_k_pool": [], "batch_size_used": None,
                "truncation_applied": True, "ce_failed": True, "attempts": attempts}


# =============================== STAGE 0: PROVENANCE ===============================
def capture_provenance() -> dict[str, Any]:
    """Read-only git fingerprint of code state + rerank probe."""
    head = git_record(["rev-parse", "HEAD"])
    porcelain = git_record(["status", "--porcelain"])
    diff_stat = git_record(["diff", "--stat", "--", "app/*.py"])

    app_hashes: dict[str, str] = {}
    for fname in ("app/rag.py", "app/ragas_eval.py", "app/cli.py"):
        p = Path(fname)
        app_hashes[fname] = sha256_file(p) if p.is_file() else "<missing>"

    # 2-doc rerank probe
    client = AlbertClient(api_key=os.environ["ALBERT_API_KEY"], base_url=ALBERT_BASE)
    probe_ok = False
    probe_note = ""
    try:
        body = _call_rerank(client, "GHG emissions",
                            ["Scope 1 emissions were 33 Mt CO2e",
                             "The staff cafeteria menu changed in May"])
        results = sorted(body.get("results", []), key=lambda x: -x["relevance_score"])
        if results and results[0]["index"] == 0:
            probe_ok = True
            probe_note = (f"probe ok: doc0 score={results[0]['relevance_score']:.6g} > "
                          f"doc1 score={results[-1]['relevance_score']:.6g}")
        else:
            probe_note = f"probe returned unexpected order: {results}"
    except Exception as exc:
        probe_note = f"{type(exc).__name__}: {exc}"

    return {
        "git_head": head,
        "git_status_porcelain": porcelain,
        "git_diff_stat_app": diff_stat,
        "code_sha256": app_hashes,
        "rerank_endpoint": f"{ALBERT_BASE.rstrip('/')}/rerank",
        "rerank_request_shape": '{"model": <str>, "query": <str>, "documents": [<str>, ...]}',
        "rerank_response_shape": '{"results": [{"index": <int>, "relevance_score": <float>}, ...]}',
        "rerank_probe_ok": probe_ok,
        "rerank_probe_note": probe_note,
    }


# =============================== STAGE 1: LOAD INPUTS ===============================
def load_inputs(prior_selected_ids: list[str] | None) -> dict[str, Any]:
    """Load index + rows, re-assert positional alignment, cross-check selected_row_ids."""
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
        raise SystemExit(f"STOP: length mismatch chunks={len(chunks)} vectors={vectors_mmap.shape[0]}")
    if vectors_mmap.dtype != np.float32 or vectors_mmap.shape[1] != 1024:
        raise SystemExit(f"STOP: vectors dtype={vectors_mmap.dtype} shape={vectors_mmap.shape}")

    print(f"[Stage 1] positional alignment re-asserted: "
          f"len(chunks)==len(vectors)==145197, vectors {vectors_mmap.shape} {vectors_mmap.dtype}")

    # narrative mask + pre-normalized narrative submatrix
    narrative_idx = np.array(
        [i for i, c in enumerate(chunks) if c.get("chunk_kind") == "narrative"],
        dtype=np.int64,
    )
    narrative_vecs = np.asarray(vectors_mmap[narrative_idx], dtype=np.float32)
    nnorms = np.linalg.norm(narrative_vecs, axis=1, keepdims=True)
    nnorms[nnorms == 0] = 1.0
    narrative_vecs /= nnorms
    print(f"[Stage 1] narrative submatrix {narrative_vecs.shape} L2-normalized")

    # Materialize + L2-normalize the FULL vector matrix once (~600 MB) so Stage 3a is fast.
    print("[Stage 1] materializing full vector matrix into RAM (this is ~600 MB)...")
    full_vecs = np.asarray(vectors_mmap, dtype=np.float32)
    fnorms = np.linalg.norm(full_vecs, axis=1)
    nmin, nmax, nmean = float(fnorms.min()), float(fnorms.max()), float(fnorms.mean())
    if not np.allclose(fnorms, 1.0, atol=1e-3):
        print(f"[Stage 1] vectors NOT pre-normalized (norms min={nmin:.4f} max={nmax:.4f} "
              f"mean={nmean:.4f}); normalizing in place")
        full_vecs /= fnorms[:, None]
    else:
        print(f"[Stage 1] vectors pre-normalized OK (norms min={nmin:.4f} max={nmax:.4f} "
              f"mean={nmean:.4f})")

    # dataset selection (must match prior manifest)
    with dataset_path.open("r", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))
    selected_rows: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for company in CONFIG["COMPANIES"]:
        cr = [r for r in all_rows if r.get("company") == company]
        cr.sort(key=lambda r: r.get("id", ""))
        picked = cr[: CONFIG["ROWS_PER_COMPANY"]]
        selected_rows.extend(picked)
        selected_ids.extend(r.get("id", "") for r in picked)

    if prior_selected_ids is None:
        row_set_matches_prior = None
        print("[Stage 1] no prior manifest -> no row-set cross-check")
    else:
        list_match = selected_ids == prior_selected_ids
        set_match = set(selected_ids) == set(prior_selected_ids)
        row_set_matches_prior = list_match
        print(f"[Stage 1] row-set assertion vs prior manifest: "
              f"list-equal={list_match}, set-equal={set_match}")
        if not set_match:
            raise SystemExit("STOP: row-set drift vs prior manifest")
        if not list_match:
            print("[Stage 1] WARNING: set matches but list-order differs (probably benign)")

    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "chunks": chunks,
        "vectors": full_vecs,            # full L2-normalized, in RAM
        "narrative_idx": narrative_idx,
        "narrative_vecs": narrative_vecs,
        "vectors_shape": list(full_vecs.shape),
        "vectors_dtype": str(full_vecs.dtype),
        "vectors_bytes": vectors_path.stat().st_size,
        "selected_rows": selected_rows,
        "selected_row_ids": selected_ids,
        "row_set_matches_prior": row_set_matches_prior,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_row_count": len(all_rows),
    }


# =============================== STAGE 2: REACHABILITY ===============================
def reachability(inputs: dict[str, Any]) -> dict[str, Any]:
    """Re-compute per-row reachable flag (any company-scoped chunk hits gold at >=90)."""
    chunks = inputs["chunks"]
    rows = inputs["selected_rows"]
    company_to_indices: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        company_to_indices.setdefault(c.get("company", ""), []).append(i)

    per_row: list[dict[str, Any]] = []
    for r in rows:
        company = r.get("company", "")
        gold = [g for g in _parse_expected_contexts(r) if isinstance(g, str) and g.strip()]
        cand_idx = company_to_indices.get(company, [])
        reached = False
        for snip in gold:
            for ci in cand_idx:
                s = _score_chunk_against_snippet(snip, chunks[ci]["text"])
                if s >= HIT_THRESHOLD:
                    reached = True
                    break
            if reached:
                break
        per_row.append({"row_id": r.get("id", ""), "company": company, "reachable": reached})

    pc = {c: {"reachable": 0, "total": 0} for c in CONFIG["COMPANIES"]}
    for rec in per_row:
        pc[rec["company"]]["total"] += 1
        if rec["reachable"]:
            pc[rec["company"]]["reachable"] += 1
    total_reach = sum(1 for r in per_row if r["reachable"])
    print(f"[Stage 2] reachable {total_reach}/{len(per_row)}; "
          f"per_company={pc}")
    return {"per_row": per_row, "per_company": pc,
            "total_reachable": total_reach, "total": len(per_row)}


# =============================== STAGE 3a: POOL-RECALL SWEEP ===============================
def pool_recall_sweep(inputs: dict[str, Any], reach: dict[str, Any]) -> dict[str, Any]:
    """Full-corpus cosine; per-row rank-of-first-hit/partialhit; recall@K + histogram. FREE."""
    chunks = inputs["chunks"]
    vectors = inputs["vectors"]      # L2-normalized, in RAM
    rows = inputs["selected_rows"]
    client = AlbertClient(api_key=os.environ["ALBERT_API_KEY"], base_url=ALBERT_BASE)
    embed_model = client.get_embedding_model(preferred="bge-m3")

    k_max = max(CONFIG["K_SWEEP"])
    K_SET = CONFIG["K_SWEEP"]
    per_row: list[dict[str, Any]] = []
    rank_hist_ph = {f"{lo}-{hi if hi < 10**9 else 'inf'}": 0 for lo, hi in HIST_BUCKETS}

    for ri, r in enumerate(rows):
        q = r.get("question", "")
        gold = [g for g in _parse_expected_contexts(r) if isinstance(g, str) and g.strip()]
        # 1 embed per row
        emb = client.create_embeddings(embed_model, [q])[0]
        qvec = np.asarray(emb, dtype=np.float32)
        qn = np.linalg.norm(qvec)
        if qn > 0:
            qvec = qvec / qn
        sims = vectors @ qvec
        # top-k_max stable
        top_idx = np.argpartition(-sims, k_max - 1)[:k_max]
        top_idx = top_idx[np.argsort(-sims[top_idx], kind="stable")]
        r_hit = float("inf")
        r_ph = float("inf")
        for rank, gi in enumerate(top_idx, start=1):
            chunk_text = chunks[int(gi)]["text"]
            best = 0.0
            for snip in gold:
                s = _score_chunk_against_snippet(snip, chunk_text)
                if s > best:
                    best = s
                if best >= HIT_THRESHOLD:
                    break
            if best >= HIT_THRESHOLD and r_hit == float("inf"):
                r_hit = rank
            if best >= PARTIAL_THRESHOLD and r_ph == float("inf"):
                r_ph = rank
            if r_hit != float("inf") and r_ph != float("inf"):
                break
        per_row.append({
            "row_id": r.get("id", ""),
            "company": r.get("company", ""),
            "reachable": reach["per_row"][ri]["reachable"],
            "r_hit": r_hit, "r_ph": r_ph,
        })
        # histogram (only for reachable rows)
        if reach["per_row"][ri]["reachable"]:
            placed = False
            for (lo, hi) in HIST_BUCKETS:
                if lo <= r_ph <= hi:
                    bkey = f"{lo}-{hi if hi < 10**9 else 'inf'}"
                    rank_hist_ph[bkey] += 1
                    placed = True
                    break
            if not placed:
                rank_hist_ph[f"{HIST_BUCKETS[-1][0]}-inf"] += 1
        if (ri + 1) % 4 == 0 or (ri + 1) == len(rows):
            print(f"[Stage 3a] row {ri+1:>2}/{len(rows)}  r_ph={r_ph}  r_hit={r_hit}")

    # Recall@K over reachable rows only
    reachable_only = [rec for rec in per_row if rec["reachable"]]
    n_reach = len(reachable_only)
    recall_at_k = {}
    for K in K_SET:
        rh = sum(1 for rec in reachable_only if rec["r_hit"] <= K) / max(n_reach, 1)
        rph = sum(1 for rec in reachable_only if rec["r_ph"] <= K) / max(n_reach, 1)
        recall_at_k[K] = {"recall_hit": rh, "recall_ph": rph, "n_reach": n_reach}
        print(f"[Stage 3a] K={K:>4}  recall_hit={rh:.1%}  recall_ph={rph:.1%}  (n_reach={n_reach})")

    return {"per_row": per_row, "recall_at_k": recall_at_k,
            "rank_hist_ph": rank_hist_ph, "embedding_model": embed_model}


# =============================== STAGE 3b: BATCHED CE ===============================
def _narrative_pool(client: AlbertClient, embed_model: str, query: str,
                    inputs: dict[str, Any], k: int,
                    cache: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    if query in cache:
        qvec = cache[query]
    else:
        emb = client.create_embeddings(embed_model, [query])[0]
        qvec = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(qvec)
        if n > 0:
            qvec = qvec / n
        cache[query] = qvec
    nidx = inputs["narrative_idx"]
    nvecs = inputs["narrative_vecs"]
    sims = nvecs @ qvec
    order = np.argpartition(-sims, k - 1)[:k]
    order = order[np.argsort(-sims[order], kind="stable")]
    out: list[dict[str, Any]] = []
    chunks = inputs["chunks"]
    for pos, local_i in enumerate(order):
        gi = int(nidx[int(local_i)])
        out.append({
            "chunk_id": chunks[gi]["chunk_id"],
            "text": chunks[gi]["text"],
            "score": float(sims[int(local_i)]),
            "pool_index": pos,
        })
    return out


def batched_ce(inputs: dict[str, Any]) -> dict[str, Any]:
    """Build dense_pool + narrative_pool at CE_POOL_K, batched-rerank both with adaptive sizes."""
    client = AlbertClient(api_key=os.environ["ALBERT_API_KEY"], base_url=ALBERT_BASE)
    embed_model = client.get_embedding_model(preferred="bge-m3")
    rows = inputs["selected_rows"]
    idx_dir = Path(CONFIG["INDEX_DIR"])
    cache: dict[str, np.ndarray] = {}

    per_row_dense_ce: list[dict[str, Any]] = []
    per_row_narr_ce: list[dict[str, Any]] = []
    dense_pools: list[list[dict[str, Any]]] = []
    narr_pools: list[list[dict[str, Any]]] = []

    for ri, r in enumerate(rows):
        q = r.get("question", "")
        company = r.get("company", "")
        # dense pool
        try:
            dp, _ = retrieve_chunks(
                index_dir=idx_dir, question=q,
                top_k=CONFIG["CE_POOL_K"],
                retrieval_architecture="dense",
                search_breadth=CONFIG["CE_POOL_K"],
                base_url=ALBERT_BASE,
            )
        except Exception as exc:
            print(f"[Stage 3b] row {ri+1} dense_pool ERROR: {type(exc).__name__}: {exc}")
            dp = []
        dense_pool = [{"chunk_id": c["chunk_id"], "text": c["text"],
                       "score": float(c.get("score", 0.0)), "pool_index": i}
                      for i, c in enumerate(dp)]
        dense_pools.append(dense_pool)

        # narrative pool
        try:
            np_pool = _narrative_pool(client, embed_model, q, inputs,
                                      CONFIG["CE_POOL_K"], cache)
        except Exception as exc:
            print(f"[Stage 3b] row {ri+1} narrative_pool ERROR: {type(exc).__name__}: {exc}")
            np_pool = []
        narr_pools.append(np_pool)

        # batched CE
        dense_ce_res = batched_rerank_adaptive(client, q, dense_pool) if dense_pool else {
            "top_k_pool": [], "batch_size_used": None,
            "truncation_applied": False, "ce_failed": True, "attempts": []}
        narr_ce_res = batched_rerank_adaptive(client, q, np_pool) if np_pool else {
            "top_k_pool": [], "batch_size_used": None,
            "truncation_applied": False, "ce_failed": True, "attempts": []}
        per_row_dense_ce.append({"row_id": r.get("id", ""), "company": company, **dense_ce_res})
        per_row_narr_ce.append({"row_id": r.get("id", ""), "company": company, **narr_ce_res})

        print(f"[Stage 3b] row {ri+1:>2}/{len(rows)} "
              f"dense_ce(size={dense_ce_res['batch_size_used']}, "
              f"trunc={dense_ce_res['truncation_applied']}, fail={dense_ce_res['ce_failed']}) "
              f"narr_ce(size={narr_ce_res['batch_size_used']}, "
              f"trunc={narr_ce_res['truncation_applied']}, fail={narr_ce_res['ce_failed']})")

    return {
        "per_row_dense_ce": per_row_dense_ce,
        "per_row_narr_ce": per_row_narr_ce,
        "dense_pools": dense_pools,
        "narr_pools": narr_pools,
    }


# =============================== STAGE 4: SCORE CE ===============================
def score_ce(inputs: dict[str, Any], ce: dict[str, Any], reach: dict[str, Any]) -> dict[str, Any]:
    """For each strategy: base top-8, pool@100 ceiling, CE top-8 — score against gold."""
    rows = inputs["selected_rows"]
    out: dict[str, list[dict[str, Any]]] = {"dense": [], "dense_ce": [],
                                            "narrative": [], "narrative_ce": []}
    for ri, r in enumerate(rows):
        gold = [g for g in _parse_expected_contexts(r) if isinstance(g, str) and g.strip()]
        dpool = ce["dense_pools"][ri]
        npool = ce["narr_pools"][ri]
        # base top-8 + pool@100 ceiling
        dense_top8 = _score_retrieval(gold, dpool[:CONFIG["TOP_K"]]) if dpool else \
            {"status": "miss", "best_match_score": 0.0, "best_chunk_id": None}
        dense_pool100 = _score_retrieval(gold, dpool) if dpool else dense_top8
        narr_top8 = _score_retrieval(gold, npool[:CONFIG["TOP_K"]]) if npool else \
            {"status": "miss", "best_match_score": 0.0, "best_chunk_id": None}
        narr_pool100 = _score_retrieval(gold, npool) if npool else narr_top8

        # CE top-8 scoring
        dce_top = ce["per_row_dense_ce"][ri]["top_k_pool"]
        dce_score = _score_retrieval(gold, dce_top) if dce_top else \
            {"status": "miss", "best_match_score": 0.0, "best_chunk_id": None}
        nce_top = ce["per_row_narr_ce"][ri]["top_k_pool"]
        nce_score = _score_retrieval(gold, nce_top) if nce_top else \
            {"status": "miss", "best_match_score": 0.0, "best_chunk_id": None}

        reachable = reach["per_row"][ri]["reachable"]
        out["dense"].append({"row_id": r.get("id", ""), "reachable": reachable,
                             "top8": dense_top8, "pool100": dense_pool100})
        out["dense_ce"].append({"row_id": r.get("id", ""), "reachable": reachable,
                                "topk": dce_score,
                                "ce_failed": ce["per_row_dense_ce"][ri]["ce_failed"],
                                "truncation_applied": ce["per_row_dense_ce"][ri]["truncation_applied"]})
        out["narrative"].append({"row_id": r.get("id", ""), "reachable": reachable,
                                 "top8": narr_top8, "pool100": narr_pool100})
        out["narrative_ce"].append({"row_id": r.get("id", ""), "reachable": reachable,
                                    "topk": nce_score,
                                    "ce_failed": ce["per_row_narr_ce"][ri]["ce_failed"],
                                    "truncation_applied": ce["per_row_narr_ce"][ri]["truncation_applied"]})
    return out


# =============================== STAGE 5: REPORT ===============================
def _agg_status_rate(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    if not rows:
        return {"n": 0, "hit_rate": 0.0, "ph_rate": 0.0}
    statuses = [r[key]["status"] for r in rows]
    n = len(statuses)
    hit = sum(1 for s in statuses if s == "hit")
    ph = sum(1 for s in statuses if s in ("hit", "partial"))
    return {"n": n, "hit_rate": hit / n, "ph_rate": ph / n}


def aggregate_and_report(provenance: dict[str, Any], inputs: dict[str, Any],
                         reach: dict[str, Any], sweep: dict[str, Any],
                         ce: dict[str, Any], scored: dict[str, Any]) -> dict[str, Any]:
    csv_path = Path("retrieval_sweep_rerank.csv")
    md_path = Path("retrieval_sweep_rerank.md")

    # --- aggregates ---
    n_reach = reach["total_reachable"]

    # base + ceiling + CE rates (over all rows; reachability is 100% in this corpus, so no separate gate)
    dense_base = _agg_status_rate(scored["dense"], "top8")
    dense_ceiling = _agg_status_rate(scored["dense"], "pool100")
    dense_ce_rate = _agg_status_rate(scored["dense_ce"], "topk")
    narr_base = _agg_status_rate(scored["narrative"], "top8")
    narr_ceiling = _agg_status_rate(scored["narrative"], "pool100")
    narr_ce_rate = _agg_status_rate(scored["narrative_ce"], "topk")

    # batch size used + truncated/failed tallies
    def _batch_summary(per_row: list[dict[str, Any]]) -> dict[str, Any]:
        sizes = [r.get("batch_size_used") for r in per_row]
        non_none = [s for s in sizes if s is not None]
        # dominant final batch size
        from collections import Counter
        c = Counter(non_none)
        most = c.most_common(1)[0][0] if c else None
        return {
            "final_batch_size_modal": most,
            "size_distribution": dict(c),
            "truncated_n": sum(1 for r in per_row if r.get("truncation_applied")),
            "ce_failed_n": sum(1 for r in per_row if r.get("ce_failed")),
        }

    dense_ce_bs = _batch_summary(ce["per_row_dense_ce"])
    narr_ce_bs = _batch_summary(ce["per_row_narr_ce"])

    # --- CSV ---
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        # sweep table
        w.writerow(["section", "K", "recall_hit_pct", "recall_ph_pct", "n_reach"])
        for K in CONFIG["K_SWEEP"]:
            r = sweep["recall_at_k"][K]
            w.writerow(["sweep", K, f"{r['recall_hit']:.4f}", f"{r['recall_ph']:.4f}", r["n_reach"]])
        # CE realized table
        w.writerow([])
        w.writerow(["section", "strategy", "base_top8_ph_pct", "pool_at_100_ph_pct",
                    "ce_top8_ph_pct", "realized_gain_pp",
                    "final_batch_size", "truncated_n", "ce_failed_n"])
        for strat, base, ceiling, ce_rate, bs in (
            ("dense_ce", dense_base, dense_ceiling, dense_ce_rate, dense_ce_bs),
            ("narrative_ce", narr_base, narr_ceiling, narr_ce_rate, narr_ce_bs),
        ):
            gain_pp = 100.0 * (ce_rate["ph_rate"] - base["ph_rate"])
            w.writerow(["ce", strat,
                        f"{100*base['ph_rate']:.2f}", f"{100*ceiling['ph_rate']:.2f}",
                        f"{100*ce_rate['ph_rate']:.2f}", f"{gain_pp:+.2f}",
                        bs["final_batch_size_modal"], bs["truncated_n"], bs["ce_failed_n"]])
        # histogram
        w.writerow([])
        w.writerow(["section", "bucket", "count"])
        for k, v in sweep["rank_hist_ph"].items():
            w.writerow(["hist_r_ph_reachable", k, v])

    # --- Markdown ---
    lines: list[str] = []
    lines.append("# Retrieval Sweep + Batched Rerank")
    lines.append("")
    lines.append(f"- Index: `{CONFIG['INDEX_DIR']}` (chunks=145197, vectors=(145197,1024) float32)")
    lines.append(f"- Dataset: `{CONFIG['DATASET']}`  rows evaluated = {len(inputs['selected_rows'])} "
                 f"({CONFIG['ROWS_PER_COMPANY']}/company for 4 companies; TotalEnergies has only 7)")
    lines.append(f"- Reachable: **{n_reach}/{reach['total']}** "
                 f"({n_reach/max(reach['total'],1):.0%})  "
                 f"row-set-vs-prior: **{inputs['row_set_matches_prior']}**")
    lines.append(f"- Embedding model: `{sweep['embedding_model']}`  "
                 f"alignment: positional (len-verified + retrieved-ids-subset-checked, from prior run)")
    lines.append(f"- Rerank: model=`{CONFIG['RERANK_MODEL']}` "
                 f"endpoint=`{provenance['rerank_endpoint']}` probe_ok={provenance['rerank_probe_ok']}")
    lines.append("")

    lines.append("## Pool-recall sweep (raw full-corpus cosine; FREE)")
    lines.append("")
    lines.append("| K | recall_hit% (≥90 in top-K) | recall (h+p)% (≥70 in top-K) |")
    lines.append("|---|---|---|")
    for K in CONFIG["K_SWEEP"]:
        r = sweep["recall_at_k"][K]
        lines.append(f"| {K} | {r['recall_hit']:.1%} | {r['recall_ph']:.1%} |")
    lines.append("")

    lines.append("### Rank distribution of first partial-hit (reachable rows only)")
    lines.append("")
    lines.append("| bucket | count |")
    lines.append("|---|---|")
    for k, v in sweep["rank_hist_ph"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Cross-check with prior dense pool@100 (~29%)
    prior_pool_100_ph = 0.29
    sweep_100_ph = sweep["recall_at_k"][100]["recall_ph"]
    sweep_100_hit = sweep["recall_at_k"][100]["recall_hit"]
    cross_check_match = abs(sweep_100_ph - prior_pool_100_ph) < 0.05
    lines.append(f"### Cross-check vs prior report card's dense pool@100 (~29%)")
    lines.append("")
    lines.append(f"- This sweep recall_ph@100 = **{sweep_100_ph:.1%}**  "
                 f"(hit@100 = {sweep_100_hit:.1%})")
    lines.append(f"- Prior dense pool@100 (h+p) = ~29.0%")
    if cross_check_match:
        lines.append(f"- **MATCH** within tolerance. The app's dense path appears to be plain top-K "
                     "cosine (no dedup/cap/MMR observed).")
    else:
        lines.append(f"- **DIVERGENCE**: raw cosine recall_ph@100 differs from app dense pool@100 "
                     "by more than 5pp. Possible causes: the app applies dedup, source-path filtering, "
                     "MMR, or a chunk-kind/year filter. Worth investigating.")
    lines.append("")

    lines.append("## Realized rerank (batched cross-encoder over CE_POOL_K=100)")
    lines.append("")
    lines.append("| strategy | base top-8 (h+p)% | pool@100 ceiling (h+p)% | CE top-8 (h+p)% realized | gain (pp) | "
                 "modal batch | truncated | ce_failed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for strat, base, ceiling, ce_rate, bs in (
        ("dense_ce", dense_base, dense_ceiling, dense_ce_rate, dense_ce_bs),
        ("narrative_ce", narr_base, narr_ceiling, narr_ce_rate, narr_ce_bs),
    ):
        gain_pp = 100.0 * (ce_rate["ph_rate"] - base["ph_rate"])
        lines.append(f"| {strat} | {base['ph_rate']:.1%} | {ceiling['ph_rate']:.1%} | "
                     f"{ce_rate['ph_rate']:.1%} | {gain_pp:+.2f} | "
                     f"{bs['final_batch_size_modal']} | {bs['truncated_n']} | {bs['ce_failed_n']} |")
    lines.append("")

    # Verdicts
    lines.append("## Verdicts")
    lines.append("")
    # (1) does recall climb with K?
    r100 = sweep["recall_at_k"][100]["recall_ph"]
    r1000 = sweep["recall_at_k"][1000]["recall_ph"]
    climb = r1000 - r100
    if climb >= 0.10:
        lines.append(f"1. **Recall climbs with K** (+{climb:.1%} from K=100 to K=1000) — "
                     "gold is often just below the top-100 cutoff. A wider candidate pool helps.")
    elif climb >= 0.03:
        lines.append(f"1. **Recall climbs slightly** (+{climb:.1%} from K=100 to K=1000) — "
                     "modest payoff from widening the pool.")
    else:
        lines.append(f"1. **Recall is essentially flat** (+{climb:.1%} from K=100 to K=1000) — "
                     "the bge-m3 embedding can't reach the gold for most rows; "
                     "this is an EMBEDDING ceiling, not a retrieval cutoff problem.")
    # (2) CE conversion
    headroom_dense_pp = 100 * (dense_ceiling["ph_rate"] - dense_base["ph_rate"])
    gain_dense_pp = 100 * (dense_ce_rate["ph_rate"] - dense_base["ph_rate"])
    headroom_narr_pp = 100 * (narr_ceiling["ph_rate"] - narr_base["ph_rate"])
    gain_narr_pp = 100 * (narr_ce_rate["ph_rate"] - narr_base["ph_rate"])
    if dense_ce_bs["ce_failed_n"] == 0:
        if gain_dense_pp >= 5:
            lines.append(f"2. **Dense cross-encoder converts headroom** "
                         f"({gain_dense_pp:+.1f}pp of {headroom_dense_pp:.1f}pp available). "
                         "The reranker is the right next lever.")
        else:
            lines.append(f"2. **Dense cross-encoder barely moves the needle** "
                         f"({gain_dense_pp:+.1f}pp of {headroom_dense_pp:.1f}pp available). "
                         "Either the headroom isn't real (the partial matches in the pool aren't truly relevant) "
                         "or the reranker doesn't help on table-heavy chunks.")
    else:
        lines.append(f"2. **Dense CE: {dense_ce_bs['ce_failed_n']} rows still failed after adaptation.** "
                     "Report numbers cover the surviving rows only.")
    lines.append("")

    lines.append(f"_Manifest: retrieval_sweep_rerank.run_manifest.json_")
    md = "\n".join(lines)
    md_path.write_text(md, encoding="utf-8")
    print()
    print(md)

    return {
        "dense_base": dense_base, "dense_ceiling": dense_ceiling, "dense_ce_rate": dense_ce_rate,
        "narr_base": narr_base, "narr_ceiling": narr_ceiling, "narr_ce_rate": narr_ce_rate,
        "dense_ce_bs": dense_ce_bs, "narr_ce_bs": narr_ce_bs,
        "sweep_100_ph": sweep_100_ph, "sweep_100_hit": sweep_100_hit,
        "cross_check_match": cross_check_match,
    }


# =============================== STAGE 6: MANIFEST ===============================
def write_manifest(provenance: dict[str, Any], inputs: dict[str, Any],
                   reach: dict[str, Any], sweep: dict[str, Any],
                   ce: dict[str, Any], aggregates: dict[str, Any],
                   prior_match: bool | None) -> None:
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
            "manifest_parsed_chunk_count": inputs["manifest"].get("chunk_count"),
            "manifest_parsed_embedding_model": inputs["manifest"].get("embedding_model"),
            "vectors_shape": inputs["vectors_shape"],
            "vectors_dtype": inputs["vectors_dtype"],
            "vectors_bytes": inputs["vectors_bytes"],
            "chunk_ids_file_present": False,
            "chunk_ids_source": "chunks.json[i].chunk_id",
            "alignment": "positional (len-verified + retrieved-ids-subset-checked, prior run)",
            "dataset_path": inputs["dataset_path"],
            "dataset_sha256": inputs["dataset_sha256"],
            "dataset_row_count": inputs["dataset_row_count"],
        },
        "rerank_endpoint": {
            "endpoint": provenance["rerank_endpoint"],
            "request_shape": provenance["rerank_request_shape"],
            "response_shape": provenance["rerank_response_shape"],
            "probe_ok": provenance["rerank_probe_ok"],
            "probe_note": provenance["rerank_probe_note"],
            "rerank_model": CONFIG["RERANK_MODEL"],
        },
        "selection": {
            "companies": CONFIG["COMPANIES"],
            "rows_per_company": CONFIG["ROWS_PER_COMPANY"],
            "selected_row_ids": inputs["selected_row_ids"],
            "row_set_matches_prior": prior_match,
            "top_k": CONFIG["TOP_K"],
            "ce_pool_k": CONFIG["CE_POOL_K"],
            "k_sweep": CONFIG["K_SWEEP"],
        },
        "reachability": {
            "total_reachable": reach["total_reachable"],
            "total": reach["total"],
            "per_company": reach["per_company"],
        },
        "sweep_results": {
            "recall_at_k": {str(K): sweep["recall_at_k"][K] for K in CONFIG["K_SWEEP"]},
            "rank_hist_ph": sweep["rank_hist_ph"],
            "cross_check_vs_prior_dense_pool_at_100": aggregates["cross_check_match"],
            "sweep_100_ph": aggregates["sweep_100_ph"],
        },
        "rerank_results": {
            "dense_ce": {
                "base_top8_ph": aggregates["dense_base"]["ph_rate"],
                "pool100_ph_ceiling": aggregates["dense_ceiling"]["ph_rate"],
                "ce_top8_ph": aggregates["dense_ce_rate"]["ph_rate"],
                "realized_gain_pp": 100.0 * (aggregates["dense_ce_rate"]["ph_rate"]
                                              - aggregates["dense_base"]["ph_rate"]),
                "batch_summary": aggregates["dense_ce_bs"],
            },
            "narrative_ce": {
                "base_top8_ph": aggregates["narr_base"]["ph_rate"],
                "pool100_ph_ceiling": aggregates["narr_ceiling"]["ph_rate"],
                "ce_top8_ph": aggregates["narr_ce_rate"]["ph_rate"],
                "realized_gain_pp": 100.0 * (aggregates["narr_ce_rate"]["ph_rate"]
                                              - aggregates["narr_base"]["ph_rate"]),
                "batch_summary": aggregates["narr_ce_bs"],
            },
        },
        "determinism_notes": {
            "sweep": "deterministic by construction (np.argpartition + stable argsort over a "
                     "fixed L2-normalized vector matrix; 1 embed per query)",
            "ce": "deterministic by construction (bge cross-encoder forward pass; "
                  "service-dependent; batched globally-sortable scores)",
            "non_deterministic_strategies": "none in this script (hyde/mqr not invoked)",
        },
        "budget": BUDGET,
        "read_only_git_commands_used": READ_ONLY_GIT_COMMANDS_USED,
    }
    Path("retrieval_sweep_rerank.run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================== MAIN ===============================
BANNER = """\
====================================================================
PIPELINE FLOW
  Stage 0  capture_provenance()    -> manifest fields (code + rerank endpoint + cross-check)
  Stage 1  load_inputs()           -> dataset + index + alignment + row-set assert
  Stage 2  reachability()          -> per-company reachable counts (reuse logic)
  Stage 3a pool_recall_sweep()     -> raw cosine; rank-of-first-hit; recall@K (FREE)
  Stage 3b batched_ce()            -> dense_ce + narrative_ce @ CE_POOL_K, adaptive batching
  Stage 4  score()                 -> CE top-8 hit/partial
  Stage 5  aggregate_and_report()  -> retrieval_sweep_rerank.md + .csv
  Stage 6  write_manifest()        -> retrieval_sweep_rerank.run_manifest.json
===================================================================="""


def main() -> int:
    t0 = time.time()
    print(BANNER)
    print()

    # Stage 0
    print("[Stage 0] capturing provenance + rerank probe")
    provenance = capture_provenance()
    print(f"[Stage 0] rerank probe ok={provenance['rerank_probe_ok']}  "
          f"note={provenance['rerank_probe_note']}")

    # Load prior manifest for row-set cross-check (only if configured + present)
    prior_ids: list[str] | None = None
    prior_manifest_path = Path(CONFIG["PRIOR_MANIFEST"]) if CONFIG.get("PRIOR_MANIFEST") else None
    if prior_manifest_path is not None and prior_manifest_path.is_file():
        try:
            prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
            prior_ids = prior.get("selection", {}).get("selected_row_ids")
            if prior_ids is None:
                print("[Stage 0] prior manifest present but no selected_row_ids field")
        except Exception as exc:
            print(f"[Stage 0] could not read prior manifest: {type(exc).__name__}: {exc}")
    else:
        print("[Stage 0] PRIOR_MANIFEST is None or missing — no row-set cross-check")

    # Stage 1
    print()
    print("[Stage 1] loading inputs + row-set cross-check")
    inputs = load_inputs(prior_ids)

    # Stage 2
    print()
    print("[Stage 2] reachability re-check")
    reach = reachability(inputs)

    # Stage 3a
    print()
    print("[Stage 3a] pool-recall sweep (raw full-corpus cosine)")
    sweep = pool_recall_sweep(inputs, reach)

    # Stage 3b
    print()
    print("[Stage 3b] batched CE rerank (adaptive batch sizes)")
    ce = batched_ce(inputs)

    # Stage 4
    print()
    print("[Stage 4] scoring CE top-8 + base/ceiling")
    scored = score_ce(inputs, ce, reach)

    # Stage 5
    print()
    print("[Stage 5] aggregating + writing report")
    aggregates = aggregate_and_report(provenance, inputs, reach, sweep, ce, scored)

    # Stage 6
    print()
    print("[Stage 6] writing manifest")
    write_manifest(provenance, inputs, reach, sweep, ce, aggregates,
                   inputs["row_set_matches_prior"])

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
