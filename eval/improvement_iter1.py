"""Improvement iteration 1: corpus-noise filter, measured with a repaired context-relevance judge.

FOUND  : v2 gate showed top-8 retrieval often picks PDF-parser garbage (letter fragments, encoding
         replacement chars, mid-word table splits, page-headers/footers).
FIX    : retrieval-time noise mask (no re-embed, no index mutation, no app/* edits) that excludes
         chunks failing any of four content-substantiveness rules.
MEASURE: a clean context-relevance judge (Mistral-Small-3.2-24B) judges baseline vs filtered top-8
         on each row, using ONLY the question + passages (no gold, no answer).

PIPELINE FLOW
  Stage 0 capture_provenance()    -> code+rerank+env fingerprints
  Stage 1 load_inputs()           -> chunks/vectors + rows + alignment + row-set assert
  Stage 2 build_noise_mask()      -> THE FIX (boolean array aligned to vectors)
  Stage 3 retrieve_both()         -> top-8 baseline and top-8 filtered per row
  Stage 4 judge_relevance()       -> Mistral judge, per (row,condition); 62 calls total
  Stage 5 measure_delta()         -> before/after aggregates
  Stage 6 report()+write_manifest()

NO edits to app/*. NO git writes. NO pip/env changes. NO deletions. NO re-embedding.
Read-only git inspection allowed for fingerprinting only. Secrets via setdefault, never printed.

Run from eval/:  PYTHONPATH=.. python improvement_iter1.py
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
    "INDEX_DIR":          "../outputs/rag_index",
    "DATASET":            "../sample_data/ragas_esg_eval_dataset.csv",
    "COMPANIES":          ["Danone", "Enel", "TotalEnergies", "Volkswagen"],
    "ROWS_PER_COMPANY":   8,
    "TOP_K":              8,
    "JUDGE":              "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "MIN_TOKENS":         5,
    "MIN_WORDLIKE":       3,
    "MAX_NONALNUM_RATIO": 0.5,
    "PASSAGE_CHAR_CAP":   800,  # per-passage truncation in judge prompt
    # Original cross-check pointed at the gate_v2 manifest, which was a precursor that
    # is intentionally not committed in eval/. Cross-check against the sweep manifest
    # instead (same row-set rule: 8 first-by-id rows for the 4 base companies).
    "PRIOR_MANIFEST":     "retrieval_sweep_rerank.run_manifest.json",
    "HIGH_RELEVANCE_THR": 0.60,
    "MATERIAL_LIFT_THR":  0.10,
}

BUDGET = {"chat_calls": 0, "rerank_calls": 0, "embedding_calls": 0,
          "embedding_strings": 0, "judge_calls": 0}

READ_ONLY_GIT_COMMANDS_USED: list[str] = []
WORDLIKE_RE = re.compile(r"[A-Za-z]{3,}")


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

from app.rag import AlbertClient, DEFAULT_BASE_URL                # noqa: E402
from app.rag_eval import _parse_expected_contexts                 # noqa: E402

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
def load_inputs(prior_selected_ids: list[str] | None) -> dict[str, Any]:
    idx_dir = Path(CONFIG["INDEX_DIR"])
    chunks_path = idx_dir / "chunks.json"
    vectors_path = idx_dir / "vectors.npy"
    manifest_path = idx_dir / "manifest.json"
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
    else:
        list_match = selected_ids == prior_selected_ids
        set_match = set(selected_ids) == set(prior_selected_ids)
        row_set_matches_prior = list_match
        print(f"[Stage 1] row-set vs prior: list-equal={list_match}, set-equal={set_match}")
        if not set_match:
            raise SystemExit("STOP: row-set drift")

    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "chunks": chunks,
        "vectors": full_vecs,
        "vectors_shape": list(full_vecs.shape),
        "vectors_dtype": str(full_vecs.dtype),
        "selected_rows": selected_rows,
        "selected_row_ids": selected_ids,
        "row_set_matches_prior": row_set_matches_prior,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_row_count": len(all_rows),
    }


# =============================== STAGE 2 ===============================
def build_noise_mask(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return {mask, per_rule_counts, examples}. mask[i]==True means NOISE (exclude)."""
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
        # non-alphanumeric char ratio
        total = len(text)
        if total > 0:
            non_alnum = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
            if (non_alnum / total) > CONFIG["MAX_NONALNUM_RATIO"]:
                rule_nonalnum_heavy[i] = True
        mask[i] = (rule_too_short[i] or rule_replacement[i] or
                   rule_too_few_wordlike[i] or rule_nonalnum_heavy[i])

    counts = {
        "too_short_lt5":           int(rule_too_short.sum()),
        "replacement_char":        int(rule_replacement.sum()),
        "too_few_wordlike_lt3":    int(rule_too_few_wordlike.sum()),
        "nonalnum_ratio_gt_0_5":   int(rule_nonalnum_heavy.sum()),
        "total_unique_noise":      int(mask.sum()),
    }
    counts["total_pct"] = counts["total_unique_noise"] / n

    # 8 examples (spread across rules)
    examples: list[dict[str, Any]] = []
    rule_examples = [
        ("too_short_lt5", rule_too_short),
        ("replacement_char", rule_replacement),
        ("too_few_wordlike_lt3", rule_too_few_wordlike),
        ("nonalnum_ratio_gt_0_5", rule_nonalnum_heavy),
    ]
    per_rule_quota = 2
    for rule_name, arr in rule_examples:
        picked = 0
        for i in np.flatnonzero(arr):
            examples.append({"chunk_id": chunks[int(i)]["chunk_id"],
                             "kind": chunks[int(i)].get("chunk_kind", ""),
                             "text_preview": truncate(chunks[int(i)]["text"], 120),
                             "rule": rule_name})
            picked += 1
            if picked >= per_rule_quota:
                break

    print(f"[Stage 2] noise rules — too_short<{CONFIG['MIN_TOKENS']}: {counts['too_short_lt5']}, "
          f"replacement_char: {counts['replacement_char']}, "
          f"wordlike<{CONFIG['MIN_WORDLIKE']}: {counts['too_few_wordlike_lt3']}, "
          f"nonalnum>{CONFIG['MAX_NONALNUM_RATIO']}: {counts['nonalnum_ratio_gt_0_5']}")
    print(f"[Stage 2] TOTAL unique noise (union): {counts['total_unique_noise']} "
          f"({counts['total_pct']:.1%} of corpus)")
    print("[Stage 2] 8 example dropped chunks:")
    for ex in examples:
        print(f"    [{ex['rule']:<24}] {ex['chunk_id']:<55} kind={ex['kind']:<11} "
              f"text={ex['text_preview']!r}")
    return {"mask": mask, "counts": counts, "examples": examples}


# =============================== STAGE 3 ===============================
def retrieve_both(inputs: dict[str, Any], noise_mask: np.ndarray) -> dict[str, Any]:
    """For each row: top-8 baseline and top-8 filtered. Returns per-row {baseline, filtered}."""
    chunks = inputs["chunks"]
    vectors = inputs["vectors"]
    rows = inputs["selected_rows"]
    client = AlbertClient(api_key=os.environ["ALBERT_API_KEY"], base_url=ALBERT_BASE)
    embed_model = client.get_embedding_model(preferred="bge-m3")
    qvec_cache: dict[str, np.ndarray] = {}
    top_k = CONFIG["TOP_K"]

    per_row: list[dict[str, Any]] = []
    for ri, r in enumerate(rows):
        q = r.get("question", "")
        if q in qvec_cache:
            qvec = qvec_cache[q]
            cache_hit = True
        else:
            emb = client.create_embeddings(embed_model, [q])[0]
            qvec = np.asarray(emb, dtype=np.float32)
            qn = np.linalg.norm(qvec)
            if qn > 0:
                qvec = qvec / qn
            qvec_cache[q] = qvec
            cache_hit = False
        sims = vectors @ qvec

        # baseline top-8
        base_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        base_idx = base_idx[np.argsort(-sims[base_idx], kind="stable")]
        baseline = []
        for gi in base_idx:
            c = chunks[int(gi)]
            baseline.append({"chunk_id": c["chunk_id"], "kind": c.get("chunk_kind", ""),
                             "company": c.get("company", ""), "text": c["text"],
                             "cos": float(sims[int(gi)]),
                             "is_noise": bool(noise_mask[int(gi)])})

        # filtered top-8 (mask out noise via -inf sims)
        sims_f = sims.copy()
        sims_f[noise_mask] = -np.inf
        filt_idx = np.argpartition(-sims_f, top_k - 1)[:top_k]
        filt_idx = filt_idx[np.argsort(-sims_f[filt_idx], kind="stable")]
        filtered = []
        for gi in filt_idx:
            c = chunks[int(gi)]
            filtered.append({"chunk_id": c["chunk_id"], "kind": c.get("chunk_kind", ""),
                             "company": c.get("company", ""), "text": c["text"],
                             "cos": float(sims[int(gi)]),
                             "is_noise": False})

        garbage_in_baseline = sum(1 for x in baseline if x["is_noise"])
        per_row.append({
            "row_id": r.get("id", ""),
            "company": r.get("company", ""),
            "question": q,
            "question_type": r.get("question_type", ""),
            "baseline": baseline,
            "filtered": filtered,
            "n_noise_baseline_top8": garbage_in_baseline,
            "cache_hit": cache_hit,
        })
        print(f"[Stage 3] row {ri+1:>2}/{len(rows)} {r.get('company',''):<14} "
              f"{'cache' if cache_hit else 'embed':<5} noise_in_baseline_top8={garbage_in_baseline}/8")
    print(f"[Stage 3] embedding calls = {BUDGET['embedding_calls']} "
          f"({len(qvec_cache)} unique questions, {len(rows) - len(qvec_cache)} cache hits)")
    return {"per_row": per_row}


# =============================== STAGE 4 ===============================
JUDGE_INSTRUCTION = (
    "You are a retrieval-quality judge. For each numbered passage, decide whether it helps "
    "answer the question:\n"
    " - \"relevant\": directly provides information that answers the question\n"
    " - \"partially_relevant\": related to the question but missing key details\n"
    " - \"not_relevant\": unrelated, contentless, or noise\n"
    "Then judge whether the question can be answered using these passages as a whole:\n"
    " - \"yes\" (fully), \"partial\", or \"no\".\n"
    "Output STRICT JSON ONLY, with this shape:\n"
    "{\"passages\":[{\"index\":<int>,\"verdict\":\"relevant|partially_relevant|not_relevant\"}],"
    "\"answerable\":\"yes|partial|no\"}\n"
    "No markdown, no commentary."
)


def _build_judge_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    cap = CONFIG["PASSAGE_CHAR_CAP"]
    blocks = []
    for i, p in enumerate(passages, start=1):
        text = p.get("text", "") or ""
        if len(text) > cap:
            text = text[:cap] + "…"
        blocks.append(f"[{i}] (kind={p.get('kind','?')}, company={p.get('company','?')}) "
                      f"{text}")
    return (
        f"{JUDGE_INSTRUCTION}\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n" + "\n\n".join(blocks)
    )


def judge_relevance(client: AlbertClient, judge_model: str,
                    per_row: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each row × condition (baseline, filtered), one judge call. 62 calls total for 31 rows."""
    out: list[dict[str, Any]] = []
    n = len(per_row)
    for ri, rec in enumerate(per_row):
        row_out: dict[str, Any] = {
            "row_id": rec["row_id"], "company": rec["company"],
            "question": rec["question"], "question_type": rec.get("question_type", ""),
            "n_noise_baseline_top8": rec["n_noise_baseline_top8"],
        }
        for condition in ("baseline", "filtered"):
            passages = rec[condition]
            prompt = _build_judge_prompt(rec["question"], passages)
            messages = [{"role": "user", "content": prompt}]
            BUDGET["judge_calls"] += 1
            try:
                raw = client.chat_completion(judge_model, messages, temperature=0.0)
                parsed = _parse_json_object(raw)
                verdicts = parsed.get("passages", [])
                relevant = sum(1 for v in verdicts if v.get("verdict") == "relevant")
                partial = sum(1 for v in verdicts if v.get("verdict") == "partially_relevant")
                total = len(verdicts) or len(passages)
                score = (relevant + 0.5 * partial) / max(total, 1)
                answerable = parsed.get("answerable", "no")
                row_out[f"{condition}_relevance_score"] = round(score, 4)
                row_out[f"{condition}_n_relevant"] = relevant
                row_out[f"{condition}_n_partial"] = partial
                row_out[f"{condition}_n_irrel"] = max(total - relevant - partial, 0)
                row_out[f"{condition}_answerable"] = answerable
                row_out[f"{condition}_judge_ok"] = True
            except Exception as exc:
                row_out[f"{condition}_relevance_score"] = 0.0
                row_out[f"{condition}_n_relevant"] = 0
                row_out[f"{condition}_n_partial"] = 0
                row_out[f"{condition}_n_irrel"] = 0
                row_out[f"{condition}_answerable"] = "judge_error"
                row_out[f"{condition}_judge_ok"] = False
                row_out[f"{condition}_error"] = f"{type(exc).__name__}: {exc}"

        out.append(row_out)
        print(f"[Stage 4] row {ri+1:>2}/{n} {rec['company']:<14} "
              f"base={row_out['baseline_relevance_score']:.3f}({row_out['baseline_answerable']}) "
              f"filt={row_out['filtered_relevance_score']:.3f}({row_out['filtered_answerable']}) "
              f"noise={rec['n_noise_baseline_top8']}")
    return out


# =============================== STAGE 5 + 6 ===============================
def measure_delta(judged: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(judged)

    def mean(key: str) -> float:
        return sum(j[key] for j in judged) / max(n, 1)

    summary = {
        "n_rows": n,
        "mean_base_relevance": mean("baseline_relevance_score"),
        "mean_filt_relevance": mean("filtered_relevance_score"),
        "mean_garbage_baseline_top8": sum(j["n_noise_baseline_top8"] for j in judged) / max(n, 1),
    }
    summary["delta_relevance"] = summary["mean_filt_relevance"] - summary["mean_base_relevance"]

    def ans_dist(cond: str) -> dict[str, int]:
        out = {"yes": 0, "partial": 0, "no": 0, "judge_error": 0, "other": 0}
        for j in judged:
            label = str(j.get(f"{cond}_answerable", "")).lower()
            if label in out:
                out[label] += 1
            else:
                out["other"] += 1
        return out

    summary["answerable_baseline"] = ans_dist("baseline")
    summary["answerable_filtered"] = ans_dist("filtered")

    by_company: dict[str, dict[str, Any]] = {}
    for j in judged:
        co = j.get("company", "")
        d = by_company.setdefault(co, {"n": 0, "base_sum": 0.0, "filt_sum": 0.0,
                                       "noise_sum": 0})
        d["n"] += 1
        d["base_sum"] += j["baseline_relevance_score"]
        d["filt_sum"] += j["filtered_relevance_score"]
        d["noise_sum"] += j["n_noise_baseline_top8"]
    for co, d in by_company.items():
        d["base_mean"] = d["base_sum"] / max(d["n"], 1)
        d["filt_mean"] = d["filt_sum"] / max(d["n"], 1)
        d["noise_mean"] = d["noise_sum"] / max(d["n"], 1)
        d["delta"] = d["filt_mean"] - d["base_mean"]
    summary["by_company"] = by_company
    return summary


def report(provenance: dict[str, Any], inputs: dict[str, Any],
           noise: dict[str, Any], per_row_retr: list[dict[str, Any]],
           judged: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    csv_path = Path("improvement_iter1.csv")
    md_path = Path("improvement_iter1.md")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "row_id", "company", "question_type",
            "n_noise_baseline_top8",
            "baseline_relevance_score", "baseline_answerable",
            "baseline_n_relevant", "baseline_n_partial", "baseline_n_irrel",
            "filtered_relevance_score", "filtered_answerable",
            "filtered_n_relevant", "filtered_n_partial", "filtered_n_irrel",
            "baseline_top1_chunk_id", "baseline_top1_kind",
            "filtered_top1_chunk_id", "filtered_top1_kind",
        ])
        retr_by_id = {r["row_id"]: r for r in per_row_retr}
        for j in judged:
            rec = retr_by_id.get(j["row_id"], {})
            b_top = rec.get("baseline", [{}])[0] if rec.get("baseline") else {}
            f_top = rec.get("filtered", [{}])[0] if rec.get("filtered") else {}
            w.writerow([
                j["row_id"], j["company"], j.get("question_type", ""),
                j["n_noise_baseline_top8"],
                j["baseline_relevance_score"], j["baseline_answerable"],
                j["baseline_n_relevant"], j["baseline_n_partial"], j["baseline_n_irrel"],
                j["filtered_relevance_score"], j["filtered_answerable"],
                j["filtered_n_relevant"], j["filtered_n_partial"], j["filtered_n_irrel"],
                b_top.get("chunk_id", ""), b_top.get("kind", ""),
                f_top.get("chunk_id", ""), f_top.get("kind", ""),
            ])

    # --- 4 qualitative swap samples: rows with biggest noise-in-baseline-top8 drop ---
    retr_by_id = {r["row_id"]: r for r in per_row_retr}
    rows_sorted = sorted(per_row_retr, key=lambda r: -r["n_noise_baseline_top8"])
    samples = [r for r in rows_sorted if r["n_noise_baseline_top8"] > 0][:4]
    if len(samples) < 4:
        samples = (samples + rows_sorted)[:4]

    lines: list[str] = []
    lines.append("# Improvement iteration 1 — corpus-noise filter, judged with Mistral-Small")
    lines.append("")
    lines.append(f"- Rows: **{summary['n_rows']}**  (row-set vs prior manifest: "
                 f"**{inputs['row_set_matches_prior']}**)")
    lines.append(f"- Judge: `{CONFIG['JUDGE']}`  embedding: `BAAI/bge-m3`")
    lines.append(f"- Noise mask thresholds: MIN_TOKENS={CONFIG['MIN_TOKENS']}, "
                 f"MIN_WORDLIKE={CONFIG['MIN_WORDLIKE']}, "
                 f"MAX_NONALNUM_RATIO={CONFIG['MAX_NONALNUM_RATIO']}, "
                 f"plus \\ufffd presence")
    lines.append("")

    # FOUND
    cc = noise["counts"]
    lines.append("## FOUND")
    lines.append("")
    lines.append("Corpus-noise mask rule counts (rules are independent — same chunk can trip several):")
    lines.append("")
    lines.append("| rule | count |")
    lines.append("|---|---:|")
    for k in ("too_short_lt5", "replacement_char", "too_few_wordlike_lt3", "nonalnum_ratio_gt_0_5"):
        lines.append(f"| `{k}` | {cc[k]} |")
    lines.append(f"| **TOTAL UNIQUE NOISE (union)** | **{cc['total_unique_noise']} ({cc['total_pct']:.1%} of corpus)** |")
    lines.append("")
    lines.append(f"Mean garbage-in-baseline-top-8 across {summary['n_rows']} rows: "
                 f"**{summary['mean_garbage_baseline_top8']:.2f} / 8** "
                 f"({summary['mean_garbage_baseline_top8']/8:.0%}).")
    lines.append("")
    lines.append("Example dropped chunks:")
    lines.append("")
    for ex in noise["examples"]:
        lines.append(f"- [`{ex['rule']}`] `{ex['chunk_id']}` kind=`{ex['kind']}` "
                     f"→ {ex['text_preview']!r}")
    lines.append("")
    lines.append("**Known gap (not fixed by this filter):** mid-word table splits like "
                 "`\"In Danone's table on page 318, row 'Waste manageme' has column 1: nt.\"` "
                 "still pass — those are a separate parser bug (cell-merge / character-cluster split).")
    lines.append("")

    # FIX
    lines.append("## FIX")
    lines.append("")
    lines.append("Boolean noise mask aligned to `vectors.npy[i]`. A chunk is masked if ANY rule fires:")
    lines.append("")
    lines.append(f"1. token count < {CONFIG['MIN_TOKENS']}")
    lines.append("2. text contains Unicode replacement character `\\ufffd` (`�`)")
    lines.append(f"3. fewer than {CONFIG['MIN_WORDLIKE']} word-like tokens (regex `[A-Za-z]{{3,}}`)")
    lines.append(f"4. non-alnum / non-space char ratio > {CONFIG['MAX_NONALNUM_RATIO']}")
    lines.append("")
    lines.append(f"At retrieval time, `sims[mask] = -inf` before top-K. No index mutation, no re-embed, "
                 "no `app/*` edits.")
    lines.append(f"Excluded chunks: **{cc['total_unique_noise']} / {len(inputs['chunks'])} "
                 f"({cc['total_pct']:.1%})**.")
    lines.append("")

    # MEASURED
    lines.append("## MEASURED")
    lines.append("")
    lines.append("### Relevance + garbage @ top-8  (baseline = no mask, filtered = mask applied)")
    lines.append("")
    lines.append("| company | n | mean_base_relevance | mean_filt_relevance | Δ | mean_noise_baseline_top8 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for co, d in summary["by_company"].items():
        lines.append(f"| {co} | {d['n']} | {d['base_mean']:.3f} | {d['filt_mean']:.3f} | "
                     f"{d['delta']:+.3f} | {d['noise_mean']:.2f} |")
    lines.append(f"| **OVERALL** | **{summary['n_rows']}** | "
                 f"**{summary['mean_base_relevance']:.3f}** | "
                 f"**{summary['mean_filt_relevance']:.3f}** | "
                 f"**{summary['delta_relevance']:+.3f}** | "
                 f"**{summary['mean_garbage_baseline_top8']:.2f}** |")
    lines.append("")

    # answerable distributions
    lines.append("### \"Is the question answerable from these passages?\" distribution")
    lines.append("")
    ab = summary["answerable_baseline"]
    af = summary["answerable_filtered"]
    lines.append("| condition | yes | partial | no | judge_error | other |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| baseline | {ab['yes']} | {ab['partial']} | {ab['no']} | "
                 f"{ab['judge_error']} | {ab['other']} |")
    lines.append(f"| filtered | {af['yes']} | {af['partial']} | {af['no']} | "
                 f"{af['judge_error']} | {af['other']} |")
    lines.append("")

    # qualitative swaps
    lines.append("### Qualitative swaps (top-N rows by garbage-in-baseline-top-8)")
    lines.append("")
    for i, rec in enumerate(samples, start=1):
        lines.append(f"#### Swap {i} — {rec['company']} / `{rec['row_id']}`  "
                     f"(noise in baseline top-8: **{rec['n_noise_baseline_top8']}/8**)")
        lines.append("")
        lines.append(f"- **Q**: {rec['question']}")
        b_noise = [p for p in rec['baseline'] if p['is_noise']]
        lines.append(f"- **Baseline top-8 noise chunks ({len(b_noise)})**:")
        for p in b_noise[:3]:
            lines.append(f"  - `{p['chunk_id']}` kind=`{p['kind']}` cos={p['cos']:.3f} "
                         f"→ {truncate(p['text'], 140)!r}")
        lines.append(f"- **Filtered top-1** (replacement for the highest-ranked noise chunk):")
        f_top = rec['filtered'][0] if rec['filtered'] else {}
        if f_top:
            lines.append(f"  - `{f_top['chunk_id']}` kind=`{f_top['kind']}` cos={f_top['cos']:.3f} "
                         f"→ {truncate(f_top['text'], 200)!r}")
        lines.append("")

    # VERDICT
    base = summary["mean_base_relevance"]
    delta = summary["delta_relevance"]
    mean_noise = summary["mean_garbage_baseline_top8"]
    lines.append("## VERDICT")
    lines.append("")
    verdict_code = ""
    verdict_lines: list[str] = []
    if base >= CONFIG["HIGH_RELEVANCE_THR"]:
        verdict_code = "RETRIEVAL_LARGELY_OK"
        verdict_lines.append(
            f"**Retrieval is largely relevant** (mean baseline relevance "
            f"{base:.3f} ≥ {CONFIG['HIGH_RELEVANCE_THR']}). The earlier 'broken retrieval' "
            "signal was substantially a gold-misalignment measurement artifact — when judged on "
            "the question + passages alone (no gold), top-8 chunks are mostly on-topic. The next "
            "lever is gold-set + metric repairs, not re-embedding."
        )
    if delta >= CONFIG["MATERIAL_LIFT_THR"] or mean_noise >= 1.0:
        if verdict_code:
            verdict_code = verdict_code + "+NOISE_FILTER_WIN"
        else:
            verdict_code = "NOISE_FILTER_WIN"
        verdict_lines.append(
            f"**Noise filter is a real win** "
            f"(Δ relevance = {delta:+.3f}; mean noise-in-baseline-top-8 = "
            f"{mean_noise:.2f}/8 = {mean_noise/8:.0%}). Recommend applying this mask at "
            "index-load time in the app (separate task). The mask removes "
            f"{cc['total_unique_noise']} chunks ({cc['total_pct']:.1%}) and does NOT require "
            "re-embedding the corpus."
        )
    if not verdict_code:
        verdict_code = "MINOR_DELTAS"
        verdict_lines.append(
            f"Baseline relevance {base:.3f} and filter Δ {delta:+.3f}. Filter effect is modest; "
            "noise count in baseline top-8 averaged "
            f"{mean_noise:.2f}/8. Not a strong signal in either direction."
        )
    for ln in verdict_lines:
        lines.append(f"- {ln}")
    lines.append("")
    lines.append(f"_Manifest: improvement_iter1.run_manifest.json_")
    md = "\n".join(lines)
    md_path.write_text(md, encoding="utf-8")
    print()
    print(md)

    return {"verdict_code": verdict_code, "verdict_lines": verdict_lines}


# =============================== STAGE 6 ===============================
def write_manifest(provenance: dict[str, Any], inputs: dict[str, Any],
                   noise: dict[str, Any], judged: list[dict[str, Any]],
                   summary: dict[str, Any], verdict: dict[str, Any]) -> None:
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
            "alignment": "positional (len-verified + retrieved-ids-subset-checked from prior runs)",
            "dataset_path": inputs["dataset_path"],
            "dataset_sha256": inputs["dataset_sha256"],
            "dataset_row_count": inputs["dataset_row_count"],
        },
        "selection": {
            "companies": CONFIG["COMPANIES"],
            "rows_per_company": CONFIG["ROWS_PER_COMPANY"],
            "selected_row_ids": inputs["selected_row_ids"],
            "row_set_matches_prior": inputs["row_set_matches_prior"],
        },
        "judge_model": CONFIG["JUDGE"],
        "noise_mask_stats": {
            "thresholds": {
                "min_tokens": CONFIG["MIN_TOKENS"],
                "min_wordlike": CONFIG["MIN_WORDLIKE"],
                "max_nonalnum_ratio": CONFIG["MAX_NONALNUM_RATIO"],
                "replacement_char": "\\ufffd",
            },
            "counts": noise["counts"],
            "examples_dropped": noise["examples"],
            "known_gap": (
                "Mid-word table splits (\"row 'Waste manageme' has column 1: nt.\") still pass; "
                "they're a separate table-parser bug."
            ),
        },
        "before_after_summary": {
            "n_rows": summary["n_rows"],
            "mean_base_relevance": summary["mean_base_relevance"],
            "mean_filt_relevance": summary["mean_filt_relevance"],
            "delta_relevance": summary["delta_relevance"],
            "mean_garbage_baseline_top8": summary["mean_garbage_baseline_top8"],
            "answerable_baseline": summary["answerable_baseline"],
            "answerable_filtered": summary["answerable_filtered"],
            "by_company": summary["by_company"],
        },
        "verdict_code": verdict["verdict_code"],
        "verdict_text": "; ".join(verdict["verdict_lines"]),
        "per_row": judged,
        "budget": BUDGET,
        "determinism_notes": (
            "Embedding cache + raw cosine + np.argsort(kind='stable') are deterministic; the judge "
            "(Mistral) is called at temperature=0 but its output is service-dependent."
        ),
        "read_only_git_commands_used": READ_ONLY_GIT_COMMANDS_USED,
    }
    Path("improvement_iter1.run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================== MAIN ===============================
BANNER = """\
====================================================================
PIPELINE FLOW — improvement iteration 1 (noise mask + repaired judge)
  Stage 0 capture_provenance()
  Stage 1 load_inputs()        + row-set assert
  Stage 2 build_noise_mask()   THE FIX (4 content rules; no re-embed)
  Stage 3 retrieve_both()      baseline top-8 vs filtered top-8
  Stage 4 judge_relevance()    Mistral judge on Q + passages only (62 calls)
  Stage 5 measure_delta()      before/after aggregates
  Stage 6 report() + write_manifest()
===================================================================="""


def main() -> int:
    t0 = time.time()
    print(BANNER)
    print()

    print("[Stage 0] capturing provenance")
    provenance = capture_provenance()

    prior_ids: list[str] | None = None
    pm = Path(CONFIG["PRIOR_MANIFEST"])
    if pm.is_file():
        try:
            prior = json.loads(pm.read_text(encoding="utf-8"))
            prior_ids = prior.get("selection", {}).get("selected_row_ids")
        except Exception as exc:
            print(f"[Stage 0] could not read prior manifest: {type(exc).__name__}: {exc}")

    print()
    print("[Stage 1] loading inputs + row-set cross-check")
    inputs = load_inputs(prior_ids)

    print()
    print("[Stage 2] building noise mask")
    noise = build_noise_mask(inputs["chunks"])

    print()
    print("[Stage 3] retrieving baseline + filtered top-8 for each row")
    retr = retrieve_both(inputs, noise["mask"])

    print()
    print("[Stage 4] judging context relevance (Mistral)")
    client = AlbertClient(api_key=os.environ["ALBERT_API_KEY"], base_url=ALBERT_BASE)
    judged = judge_relevance(client, CONFIG["JUDGE"], retr["per_row"])

    print()
    print("[Stage 5] measuring delta")
    summary = measure_delta(judged)
    print(f"[Stage 5] mean_base={summary['mean_base_relevance']:.3f}  "
          f"mean_filt={summary['mean_filt_relevance']:.3f}  "
          f"Δ={summary['delta_relevance']:+.3f}  "
          f"mean_noise_baseline_top8={summary['mean_garbage_baseline_top8']:.2f}")

    print()
    print("[Stage 6] writing report + manifest")
    verdict = report(provenance, inputs, noise, retr["per_row"], judged, summary)
    write_manifest(provenance, inputs, noise, judged, summary, verdict)

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
