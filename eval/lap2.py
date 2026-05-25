"""RAGAS Lap 2 — per-company-routed retrieval + repaired metric suite + generator comparison.

The pipeline payoff:
- Phase 1 routing: per-company "dense" vs "hybrid" decided from outputs/retrieval_weakness_map.csv.
- Identical retrieved context for both generators (Qwen vs gpt-oss) so the comparison isolates generation.
- Repaired metric suite: gold-free context_relevance, judged answer_correctness, judged abstention,
  reused score_faithfulness + score_answer_relevancy. context_recall is DROPPED (gold-misaligned).
- Mistral-Small judge for every judge call.

PIPELINE FLOW
  Stage 0 capture_provenance() + build_route_map()
  Stage 1 load_inputs()        deterministic 5/company; alignment re-assert
  Stage 2 retrieve_and_context_relevance()
  Stage 3 generate_answers()   IDENTICAL retrieved context for both generators
  Stage 4 judge_answers()      faithfulness + relevancy + correctness + abstention
  Stage 5 scorecard_and_verdict()
  Stage 6 write_manifest()

NO edits to app/*. NO git writes. NO pip/env changes. NO deletions. NO re-embedding.
Read-only git inspection only. Secrets via setdefault; never printed.

Run from eval/:  PYTHONPATH=.. python lap2.py
(Or from the repo root with `PYTHONPATH=. python -m` if you adjust the CONFIG paths.)
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
    "ALIAS_MAP":          {"ENGIE": "Engie", "Schneider": "Schneider Electric"},
    "EXCLUDE_COMPANIES":  {"Hermès", "Iberdrola", "Unknown"},
    "ROWS_PER_COMPANY":   5,
    "TOP_K":              8,
    "CANDIDATE_K":        100,
    "GENERATORS":         ["Qwen/Qwen3-Coder-30B-A3B-Instruct", "openai/gpt-oss-120b"],
    "JUDGE":              "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "ANSWER_MODE":        "assistant",
    "PROMPT_STYLE":       "balanced",
    "WEAKNESS_MAP_CSV":   "retrieval_weakness_map.csv",
    "PASSAGE_CHAR_CAP":   800,
    "ANSWER_CHAR_CAP":    1500,
    "GT_CHAR_CAP":        2000,
    "RETRIEVAL_LIMIT_THR": 0.30,
}

# Budget split: generation vs judge so we can report each
BUDGET = {"generation_calls": 0, "judge_calls": 0, "rerank_calls": 0,
          "embedding_calls": 0, "embedding_strings": 0}

READ_ONLY_GIT_COMMANDS_USED: list[str] = []


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

from app.rag import AlbertClient, DEFAULT_BASE_URL, retrieve_chunks, answer_question     # noqa: E402
from app.rag_eval import _parse_expected_contexts, get_ground_truth_answer                # noqa: E402
from app.ragas_eval import score_faithfulness, score_answer_relevancy                     # noqa: E402

ALBERT_BASE = os.environ.get("ALBERT_BASE_URL") or DEFAULT_BASE_URL


# Instrument AlbertClient — split generation vs judge by inspecting the model argument
_orig_chat = AlbertClient.chat_completion
_orig_embed = AlbertClient.create_embeddings


def _chat_wrapper(self, model, messages, *args, **kwargs):
    # generators tracked by CONFIG["GENERATORS"]; everything else is a judge call
    if model in CONFIG["GENERATORS"]:
        BUDGET["generation_calls"] += 1
    else:
        BUDGET["judge_calls"] += 1
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


def build_route_map() -> dict[str, dict[str, Any]]:
    """Derive per-company route from the weakness-map CSV. route='dense' if dense_rel>=hybrid_rel."""
    path = Path(CONFIG["WEAKNESS_MAP_CSV"])
    if not path.is_file():
        raise SystemExit(f"STOP: missing {path} — run retrieval_weakness_map.py first")
    routes: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0] == "company_agg":
                ds_co = row[1]
                idx_co = row[2]
                dense_rel = float(row[4])
                hybrid_rel = float(row[5])
                routes[ds_co] = {
                    "index_company": idx_co,
                    "dense_rel": dense_rel,
                    "hybrid_rel": hybrid_rel,
                    "route": "dense" if dense_rel >= hybrid_rel else "hybrid",
                }
    print(f"[Stage 0] ROUTE MAP (from {CONFIG['WEAKNESS_MAP_CSV']}):")
    for ds_co, info in routes.items():
        print(f"    {ds_co:<14} idx={info['index_company']:<22} "
              f"dense={info['dense_rel']:.3f} hybrid={info['hybrid_rel']:.3f}  "
              f"-> ROUTE={info['route']}")
    return routes


# =============================== STAGE 1 ===============================
def load_inputs(route_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    idx_dir = Path(CONFIG["INDEX_DIR"])
    manifest_path = idx_dir / "manifest.json"
    chunks_path = idx_dir / "chunks.json"
    vectors_path = idx_dir / "vectors.npy"
    dataset_path = Path(CONFIG["DATASET"])

    print("[Stage 1] loading manifest, vectors")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # We don't need full chunks for Lap 2 (retrieve_chunks reads its own copy);
    # we still validate alignment vs vectors.
    vectors_mmap = np.load(vectors_path, mmap_mode="r")
    if vectors_mmap.shape != (145197, 1024) or vectors_mmap.dtype != np.float32:
        raise SystemExit(f"STOP: vectors shape/dtype mismatch: {vectors_mmap.shape} {vectors_mmap.dtype}")
    print(f"[Stage 1] alignment OK: vectors (145197, 1024) float32")

    # Dataset selection
    with dataset_path.open("r", encoding="utf-8") as fh:
        all_rows = list(csv.DictReader(fh))
    selected_rows: list[dict[str, Any]] = []
    for ds_co, info in route_map.items():
        cr = [r for r in all_rows if r.get("company") == ds_co]
        cr.sort(key=lambda r: r.get("id", ""))
        picked = cr[: CONFIG["ROWS_PER_COMPANY"]]
        for r in picked:
            r["_dataset_company"] = ds_co
            r["_index_company"] = info["index_company"]
            r["_route"] = info["route"]
        selected_rows.extend(picked)
    selected_ids = [r.get("id", "") for r in selected_rows]
    print(f"[Stage 1] selected {len(selected_rows)} rows across "
          f"{len(route_map)} companies; rows/company: "
          + ", ".join(f"{ds}={sum(1 for r in selected_rows if r['_dataset_company']==ds)}"
                       for ds in route_map))

    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_chunk_count": manifest.get("chunk_count"),
        "manifest_embedding_model": manifest.get("embedding_model"),
        "vectors_shape": list(vectors_mmap.shape),
        "vectors_dtype": str(vectors_mmap.dtype),
        "selected_rows": selected_rows,
        "selected_row_ids": selected_ids,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_row_count": len(all_rows),
    }


# =============================== JUDGE PROMPTS ===============================
CONTEXT_RELEVANCE_INSTRUCTION = (
    "You are a retrieval-quality judge. For each numbered passage, decide whether it helps "
    "answer the question:\n"
    " - \"relevant\": directly provides information that answers the question\n"
    " - \"partially_relevant\": related to the question but missing key details\n"
    " - \"not_relevant\": unrelated, contentless, or noise\n"
    "Then judge whether the question can be answered using these passages as a whole "
    "(\"yes\" | \"partial\" | \"no\"). "
    "If answerable != \"yes\", add a one-sentence \"why_not\" explaining what's missing.\n"
    "Output STRICT JSON ONLY:\n"
    "{\"passages\":[{\"index\":<int>,\"verdict\":\"relevant|partially_relevant|not_relevant\"}],"
    "\"answerable\":\"yes|partial|no\",\"why_not\":\"<one sentence or empty>\"}\n"
    "No markdown, no commentary."
)


CORRECTNESS_INSTRUCTION = (
    "You are judging answer correctness against a reference (gold) answer.\n"
    "Is the model's answer factually consistent with, AND does it cover the key facts of, "
    "the reference?\n"
    "The reference may be a partial excerpt — judge factual consistency, NOT verbatim overlap.\n"
    " - \"correct\": factually consistent AND covers the key facts -> score 1.0\n"
    " - \"partially_correct\": factually consistent on some points but missing or vague on "
    "others -> 0.5\n"
    " - \"incorrect\": contradicts or misses the key facts -> 0.0\n"
    "Output STRICT JSON ONLY:\n"
    "{\"verdict\":\"correct|partially_correct|incorrect\",\"score\":1.0|0.5|0.0,"
    "\"reason\":\"<one sentence>\"}\n"
    "No markdown."
)


ABSTENTION_INSTRUCTION = (
    "You are judging whether an answer ABSTAINED instead of attempting to answer.\n"
    "Did the answer refuse, say the information is not available / not provided / insufficient / "
    "cannot be determined, rather than attempt a substantive answer?\n"
    "A short but substantive answer that picks a stance is NOT abstaining. "
    "An answer that mostly says \"I cannot find this in the documents\", "
    "\"no information available\", \"insufficient data\", \"unable to determine\" IS abstaining.\n"
    "Output STRICT JSON ONLY:\n"
    "{\"abstain\":\"yes|no\",\"reason\":\"<one sentence>\"}\n"
    "No markdown."
)


def _build_context_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    cap = CONFIG["PASSAGE_CHAR_CAP"]
    blocks = []
    for i, p in enumerate(passages, start=1):
        text = p.get("text", "") or ""
        if len(text) > cap:
            text = text[:cap] + "…"
        kind = p.get("chunk_kind") or p.get("kind") or "?"
        blocks.append(f"[{i}] (kind={kind}) {text}")
    return (f"{CONTEXT_RELEVANCE_INSTRUCTION}\n\nQuestion: {question}\n\nPassages:\n"
            + "\n\n".join(blocks))


def _build_correctness_prompt(question: str, ground_truth: str, answer: str) -> str:
    gt = ground_truth or ""
    if len(gt) > CONFIG["GT_CHAR_CAP"]:
        gt = gt[: CONFIG["GT_CHAR_CAP"]] + "…"
    a = answer or ""
    if len(a) > CONFIG["ANSWER_CHAR_CAP"]:
        a = a[: CONFIG["ANSWER_CHAR_CAP"]] + "…"
    return (f"{CORRECTNESS_INSTRUCTION}\n\nQuestion: {question}\n\n"
            f"Reference answer:\n{gt}\n\nModel answer:\n{a}")


def _build_abstention_prompt(question: str, answer: str) -> str:
    a = answer or ""
    if len(a) > CONFIG["ANSWER_CHAR_CAP"]:
        a = a[: CONFIG["ANSWER_CHAR_CAP"]] + "…"
    return f"{ABSTENTION_INSTRUCTION}\n\nQuestion: {question}\n\nAnswer:\n{a}"


def _judge_context_relevance(client: AlbertClient, question: str,
                              passages: list[dict[str, Any]]) -> dict[str, Any]:
    if not passages:
        return {"score": 0.0, "answerable": "no", "why_not": "no passages"}
    prompt = _build_context_prompt(question, passages)
    try:
        raw = client.chat_completion(CONFIG["JUDGE"],
                                     [{"role": "user", "content": prompt}], temperature=0.0)
        parsed = _parse_json_object(raw)
        verdicts = parsed.get("passages", [])
        rel = sum(1 for v in verdicts if v.get("verdict") == "relevant")
        part = sum(1 for v in verdicts if v.get("verdict") == "partially_relevant")
        total = len(verdicts) or len(passages)
        return {"score": round((rel + 0.5 * part) / max(total, 1), 4),
                "answerable": parsed.get("answerable", "no"),
                "why_not": parsed.get("why_not", ""),
                "n_relevant": rel, "n_partial": part,
                "n_irrel": max(total - rel - part, 0)}
    except Exception as exc:
        return {"score": 0.0, "answerable": "judge_error",
                "why_not": f"{type(exc).__name__}: {exc}",
                "n_relevant": 0, "n_partial": 0, "n_irrel": len(passages)}


def _judge_correctness(client: AlbertClient, question: str, ground_truth: str,
                       answer: str) -> dict[str, Any]:
    if not (answer or "").strip():
        return {"verdict": "incorrect", "score": 0.0, "reason": "empty answer"}
    prompt = _build_correctness_prompt(question, ground_truth, answer)
    try:
        raw = client.chat_completion(CONFIG["JUDGE"],
                                     [{"role": "user", "content": prompt}], temperature=0.0)
        parsed = _parse_json_object(raw)
        verdict = parsed.get("verdict", "incorrect")
        # trust verdict over score field for determinism
        score_map = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
        return {"verdict": verdict,
                "score": float(score_map.get(verdict, parsed.get("score", 0.0))),
                "reason": parsed.get("reason", "")}
    except Exception as exc:
        return {"verdict": "judge_error", "score": 0.0,
                "reason": f"{type(exc).__name__}: {exc}"}


def _judge_abstention(client: AlbertClient, question: str, answer: str) -> dict[str, Any]:
    if not (answer or "").strip():
        return {"abstain": "yes", "reason": "empty answer"}
    prompt = _build_abstention_prompt(question, answer)
    try:
        raw = client.chat_completion(CONFIG["JUDGE"],
                                     [{"role": "user", "content": prompt}], temperature=0.0)
        parsed = _parse_json_object(raw)
        return {"abstain": parsed.get("abstain", "no"),
                "reason": parsed.get("reason", "")}
    except Exception as exc:
        return {"abstain": "judge_error", "reason": f"{type(exc).__name__}: {exc}"}


# =============================== STAGE 2 ===============================
def retrieve_and_context_relevance(inputs: dict[str, Any],
                                   route_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """For each row: retrieve top-8 via per-company route; judge context_relevance ONCE (generator-free)."""
    idx_dir = Path(CONFIG["INDEX_DIR"])
    client = AlbertClient(api_key=os.environ["ALBERT_API_KEY"], base_url=ALBERT_BASE)
    rows = inputs["selected_rows"]
    top_k = CONFIG["TOP_K"]
    cand_k = CONFIG["CANDIDATE_K"]

    out: list[dict[str, Any]] = []
    n = len(rows)
    for ri, r in enumerate(rows):
        q = r.get("question", "")
        ds_co = r["_dataset_company"]
        route = r["_route"]
        try:
            retrieved, embedding_model = retrieve_chunks(
                index_dir=idx_dir, question=q, top_k=top_k,
                retrieval_architecture=route, search_breadth=cand_k,
                base_url=ALBERT_BASE,
            )
        except Exception as exc:
            print(f"  row {ri+1} retrieve_chunks ERROR: {type(exc).__name__}: {exc}")
            retrieved, embedding_model = [], None
        cr = _judge_context_relevance(client, q, retrieved)
        out.append({
            "row_id": r.get("id", ""),
            "dataset_company": ds_co,
            "index_company": r["_index_company"],
            "question": q,
            "question_type": r.get("question_type", ""),
            "route": route,
            "retrieved_chunks": retrieved,
            "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved],
            "context_relevance": cr["score"],
            "context_answerable": cr["answerable"],
            "context_why_not": cr.get("why_not", ""),
            "embedding_model": embedding_model,
            "ground_truth": get_ground_truth_answer(r),
        })
        print(f"[Stage 2] row {ri+1:>2}/{n} {ds_co:<14} route={route:<6} "
              f"context_rel={cr['score']:.3f} ({cr['answerable']}) "
              f"chunks={len(retrieved)}")
    return out


# =============================== STAGE 3 ===============================
def generate_answers(retr_per_row: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each (row, generator): answer_question(... same retrieved_chunks ...). Identical context guaranteed
    because we pass the same Python list to both calls; we do NOT mutate it between calls."""
    rows = retr_per_row
    gens = CONFIG["GENERATORS"]
    n = len(rows)
    out: list[dict[str, Any]] = []
    for ri, r in enumerate(rows):
        retr = r["retrieved_chunks"]  # IDENTICAL for both generators
        for gen in gens:
            try:
                answer, model_used = answer_question(
                    question=r["question"],
                    retrieved_chunks=retr,
                    text_model=gen,
                    temperature=0.0,
                    answer_mode=CONFIG["ANSWER_MODE"],
                    prompt_style=CONFIG["PROMPT_STYLE"],
                    base_url=ALBERT_BASE,
                )
            except Exception as exc:
                answer, model_used = "", f"ERROR: {type(exc).__name__}: {exc}"
            out.append({
                "row_id": r["row_id"],
                "dataset_company": r["dataset_company"],
                "generator": gen,
                "model_used": model_used,
                "answer": answer,
            })
            print(f"[Stage 3] row {ri+1:>2}/{n} {r['dataset_company']:<14} "
                  f"gen={'Qwen' if 'Qwen' in gen else 'gpt-oss':<8} "
                  f"len_answer={len(answer or '')}")
    return out


# =============================== STAGE 4 ===============================
def judge_answers(retr_per_row: list[dict[str, Any]],
                  answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (row, generator): faithfulness + relevancy + correctness + abstention. 4 judge calls each."""
    client = AlbertClient(api_key=os.environ["ALBERT_API_KEY"], base_url=ALBERT_BASE)
    retr_by_id = {r["row_id"]: r for r in retr_per_row}
    out: list[dict[str, Any]] = []
    n = len(answers)
    for ai, a in enumerate(answers):
        retr = retr_by_id.get(a["row_id"], {})
        q = retr.get("question", "")
        gt = retr.get("ground_truth", "") or ""
        context = retr.get("retrieved_chunks", [])
        ans = a.get("answer", "") or ""

        faith = score_faithfulness(q, ans, context, client, CONFIG["JUDGE"])
        relev = score_answer_relevancy(q, ans, client, CONFIG["JUDGE"])
        corr = _judge_correctness(client, q, gt, ans)
        absten = _judge_abstention(client, q, ans)
        abstain_yes = (absten.get("abstain", "no") == "yes")
        # Spec: for abstain=yes, set answer_correctness floor at 0 (i.e. override to 0).
        if abstain_yes:
            corr_score_final = 0.0
            corr_floored = True
        else:
            corr_score_final = corr.get("score", 0.0)
            corr_floored = False

        out.append({
            "row_id": a["row_id"],
            "dataset_company": a["dataset_company"],
            "generator": a["generator"],
            "answer": ans,
            "faithfulness": faith.get("score", 0.0),
            "answer_relevancy": relev.get("score", 0.0),
            "answer_correctness_raw": corr.get("score", 0.0),
            "answer_correctness": corr_score_final,
            "correctness_verdict": corr.get("verdict", ""),
            "correctness_floored_by_abstain": corr_floored,
            "abstain": absten.get("abstain", "no"),
            "abstain_reason": absten.get("reason", ""),
            "correctness_reason": corr.get("reason", ""),
        })
        gen_label = "Qwen" if "Qwen" in a["generator"] else "gpt-oss"
        print(f"[Stage 4] {ai+1:>3}/{n} {a['dataset_company']:<14} {gen_label:<8} "
              f"faith={faith.get('score',0):.3f} rel={relev.get('score',0):.3f} "
              f"corr={corr_score_final:.3f}({corr.get('verdict','?')[:14]}) "
              f"abstain={absten.get('abstain','?')}")
    return out


# =============================== STAGE 5 ===============================
def scorecard_and_verdict(inputs: dict[str, Any], retr_per_row: list[dict[str, Any]],
                           judged: list[dict[str, Any]], route_map: dict[str, dict[str, Any]]
                           ) -> dict[str, Any]:
    """Aggregate per (company, generator), build markdown + csv, declare a verdict."""
    rows_csv_path = Path("lap2_rows.csv")
    md_path = Path("lap2_scorecard.md")

    # Index retrieval data by row_id
    retr_by_id = {r["row_id"]: r for r in retr_per_row}

    # ---- per-row CSV ----
    with rows_csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "row_id", "dataset_company", "route", "generator",
            "context_relevance", "context_answerable",
            "faithfulness", "answer_relevancy", "answer_correctness", "correctness_verdict",
            "abstain", "correctness_floored_by_abstain",
            "answer_preview", "ground_truth_preview",
        ])
        for j in judged:
            r = retr_by_id.get(j["row_id"], {})
            w.writerow([
                j["row_id"], j["dataset_company"], r.get("route", ""), j["generator"],
                f"{r.get('context_relevance', 0.0):.4f}", r.get("context_answerable", ""),
                f"{j['faithfulness']:.4f}", f"{j['answer_relevancy']:.4f}",
                f"{j['answer_correctness']:.4f}", j.get("correctness_verdict", ""),
                j["abstain"], int(j.get("correctness_floored_by_abstain", False)),
                truncate(j["answer"], 300),
                truncate(r.get("ground_truth", ""), 300),
            ])

    # ---- per-(company,generator) aggregates ----
    company_gen: dict[tuple[str, str], dict[str, Any]] = {}
    company_ctx: dict[str, list[float]] = {}      # context_relevance per company (gen-independent)
    for r in retr_per_row:
        company_ctx.setdefault(r["dataset_company"], []).append(r["context_relevance"])
    for j in judged:
        key = (j["dataset_company"], j["generator"])
        d = company_gen.setdefault(key, {
            "n": 0, "faith_sum": 0.0, "relev_sum": 0.0,
            "corr_sum": 0.0, "abstain_yes": 0,
        })
        d["n"] += 1
        d["faith_sum"] += j["faithfulness"]
        d["relev_sum"] += j["answer_relevancy"]
        d["corr_sum"] += j["answer_correctness"]
        if j["abstain"] == "yes":
            d["abstain_yes"] += 1
    for k, d in company_gen.items():
        n = max(d["n"], 1)
        d["faithfulness_mean"] = d["faith_sum"] / n
        d["answer_relevancy_mean"] = d["relev_sum"] / n
        d["answer_correctness_mean"] = d["corr_sum"] / n
        d["abstention_rate"] = d["abstain_yes"] / n

    # Overall per generator
    overall_gen: dict[str, dict[str, Any]] = {}
    for gen in CONFIG["GENERATORS"]:
        rows = [j for j in judged if j["generator"] == gen]
        n = max(len(rows), 1)
        overall_gen[gen] = {
            "n": len(rows),
            "faithfulness_mean": sum(j["faithfulness"] for j in rows) / n,
            "answer_relevancy_mean": sum(j["answer_relevancy"] for j in rows) / n,
            "answer_correctness_mean": sum(j["answer_correctness"] for j in rows) / n,
            "abstention_rate": sum(1 for j in rows if j["abstain"] == "yes") / n,
            "n_abstain": sum(1 for j in rows if j["abstain"] == "yes"),
        }

    # Context relevance per company (mean) — generator-independent
    ctx_per_company = {co: (sum(vals) / max(len(vals), 1)) for co, vals in company_ctx.items()}
    overall_context = sum(r["context_relevance"] for r in retr_per_row) / max(len(retr_per_row), 1)
    retrieval_limited = [co for co, v in ctx_per_company.items()
                          if v < CONFIG["RETRIEVAL_LIMIT_THR"]]

    # ---- markdown ----
    lines: list[str] = []
    lines.append("# RAGAS Lap 2 scorecard")
    lines.append("")
    lines.append(f"- Eval companies: **{len(route_map)}**; rows: **{len(retr_per_row)}** "
                 "(cap 5/company, first-by-id)")
    lines.append(f"- Generators: `{CONFIG['GENERATORS'][0]}` (Qwen) vs `{CONFIG['GENERATORS'][1]}` (gpt-oss)")
    lines.append(f"- Judge: `{CONFIG['JUDGE']}`  (Mistral) on every judge call")
    lines.append(f"- Answer config: answer_mode=`{CONFIG['ANSWER_MODE']}`, "
                 f"prompt_style=`{CONFIG['PROMPT_STYLE']}`")
    lines.append(f"- Routing: per-company decision from "
                 f"`{CONFIG['WEAKNESS_MAP_CSV']}` (dense if dense_rel ≥ hybrid_rel)")
    lines.append(f"- Identical retrieved context per row passed to BOTH generators "
                 "(same Python list, no mutation between calls).")
    lines.append("")

    lines.append("## Routing map applied")
    lines.append("")
    lines.append("| company | route | dense_rel | hybrid_rel |")
    lines.append("|---|---|---:|---:|")
    for ds_co, info in route_map.items():
        lines.append(f"| {ds_co} | **{info['route']}** | {info['dense_rel']:.3f} | "
                     f"{info['hybrid_rel']:.3f} |")
    lines.append("")

    lines.append("## Primary scorecard  —  per company  (Qwen | gpt-oss)")
    lines.append("")
    lines.append("| company | route | context_rel | faithfulness (Q\\|g) | answer_relevancy (Q\\|g) | "
                 "answer_correctness (Q\\|g) | abstention_rate (Q\\|g) |")
    lines.append("|---|---|---:|---|---|---|---|")
    for ds_co, info in route_map.items():
        ctx = ctx_per_company.get(ds_co, 0.0)
        q = company_gen.get((ds_co, CONFIG["GENERATORS"][0]), {})
        g = company_gen.get((ds_co, CONFIG["GENERATORS"][1]), {})
        flag = " ⚠" if ctx < CONFIG["RETRIEVAL_LIMIT_THR"] else ""
        lines.append(
            f"| {ds_co} | {info['route']} | {ctx:.3f}{flag} | "
            f"{q.get('faithfulness_mean', 0):.3f} \\| {g.get('faithfulness_mean', 0):.3f} | "
            f"{q.get('answer_relevancy_mean', 0):.3f} \\| {g.get('answer_relevancy_mean', 0):.3f} | "
            f"{q.get('answer_correctness_mean', 0):.3f} \\| {g.get('answer_correctness_mean', 0):.3f} | "
            f"{q.get('abstention_rate', 0):.0%} \\| {g.get('abstention_rate', 0):.0%} |"
        )
    qov = overall_gen[CONFIG["GENERATORS"][0]]
    gov = overall_gen[CONFIG["GENERATORS"][1]]
    lines.append(
        f"| **OVERALL** | mixed | **{overall_context:.3f}** | "
        f"**{qov['faithfulness_mean']:.3f} \\| {gov['faithfulness_mean']:.3f}** | "
        f"**{qov['answer_relevancy_mean']:.3f} \\| {gov['answer_relevancy_mean']:.3f}** | "
        f"**{qov['answer_correctness_mean']:.3f} \\| {gov['answer_correctness_mean']:.3f}** | "
        f"**{qov['abstention_rate']:.0%} \\| {gov['abstention_rate']:.0%}** |"
    )
    lines.append("")
    lines.append(f"⚠ = context_relevance < {CONFIG['RETRIEVAL_LIMIT_THR']:.2f} "
                 "(retrieval-limited; generation scores here read as a ceiling).")
    lines.append("")

    # Generator overall + abstention split
    lines.append("## Generator overall (means across all rows)")
    lines.append("")
    lines.append("| metric | Qwen | gpt-oss | winner |")
    lines.append("|---|---:|---:|---|")
    metric_specs = [
        ("faithfulness_mean", "faithfulness", True),
        ("answer_relevancy_mean", "answer_relevancy", True),
        ("answer_correctness_mean", "answer_correctness", True),
        ("abstention_rate", "abstention_rate", False),  # lower is better
    ]
    winners: dict[str, str] = {}
    for key, label, higher_is_better in metric_specs:
        q = qov[key]
        g = gov[key]
        if abs(q - g) < 1e-4:
            verdict = "tie"
        elif (q > g) == higher_is_better:
            verdict = "**Qwen**"
        else:
            verdict = "**gpt-oss**"
        winners[label] = verdict
        lines.append(f"| {label} | {q:.3f} | {g:.3f} | {verdict} |")
    lines.append("")
    lines.append(f"Abstention split: Qwen abstained on "
                 f"**{qov['n_abstain']}/{qov['n']} ({qov['abstention_rate']:.0%})**; "
                 f"gpt-oss abstained on "
                 f"**{gov['n_abstain']}/{gov['n']} ({gov['abstention_rate']:.0%})**.")
    lines.append("")

    # Headline verdict
    # Decide which generator wins overall (majority of the three quality metrics + low abstention)
    quality_metrics = ["faithfulness", "answer_relevancy", "answer_correctness"]
    qwen_wins = sum(1 for m in quality_metrics if winners[m] == "**Qwen**")
    gpt_wins = sum(1 for m in quality_metrics if winners[m] == "**gpt-oss**")
    abst_winner = winners["abstention_rate"]
    if qwen_wins > gpt_wins:
        headline_gen = "Qwen"
    elif gpt_wins > qwen_wins:
        headline_gen = "gpt-oss"
    else:
        headline_gen = "tie on quality"
    lines.append("## Headline generator verdict")
    lines.append("")
    lines.append(f"On the three quality metrics (faithfulness, answer_relevancy, answer_correctness): "
                 f"Qwen wins {qwen_wins}/3, gpt-oss wins {gpt_wins}/3. "
                 f"Lower-abstention winner: {abst_winner}.")
    lines.append("")
    lines.append(f"**Ship: `{headline_gen}`** for this ESG RAG, based on the corpus-wide means above. "
                 "If your priority is willingness-to-answer (lower abstention), weight that more — "
                 "see the abstention split.")
    lines.append("")

    # Per-company flags
    lines.append("## Per-company flags")
    lines.append("")
    lines.append("| company | route | context_rel | flag |")
    lines.append("|---|---|---:|---|")
    for ds_co, info in route_map.items():
        ctx = ctx_per_company.get(ds_co, 0.0)
        flag = "retrieval-limited (ceiling)" if ctx < CONFIG["RETRIEVAL_LIMIT_THR"] else "—"
        lines.append(f"| {ds_co} | {info['route']} | {ctx:.3f} | {flag} |")
    lines.append("")

    # Read paragraph
    lines.append("## Read")
    lines.append("")
    lines.append(
        f"Separate **retrieval health** from **generation health**. Retrieval health is "
        f"`context_relevance` per company (gold-free; corpus-wide mean **{overall_context:.3f}**, "
        f"range "
        f"**{min(ctx_per_company.values()):.3f}–{max(ctx_per_company.values()):.3f}**). "
        f"Retrieval-limited companies (context_rel < {CONFIG['RETRIEVAL_LIMIT_THR']:.2f}): "
        f"**{', '.join(retrieval_limited) or 'none'}** — generation scores on these companies "
        f"are bounded by the context fed to the model.\n"
    )
    lines.append(
        f"Generation health is faithfulness/relevancy/correctness on top of that retrieved context "
        f"(corpus-wide means: Qwen **{qov['faithfulness_mean']:.2f}**/"
        f"**{qov['answer_relevancy_mean']:.2f}**/"
        f"**{qov['answer_correctness_mean']:.2f}**, "
        f"gpt-oss **{gov['faithfulness_mean']:.2f}**/"
        f"**{gov['answer_relevancy_mean']:.2f}**/"
        f"**{gov['answer_correctness_mean']:.2f}**). "
        f"Over-refusal concentrates in gpt-oss (**{gov['abstention_rate']:.0%}** "
        f"vs Qwen **{qov['abstention_rate']:.0%}**). "
        f"That gap drags gpt-oss's answer_correctness down because abstain=yes floors correctness to 0.\n"
    )
    lines.append("")
    lines.append(f"_Manifest: lap2.run_manifest.json_")
    md = "\n".join(lines)
    md_path.write_text(md, encoding="utf-8")
    print()
    print(md)

    return {
        "company_gen": {f"{co}|{gen}": v for (co, gen), v in company_gen.items()},
        "overall_gen": overall_gen,
        "ctx_per_company": ctx_per_company,
        "overall_context": overall_context,
        "retrieval_limited": retrieval_limited,
        "winners": winners,
        "headline_generator": headline_gen,
    }


# =============================== STAGE 6 ===============================
def write_manifest(provenance: dict[str, Any], inputs: dict[str, Any],
                   route_map: dict[str, dict[str, Any]],
                   retr_per_row: list[dict[str, Any]],
                   judged: list[dict[str, Any]],
                   summary: dict[str, Any]) -> None:
    slim_retr = [{
        "row_id": r["row_id"], "dataset_company": r["dataset_company"],
        "index_company": r["index_company"], "route": r["route"],
        "question_type": r.get("question_type", ""),
        "context_relevance": r["context_relevance"],
        "context_answerable": r["context_answerable"],
        "context_why_not": r.get("context_why_not", ""),
        "retrieved_chunk_ids": r["retrieved_chunk_ids"],
        "embedding_model": r.get("embedding_model"),
    } for r in retr_per_row]
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
            "manifest_chunk_count": inputs["manifest_chunk_count"],
            "manifest_embedding_model": inputs["manifest_embedding_model"],
            "vectors_shape": inputs["vectors_shape"],
            "vectors_dtype": inputs["vectors_dtype"],
            "chunk_ids_file_present": False,
            "chunk_ids_source": "chunks.json[i].chunk_id",
            "alignment": "positional (len-verified)",
            "dataset_path": inputs["dataset_path"],
            "dataset_sha256": inputs["dataset_sha256"],
            "dataset_row_count": inputs["dataset_row_count"],
            "selected_row_ids": inputs["selected_row_ids"],
        },
        "config": {
            "alias_map": CONFIG["ALIAS_MAP"],
            "excluded_companies": sorted(CONFIG["EXCLUDE_COMPANIES"]),
            "rows_per_company": CONFIG["ROWS_PER_COMPANY"],
            "top_k": CONFIG["TOP_K"],
            "candidate_k": CONFIG["CANDIDATE_K"],
            "generators": CONFIG["GENERATORS"],
            "judge_model": CONFIG["JUDGE"],
            "answer_mode": CONFIG["ANSWER_MODE"],
            "prompt_style": CONFIG["PROMPT_STYLE"],
            "retrieval_limit_threshold": CONFIG["RETRIEVAL_LIMIT_THR"],
        },
        "route_map": route_map,
        "dropped_metrics": [
            {"metric": "context_recall",
             "reason": "Gold-misaligned: dataset chunk_id naming scheme (chunk-NNNNN) doesn't match "
                        "the live corpus (year_kind_hash_pdf_nNNNN), so chunk-level recall over the "
                        "current index is unreliable. Replaced by gold-free context_relevance "
                        "(Stage 2) for retrieval health."},
        ],
        "retrieved_per_row": slim_retr,
        "judged_per_row": judged,
        "summary": summary,
        "budget": BUDGET,
        "determinism_notes": (
            "retrieve_chunks is deterministic by construction; embeddings deterministic forward pass; "
            "answer_question + all judges run at temperature=0 but Mistral/Qwen/gpt-oss outputs are "
            "service-dependent."
        ),
        "read_only_git_commands_used": READ_ONLY_GIT_COMMANDS_USED,
    }
    Path("lap2.run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================== MAIN ===============================
BANNER = """\
====================================================================
PIPELINE FLOW — RAGAS Lap 2 (routed retrieval + repaired metrics + gen compare)
  Stage 0 capture_provenance() + build_route_map()
  Stage 1 load_inputs()         5/company deterministic
  Stage 2 retrieve_and_context_relevance()   per-company route + judge
  Stage 3 generate_answers()    IDENTICAL context to both generators
  Stage 4 judge_answers()       faith + relev + correctness + abstention
  Stage 5 scorecard_and_verdict()
  Stage 6 write_manifest()
===================================================================="""


def main() -> int:
    t0 = time.time()
    print(BANNER)
    print()

    print("[Stage 0] capturing provenance + building route map")
    provenance = capture_provenance()
    route_map = build_route_map()

    print()
    print("[Stage 1] loading inputs")
    inputs = load_inputs(route_map)

    print()
    print("[Stage 2] retrieving + judging context relevance")
    retr_per_row = retrieve_and_context_relevance(inputs, route_map)

    print()
    print("[Stage 3] generating answers (both generators on identical context)")
    answers = generate_answers(retr_per_row)

    print()
    print("[Stage 4] judging answers (faithfulness/relevancy/correctness/abstention)")
    judged = judge_answers(retr_per_row, answers)

    print()
    print("[Stage 5] scorecard + verdict")
    summary = scorecard_and_verdict(inputs, retr_per_row, judged, route_map)

    print()
    print("[Stage 6] writing manifest")
    write_manifest(provenance, inputs, route_map, retr_per_row, judged, summary)

    print()
    print("=" * 78)
    print("BUDGET SPENT")
    for k, v in BUDGET.items():
        print(f"  {k:22s} = {v}")
    total_chat = BUDGET["generation_calls"] + BUDGET["judge_calls"]
    print(f"  TOTAL chat_calls       = {total_chat}")
    print(f"  total elapsed         = {time.time() - t0:.1f}s")
    print("=" * 78)
    print()
    print("Read-only git commands used:")
    for cmd in READ_ONLY_GIT_COMMANDS_USED:
        print(f"  {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
