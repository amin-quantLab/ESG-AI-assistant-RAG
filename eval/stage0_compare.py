"""Stage 0: generator comparison (Qwen vs gpt-oss) on shipped retrieval, fixed neutral judge.

Read-only with respect to app/*. Writes only under outputs/ (per-run payloads under
outputs/stage0/, plus three aggregate files at outputs/stage0_*).

Run from eval/ with PYTHONPATH=..:
    python stage0_compare.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# -------------------------- CONFIG (edit here) --------------------------
INDEX_DIR        = "../outputs/rag_index"
DATASET          = "../sample_data/ragas_esg_eval_dataset.csv"
COMPANIES        = ["Danone", "Enel", "TotalEnergies", "Volkswagen"]
GENERATORS       = ["Qwen/Qwen3-Coder-30B-A3B-Instruct", "openai/gpt-oss-120b"]
JUDGE            = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
RETRIEVAL_MODE   = "semantic_rerank"
CANDIDATE_K      = 100
TOP_K            = 8
ANSWER_MODE      = "assistant"
PROMPT_STYLE     = "balanced"
LIMIT_ROWS       = 5

REFUSAL_RE = re.compile(
    r"cannot|not (available|provided|found|disclosed|able)|"
    r"no (information|data)|insufficient|unable to",
    re.IGNORECASE,
)


# -------------------------- STEP -1: ENV --------------------------
def load_env_files() -> None:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / ".." / "esg_scraper" / ".env",   # eval/ -> Script/esg_scraper/.env
        here / ".." / "esg_scraper" / ".env",          # legacy: from repo root
        Path(".env"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, val)
        return


load_env_files()
if not os.environ.get("ALBERT_API_KEY"):
    print(
        "ALBERT_API_KEY is not set and no .env file provided one.\n"
        "Set it before running, for example:\n"
        "  PowerShell:  $env:ALBERT_API_KEY = '...'; $env:ALBERT_BASE_URL = '...'\n"
        "  bash:        export ALBERT_API_KEY='...'; export ALBERT_BASE_URL='...'\n",
        file=sys.stderr,
    )
    sys.exit(1)

# Imports that touch Albert come AFTER env is ensured.
from app.rag import AlbertClient  # noqa: E402
from app.ragas_eval import run_full_ragas_eval  # noqa: E402


# -------------------------- BUDGET COUNTERS --------------------------
_BUDGET = {"chat_completions": 0, "embedding_calls": 0, "embedding_strings": 0}

_orig_chat = AlbertClient.chat_completion
_orig_embed = AlbertClient.create_embeddings


def _chat_wrapper(self, model, messages, *args, **kwargs):
    _BUDGET["chat_completions"] += 1
    return _orig_chat(self, model, messages, *args, **kwargs)


def _embed_wrapper(self, model, inputs):
    _BUDGET["embedding_calls"] += 1
    _BUDGET["embedding_strings"] += len(inputs)
    return _orig_embed(self, model, inputs)


AlbertClient.chat_completion = _chat_wrapper
AlbertClient.create_embeddings = _embed_wrapper


# -------------------------- HELPERS --------------------------
def gen_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]", "_", model_id)


def short_gen_label(model_id: str) -> str:
    if "Qwen" in model_id:
        return "Qwen"
    if "gpt-oss" in model_id:
        return "gpt-oss"
    return model_id


def is_refusal_text(text: str) -> bool:
    if not text or not text.strip():
        return True
    return bool(REFUSAL_RE.search(text))


def preview(text: str, n: int = 300) -> str:
    if not text:
        return ""
    return " ".join(text.split())[:n]


def harmonic_mean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def l2_norm(vec):
    s = math.sqrt(sum(x * x for x in vec))
    if s == 0:
        return vec
    return [x / s for x in vec]


def dot(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def fmt(v: float) -> str:
    return f"{v:.3f}"


# -------------------------- STEP 0: PRE-FLIGHT --------------------------
print("=" * 78)
print("PRE-FLIGHT: semantic_rerank sanity gate (Danone, 3 rows, retrieval-only)")
print("=" * 78)
try:
    pf_payload = run_full_ragas_eval(
        index_dir=Path(INDEX_DIR),
        dataset_path=Path(DATASET),
        company="Danone",
        eval_mode="retrieval",
        retrieval_mode="semantic_rerank",
        candidate_k=CANDIDATE_K,
        top_k=TOP_K,
        judge_model=JUDGE,
        limit_rows=3,
    )
except Exception as exc:
    print(f"PRE-FLIGHT ERROR: {type(exc).__name__}: {exc}")
    print("STOP - semantic_rerank failed. Not proceeding to comparison.")
    sys.exit(2)

pf_rsum = pf_payload.get("retrieval_summary", {})
pf_hit = float(pf_rsum.get("hit_rate", 0.0))
pf_partial = float(pf_rsum.get("partial_or_hit_rate", 0.0))
pf_avg = float(pf_rsum.get("average_best_match_score", 0.0))
print(f"PRE-FLIGHT hit_rate={pf_hit:.1%}  partial+hit={pf_partial:.1%}  avg_match={pf_avg:.2f}")
if pf_hit <= 0:
    print("NOTE: semantic_rerank hit_rate is 0% on the preflight sample - proceeding "
          "since no error was raised, but flagging this for the floor-effect check.")


# -------------------------- STEP 1: MAIN LOOP --------------------------
out_dir = Path("stage0")
out_dir.mkdir(parents=True, exist_ok=True)
payloads: dict[tuple[str, str], dict[str, Any]] = {}

print()
print("=" * 78)
print(f"MAIN LOOP: {len(COMPANIES)} companies x {len(GENERATORS)} generators = "
      f"{len(COMPANIES) * len(GENERATORS)} runs, {LIMIT_ROWS} rows each "
      f"(target {len(COMPANIES) * len(GENERATORS) * LIMIT_ROWS} rows)")
print("=" * 78)

for company in COMPANIES:
    for gen in GENERATORS:
        key = (company, gen)
        path = out_dir / f"{company}__{gen_slug(gen)}.json"
        print(f"[run] company={company:<14} gen={short_gen_label(gen):<8} -> {path.name}")
        payload = run_full_ragas_eval(
            index_dir=Path(INDEX_DIR),
            dataset_path=Path(DATASET),
            company=company,
            eval_mode="ragas",
            generator_model=gen,
            judge_model=JUDGE,
            retrieval_mode=RETRIEVAL_MODE,
            candidate_k=CANDIDATE_K,
            top_k=TOP_K,
            answer_mode=ANSWER_MODE,
            prompt_style=PROMPT_STYLE,
            limit_rows=LIMIT_ROWS,
        )
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payloads[key] = payload


# -------------------------- STEP 2: BONUS METRICS --------------------------
embed_client = AlbertClient(
    api_key=os.environ["ALBERT_API_KEY"],
    base_url=os.environ.get("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1"),
)
embed_model = embed_client.get_embedding_model(preferred="bge-m3")

for (company, gen), payload in payloads.items():
    rows = payload.get("results", [])
    answers: list[str] = []
    gts: list[str] = []
    answered_idx: list[int] = []
    for i, r in enumerate(rows):
        ans = r.get("answer") or ""
        gt = r.get("ground_truth") or ""
        ragas = r.get("ragas")
        answered = bool(ans.strip()) and isinstance(ragas, dict)
        r["_answered"] = answered
        r["_refusal"] = (not answered) or is_refusal_text(ans)
        r["_answer_correctness"] = 0.0
        r["_f1"] = 0.0
        if answered and gt.strip():
            answers.append(ans)
            gts.append(gt)
            answered_idx.append(i)

    if answers:
        try:
            vecs = embed_client.create_embeddings(embed_model, answers + gts)
            n = len(answers)
            ans_vecs = [l2_norm(v) for v in vecs[:n]]
            gt_vecs = [l2_norm(v) for v in vecs[n:]]
            for j, i in enumerate(answered_idx):
                rows[i]["_answer_correctness"] = dot(ans_vecs[j], gt_vecs[j])
        except Exception as exc:
            print(f"  [warn] embedding call failed for {company}/{short_gen_label(gen)}: "
                  f"{type(exc).__name__}: {exc}")

    for r in rows:
        if r["_answered"]:
            cp = float(r["ragas"].get("context_precision", 0.0))
            cr = float(r["ragas"].get("context_recall", 0.0))
            r["_f1"] = harmonic_mean(cp, cr)


# -------------------------- STEP 4: AGGREGATE --------------------------
PRIMARY_METRICS = ["faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_correctness", "f1"]
EXT_METRICS = ["hallucination_free", "citation_coverage", "unsupported_claims_score"]


def metric_value(row: dict, m: str) -> float:
    if not row.get("_answered"):
        return 0.0
    if m == "answer_correctness":
        return float(row.get("_answer_correctness", 0.0))
    if m == "f1":
        return float(row.get("_f1", 0.0))
    return float(row.get("ragas", {}).get(m, 0.0))


summary: dict[tuple[str, str], dict[str, Any]] = {}
for (company, gen), payload in payloads.items():
    rows = payload.get("results", [])
    n_rows = len(rows)
    n_answered = sum(1 for r in rows if r.get("_answered"))
    n_refusal = sum(1 for r in rows if r.get("_refusal"))
    means = {m: (sum(metric_value(r, m) for r in rows) / n_rows if n_rows else 0.0)
             for m in PRIMARY_METRICS + EXT_METRICS}
    summary[(company, gen)] = {"n_rows": n_rows, "n_answered": n_answered,
                               "n_refusal": n_refusal, **means}

overall: dict[str, dict[str, Any]] = {}
for gen in GENERATORS:
    pooled: list[dict[str, Any]] = []
    for company in COMPANIES:
        pooled.extend(payloads[(company, gen)].get("results", []))
    n_rows = len(pooled)
    n_answered = sum(1 for r in pooled if r.get("_answered"))
    n_refusal = sum(1 for r in pooled if r.get("_refusal"))
    means = {m: (sum(metric_value(r, m) for r in pooled) / n_rows if n_rows else 0.0)
             for m in PRIMARY_METRICS + EXT_METRICS}
    overall[gen] = {"n_rows": n_rows, "n_answered": n_answered,
                    "n_refusal": n_refusal, **means}


# -------------------------- ARTIFACT 1: stage0_rows.csv --------------------------
rows_csv = Path("stage0_rows.csv")
with rows_csv.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow([
        "company", "generator", "question", "topic", "answered", "refusal",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "answer_correctness", "f1",
        "hallucination_free", "citation_coverage", "unsupported_claims_score",
        "answer_preview", "ground_truth_preview",
    ])
    for (company, gen), payload in payloads.items():
        for r in payload.get("results", []):
            w.writerow([
                company, gen, r.get("question", ""), r.get("topic", ""),
                int(bool(r.get("_answered"))), int(bool(r.get("_refusal"))),
                f"{metric_value(r, 'faithfulness'):.4f}",
                f"{metric_value(r, 'answer_relevancy'):.4f}",
                f"{metric_value(r, 'context_precision'):.4f}",
                f"{metric_value(r, 'context_recall'):.4f}",
                f"{metric_value(r, 'answer_correctness'):.4f}",
                f"{metric_value(r, 'f1'):.4f}",
                f"{metric_value(r, 'hallucination_free'):.4f}",
                f"{metric_value(r, 'citation_coverage'):.4f}",
                f"{metric_value(r, 'unsupported_claims_score'):.4f}",
                preview(r.get("answer", "")),
                preview(r.get("ground_truth", "")),
            ])

# -------------------------- ARTIFACT 2: stage0_summary.csv --------------------------
sum_csv = Path("stage0_summary.csv")
header_metrics = PRIMARY_METRICS + EXT_METRICS
with sum_csv.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["company", "generator", "n_rows", "n_answered", "n_refusal", *header_metrics])
    for company in COMPANIES:
        for gen in GENERATORS:
            s = summary[(company, gen)]
            w.writerow([company, gen, s["n_rows"], s["n_answered"], s["n_refusal"],
                        *[f"{s[m]:.4f}" for m in header_metrics]])
    for gen in GENERATORS:
        s = overall[gen]
        w.writerow(["OVERALL", gen, s["n_rows"], s["n_answered"], s["n_refusal"],
                    *[f"{s[m]:.4f}" for m in header_metrics]])


# -------------------------- ARTIFACT 3: stage0_comparison.md --------------------------
lines: list[str] = []
lines.append("# Stage 0 - Generator comparison (Qwen vs gpt-oss)")
lines.append("")
lines.append(f"- Retrieval: `{RETRIEVAL_MODE}` (candidate_k={CANDIDATE_K}, top_k={TOP_K})")
lines.append(f"- Answer config: answer_mode=`{ANSWER_MODE}`, prompt_style=`{PROMPT_STYLE}`")
lines.append(f"- Companies: {', '.join(COMPANIES)}; rows/company={LIMIT_ROWS}; judge=`{JUDGE}`")
lines.append("- Metric means are **micro-averaged (pooled rows)**; non-answered rows count as 0.")
lines.append("")
lines.append("## Primary table (Qwen | gpt-oss)")
lines.append("")
header = "| Company | " + " | ".join(PRIMARY_METRICS) + " |"
sep = "|" + "|".join(["---"] * (len(PRIMARY_METRICS) + 1)) + "|"
lines.append(header)
lines.append(sep)
for company in COMPANIES:
    cells = []
    for m in PRIMARY_METRICS:
        q = summary[(company, GENERATORS[0])][m]
        g = summary[(company, GENERATORS[1])][m]
        cells.append(f"{fmt(q)} \\| {fmt(g)}")
    lines.append(f"| {company} | " + " | ".join(cells) + " |")
ov_cells = []
for m in PRIMARY_METRICS:
    q = overall[GENERATORS[0]][m]
    g = overall[GENERATORS[1]][m]
    ov_cells.append(f"**{fmt(q)} \\| {fmt(g)}**")
lines.append("| **OVERALL (mean)** | " + " | ".join(ov_cells) + " |")
lines.append("")

lines.append("## Extended diagnostics (OVERALL row only, Qwen | gpt-oss)")
lines.append("")
diag_cols = EXT_METRICS + ["n_answered/n_rows", "n_refusal"]
lines.append("| " + " | ".join(diag_cols) + " |")
lines.append("|" + "|".join(["---"] * len(diag_cols)) + "|")
q_ov = overall[GENERATORS[0]]
g_ov = overall[GENERATORS[1]]
diag_cells = []
for m in EXT_METRICS:
    diag_cells.append(f"{fmt(q_ov[m])} \\| {fmt(g_ov[m])}")
diag_cells.append(f"{q_ov['n_answered']}/{q_ov['n_rows']} \\| "
                  f"{g_ov['n_answered']}/{g_ov['n_rows']}")
diag_cells.append(f"{q_ov['n_refusal']} \\| {g_ov['n_refusal']}")
lines.append("| " + " | ".join(diag_cells) + " |")
lines.append("")

lines.append("## Winners (OVERALL, primary metrics)")
lines.append("")
for m in PRIMARY_METRICS:
    q = q_ov[m]
    g = g_ov[m]
    if abs(q - g) < 1e-4:
        lines.append(f"- **{m}**: tie ({fmt(q)} vs {fmt(g)})")
    elif q > g:
        lines.append(f"- **{m}**: Qwen by {fmt(q - g)} ({fmt(q)} vs {fmt(g)})")
    else:
        lines.append(f"- **{m}**: gpt-oss by {fmt(g - q)} ({fmt(q)} vs {fmt(g)})")
lines.append("")

if q_ov["context_recall"] < 0.2 and g_ov["context_recall"] < 0.2:
    lines.append("> **FLOOR-EFFECT FLAG**: OVERALL context_recall < 0.2 for BOTH generators. "
                 "Context-dependent metrics (context_precision, context_recall, faithfulness, "
                 "answer_correctness) are retrieval-limited and may not reflect generator "
                 "quality.")
    lines.append("")

refusal_gap = abs(q_ov["n_refusal"] - g_ov["n_refusal"])
if refusal_gap >= 2:
    higher = "Qwen" if q_ov["n_refusal"] > g_ov["n_refusal"] else "gpt-oss"
    lines.append(f"> **REFUSAL CALLOUT**: {higher} refused {refusal_gap} more rows overall "
                 f"(Qwen={q_ov['n_refusal']}, gpt-oss={g_ov['n_refusal']}). "
                 "Feed into the later prompt-redaction loop.")
    lines.append("")

md = "\n".join(lines)
Path("stage0_comparison.md").write_text(md, encoding="utf-8")

print()
print(md)


# -------------------------- VERIFICATION + BUDGET --------------------------
sample_judge = next(iter(payloads.values())).get("judge_model")
all_same_judge = all(p.get("judge_model") == JUDGE for p in payloads.values())

print()
print("=" * 78)
print("VERIFICATION")
print("=" * 78)
print(f"  preflight semantic_rerank hit_rate            = {pf_hit:.1%}")
print(f"  payloads written to stage0/                   = "
      f"{sum(1 for p in out_dir.glob('*.json'))}")
print(f"  aggregate files present:")
for p in ["stage0_rows.csv", "stage0_summary.csv",
          "stage0_comparison.md"]:
    print(f"    {'OK ' if Path(p).is_file() else 'MISS '} {p}")
print(f"  judge_model from first payload                = {sample_judge}")
print(f"  every payload used judge={JUDGE!r}            = {all_same_judge}")
print(f"  any answered<n_rows (refusal/empty observed)? = "
      f"{any(s['n_answered'] != s['n_rows'] for s in summary.values())}")
print()
print(f"BUDGET SPENT")
print(f"  chat completions (generation + judges): {_BUDGET['chat_completions']}")
print(f"  embedding API calls:                    {_BUDGET['embedding_calls']}"
      f"  (total strings embedded: {_BUDGET['embedding_strings']})")
print("=" * 78)
