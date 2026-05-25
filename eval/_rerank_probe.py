"""Read-only rerank endpoint probe. Discards on stdout; writes nothing."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# load .env via setdefault, never print values. Try multiple candidate paths so the script
# works whether run from the repo root (Git_clone/) or from eval/.
here = Path(__file__).resolve().parent
for env_path in (
    here.parent / ".." / "esg_scraper" / ".env",  # eval/ -> Script/esg_scraper/.env
    here / ".." / "esg_scraper" / ".env",          # Git_clone/ -> Script/esg_scraper/.env (legacy)
    Path(".env"),
):
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
        break

from app.rag import AlbertClient

base = os.environ["ALBERT_BASE_URL"].rstrip("/")
# base ends in /v1 by convention; host root = base without trailing /v1
host_root = base[:-3] if base.endswith("/v1") else base
client = AlbertClient(api_key=os.environ["ALBERT_API_KEY"], base_url=base)

print("=== STEP 1: openapi.json discovery ===")
r = client.session.get(f"{host_root}/openapi.json", timeout=20)
print(f"  GET {host_root}/openapi.json -> {r.status_code}")
rerank_paths: list[str] = []
if r.status_code == 200:
    spec = r.json()
    for p in spec.get("paths", {}):
        if "rerank" in p.lower():
            rerank_paths.append(p)
    print(f"  discovered rerank paths: {rerank_paths}")
    for p in rerank_paths:
        ops = spec["paths"][p]
        for method, info in ops.items():
            if not isinstance(info, dict):
                continue
            summary = info.get("summary", "")
            rb = info.get("requestBody", {}).get("content", {}).get("application/json", {})
            schema = rb.get("schema", {})
            props = list(schema.get("properties", {}).keys())
            ref = schema.get("$ref", "")
            print(f"    {method.upper()} {p}  summary={summary[:80]}")
            print(f"      request schema props={props} ref={ref}")
else:
    print(f"  openapi.json not reachable ({r.status_code})")

print()
print("=== STEP 2: candidate endpoint probes ===")
candidates = [f"{host_root}/v1/rerank", f"{host_root}/rerank"]
variants = [
    ("documents", {"model": "BAAI/bge-reranker-v2-m3", "query": "GHG emissions",
                   "documents": ["Scope 1 emissions were 33 Mt CO2e",
                                 "The staff cafeteria menu changed in May"]}),
    ("input",     {"model": "BAAI/bge-reranker-v2-m3", "query": "GHG emissions",
                   "input": ["Scope 1 emissions were 33 Mt CO2e",
                             "The staff cafeteria menu changed in May"]}),
    ("texts",     {"model": "BAAI/bge-reranker-v2-m3", "query": "GHG emissions",
                   "texts": ["Scope 1 emissions were 33 Mt CO2e",
                             "The staff cafeteria menu changed in May"]}),
]

found = None
for url in candidates:
    for vname, body in variants:
        try:
            rr = client.session.post(url, json=body, timeout=20)
            snippet = rr.text[:160].replace("\n", " ")
            print(f"  POST {url:<60} key={vname:<10} -> {rr.status_code}  body[0:160]={snippet}")
            if rr.status_code == 200:
                found = (url, vname, rr.json())
                break
        except Exception as exc:
            print(f"  POST {url}  key={vname}  -> ERROR {type(exc).__name__}: {exc}")
    if found:
        break

print()
if found:
    url, vkey, resp = found
    print(f"WORKING ENDPOINT: {url}")
    print(f"payload key for documents list: {vkey!r}")
    print(f"response top-level keys: {list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__}")
    print(f"response snippet: {json.dumps(resp, default=str)[:600]}")
else:
    print("NO RERANK ENDPOINT REACHABLE -> RERANK_AVAILABLE = False")
