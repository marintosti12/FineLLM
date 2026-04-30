"""
Bench de latence de l'API de triage CHSA.

Mesure latence p50/p90/p95/p99, throughput, erreurs sous charge.
Resultats publies dans le rapport technique (section 7.2).

Usage:
    # Local (mode mock)
    python scripts/bench_latency.py --requests 200 --concurrency 10

    # Sur le Space HF deploye
    API_URL=https://marintosti-chsa-triage-api.hf.space \
    API_KEY=chsa-demo-2026 \
    python scripts/bench_latency.py --requests 100 --concurrency 4 \
        --output docs/bench_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    import requests
except ImportError:
    print("ERREUR: pip install requests", file=sys.stderr)
    sys.exit(1)


# Cas cliniques representatifs (cf. demo_api.py) — variete P1/P2/P3
CASES = [
    {
        "patient_id": "BENCH-P1-001",
        "symptoms": "Douleur thoracique aigue irradiant au bras gauche, sueurs froides",
        "age": 58, "sex": "M",
        "medical_history": "Hypertension, ancien fumeur",
        "vital_signs": {"fc": 112, "ta": "150/95", "spo2": 94},
    },
    {
        "patient_id": "BENCH-P1-002",
        "symptoms": "Eruption diffuse, oedeme des levres, dyspnee apres ingestion de cacahuetes",
        "age": 3, "sex": "F",
        "vital_signs": {"fc": 140, "spo2": 91},
    },
    {
        "patient_id": "BENCH-P2-001",
        "symptoms": "Cephalees intenses depuis 6h, photophobie, fievre 39",
        "age": 28, "sex": "F",
        "vital_signs": {"temperature": 39.1, "fc": 105},
    },
    {
        "patient_id": "BENCH-P3-001",
        "symptoms": "Mal de gorge depuis 2 jours avec fievre legere",
        "age": 30, "sex": "F",
        "vital_signs": {"temperature": 38.2},
    },
    {
        "patient_id": "BENCH-P3-002",
        "symptoms": "Entorse de cheville suite a chute, douleur 4/10, mobilite conservee",
        "age": 24, "sex": "M",
    },
]


@dataclass
class Result:
    ok: bool
    status: int
    latency_ms_client: float  # mesuree cote client (inclut le reseau)
    latency_ms_server: float | None  # rapportee par l'API (inference uniquement)
    priority: str | None
    error: str | None


def call_once(url: str, headers: dict, payload: dict, timeout: float) -> Result:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{url}/triage", headers=headers, json=payload, timeout=timeout)
        dt = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            return Result(False, r.status_code, dt, None, None, r.text[:200])
        body = r.json()
        return Result(True, 200, dt, body.get("latency_ms"), body.get("priority_level"), None)
    except requests.RequestException as exc:
        dt = (time.perf_counter() - t0) * 1000.0
        return Result(False, 0, dt, None, None, str(exc)[:200])


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: list[Result], total_wall_s: float) -> dict:
    ok = [r for r in results if r.ok]
    ko = [r for r in results if not r.ok]
    client_lat = [r.latency_ms_client for r in ok]
    server_lat = [r.latency_ms_server for r in ok if r.latency_ms_server is not None]

    def stats(label: str, vals: list[float]) -> dict:
        if not vals:
            return {f"{label}_count": 0}
        return {
            f"{label}_count": len(vals),
            f"{label}_min_ms": round(min(vals), 1),
            f"{label}_p50_ms": round(percentile(vals, 0.50), 1),
            f"{label}_p90_ms": round(percentile(vals, 0.90), 1),
            f"{label}_p95_ms": round(percentile(vals, 0.95), 1),
            f"{label}_p99_ms": round(percentile(vals, 0.99), 1),
            f"{label}_max_ms": round(max(vals), 1),
            f"{label}_mean_ms": round(statistics.fmean(vals), 1),
        }

    summary = {
        "total_requests": len(results),
        "success": len(ok),
        "errors": len(ko),
        "error_rate": round(len(ko) / max(1, len(results)), 4),
        "wall_time_s": round(total_wall_s, 2),
        "throughput_rps": round(len(ok) / max(0.001, total_wall_s), 2),
    }
    summary.update(stats("client", client_lat))
    summary.update(stats("server", server_lat))

    # Distribution des priorites attribuees (controle qualitatif)
    prio_counts: dict[str, int] = {}
    for r in ok:
        key = r.priority or "?"
        prio_counts[key] = prio_counts.get(key, 0) + 1
    summary["priority_distribution"] = prio_counts

    # Premieres erreurs (max 5) pour diagnostic
    summary["sample_errors"] = [
        {"status": r.status, "error": r.error} for r in ko[:5]
    ]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Bench latence API triage CHSA")
    parser.add_argument("--url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--requests", type=int, default=100, help="Nombre total de requetes")
    parser.add_argument("--concurrency", type=int, default=4, help="Workers paralleles")
    parser.add_argument("--warmup", type=int, default=3, help="Requetes de warmup (ignorees)")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default=None, help="Fichier JSON pour ecrire les resultats")
    args = parser.parse_args()

    url = args.url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    print(f"Bench triage CHSA — {args.requests} req / concurrency={args.concurrency} → {url}")

    # Health check
    try:
        h = requests.get(f"{url}/health", timeout=10)
        print(f"  /health: {h.status_code} {h.json()}")
    except Exception as exc:
        print(f"  /health KO: {exc}")
        return 1

    # Warmup (chauffe le cache vLLM / KV cache)
    if args.warmup > 0:
        print(f"  warmup: {args.warmup} requetes ignorees")
        for i in range(args.warmup):
            call_once(url, headers, CASES[i % len(CASES)], args.timeout)

    print("  bench en cours ...")
    results: list[Result] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(call_once, url, headers, CASES[i % len(CASES)], args.timeout)
            for i in range(args.requests)
        ]
        for f in as_completed(futures):
            results.append(f.result())
    wall = time.perf_counter() - t0

    summary = summarize(results, wall)
    summary["config"] = {
        "url": url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
    }
    summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nEcrit dans {args.output}")

    # Exit 1 si taux d'erreur > 5% (utilisable en CI smoke test)
    if summary["error_rate"] > 0.05:
        print(f"\nERREUR: taux d'erreur eleve ({summary['error_rate']:.1%})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
