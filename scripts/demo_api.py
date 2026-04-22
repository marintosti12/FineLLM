"""
Demo rapide de l'API de triage CHSA — 3 cas cliniques couvrant P1/P2/P3.

Usage:
    python scripts/demo_api.py
    API_URL=https://marintosti-chsa-triage-api.hf.space \
    API_KEY=chsa-demo-2026 \
    python scripts/demo_api.py
"""

import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("ERREUR: pip install requests")
    sys.exit(1)


API_URL = os.environ.get("API_URL", "https://marintosti-chsa-triage-api.hf.space").rstrip("/")
API_KEY = os.environ.get("API_KEY", "123456")

CASES = [
    {
        "label": "CAS 1 - Suspicion d'infarctus (P1 attendu)",
        "payload": {
            "patient_id": "DEMO-001",
            "symptoms": "Douleur thoracique aigue irradiant au bras gauche, sueurs froides, depuis 30 minutes",
            "age": 58,
            "sex": "M",
            "medical_history": "Hypertension, ancien fumeur",
            "vital_signs": {"fc": 112, "ta": "150/95", "spo2": 94},
        },
    },
    {
        "label": "CAS 2 - Choc anaphylactique pediatrique (P1 attendu)",
        "payload": {
            "patient_id": "DEMO-002",
            "symptoms": "Eruption cutanee diffuse, oedeme des levres, dyspnee apres ingestion de cacahuetes",
            "age": 3,
            "sex": "F",
            "vital_signs": {"fc": 140, "spo2": 91},
        },
    },
    {
        "label": "CAS 3 - Angine simple (P3 attendu)",
        "payload": {
            "patient_id": "DEMO-003",
            "symptoms": "Mal de gorge depuis 2 jours avec fievre legere",
            "age": 30,
            "sex": "F",
            "vital_signs": {"temperature": 38.2},
        },
    },
]


def check_health() -> bool:
    print(f"→ Health check sur {API_URL}/health ...")
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        print(f"  Status: {r.status_code}")
        print(f"  Body  : {r.json()}")
        return r.status_code == 200
    except Exception as e:
        print(f"  ERREUR: {e}")
        return False


def run_case(case: dict) -> None:
    print("\n" + "=" * 70)
    print(case["label"])
    print("=" * 70)
    print(f"Symptomes : {case['payload']['symptoms']}")
    print(f"Age / sex : {case['payload'].get('age')} / {case['payload'].get('sex')}")

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    t0 = time.perf_counter()
    r = requests.post(f"{API_URL}/triage", headers=headers, json=case["payload"], timeout=60)
    dt = (time.perf_counter() - t0) * 1000

    print(f"\n→ HTTP {r.status_code} en {dt:.0f} ms")
    if r.status_code != 200:
        print(f"  ERREUR: {r.text}")
        return

    body = r.json()
    print(f"  Priorite   : {body['priority_level']}")
    print(f"  Latence    : {body['latency_ms']} ms (cote serveur)")
    print(f"  Backend    : {body['backend']}")
    print(f"  Reponse    :")
    print("  " + "-" * 66)
    for line in body["explanation"].splitlines():
        print(f"  | {line}")
    print("  " + "-" * 66)


def main() -> None:
    print("CHSA - Agent IA Triage Medical - Demo API")
    print(f"URL: {API_URL}")
    print(f"Key: {'***' + API_KEY[-4:] if API_KEY else '(aucune)'}")
    print()

    if not check_health():
        print("\nLe Space n'est pas joignable ou le modele n'est pas charge. Arret.")
        sys.exit(1)

    for case in CASES:
        run_case(case)

    # Affiche l'audit log
    print("\n" + "=" * 70)
    print("AUDIT LOG")
    print("=" * 70)
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    r = requests.get(f"{API_URL}/audit?limit=5", headers=headers, timeout=10)
    if r.status_code == 200:
        audit = r.json()
        print(f"Total interactions : {audit['total_interactions']}")
        for entry in audit["entries"][-3:]:
            print(f"  - {entry['timestamp']} | {entry['patient_id']} | "
                  f"{entry['priority_level']} | {entry['latency_ms']} ms")
    else:
        print(f"  ERREUR /audit: {r.status_code}")


if __name__ == "__main__":
    main()
