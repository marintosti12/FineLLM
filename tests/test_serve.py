"""Tests de l'API de triage - mode mock (aucun modele charge)."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

# Active le mode mock AVANT d'importer l'app
os.environ.pop("MODEL_ID", None)
os.environ["USE_VLLM"] = "0"
os.environ.pop("API_KEY", None)

from src.deployment.serve import app, audit_log  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        audit_log.clear()
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"].startswith("CHSA")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["backend"] == "mock"
    assert body["auth_enabled"] is False


def test_triage_basic(client):
    payload = {
        "symptoms": "Douleur thoracique intense, essoufflement",
        "age": 60,
        "sex": "M",
    }
    r = client.post("/triage", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["interaction_id"]
    assert body["priority_level"].startswith(("P1", "P2", "P3", "NON"))
    assert body["backend"] == "mock"
    assert body["latency_ms"] >= 0


def test_triage_required_field(client):
    r = client.post("/triage", json={})
    assert r.status_code == 422


def test_audit_traceability(client):
    for i in range(3):
        r = client.post("/triage", json={"symptoms": f"test {i}", "patient_id": f"P{i}"})
        assert r.status_code == 200
    r = client.get("/audit")
    assert r.status_code == 200
    body = r.json()
    assert body["total_interactions"] >= 3
    ids = {e["patient_id"] for e in body["entries"]}
    assert {"P0", "P1", "P2"}.issubset(ids)
    # Chaque entree d'audit contient les champs obligatoires
    for e in body["entries"]:
        assert e["interaction_id"]
        assert e["timestamp"]
        assert "priority_level" in e
        assert "latency_ms" in e


def test_latency_threshold(client):
    """Le mode mock doit repondre largement sous 500 ms."""
    t0 = time.perf_counter()
    r = client.post("/triage", json={"symptoms": "fievre"})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200
    assert elapsed_ms < 500, f"Latence trop elevee: {elapsed_ms:.0f} ms"


def test_api_key_enforcement(monkeypatch):
    """Si API_KEY est defini, les routes protegees doivent exiger le header."""
    from src.deployment import serve

    monkeypatch.setattr(serve, "API_KEY", "secret-test")
    with TestClient(serve.app) as c:
        # sans header -> 401
        r = c.post("/triage", json={"symptoms": "test"})
        assert r.status_code == 401
        # avec mauvais header -> 401
        r = c.post("/triage", json={"symptoms": "test"}, headers={"X-API-Key": "wrong"})
        assert r.status_code == 401
        # avec bon header -> 200
        r = c.post("/triage", json={"symptoms": "test"}, headers={"X-API-Key": "secret-test"})
        assert r.status_code == 200
