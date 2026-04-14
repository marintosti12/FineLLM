"""
Semaine 4 - Deploiement de l'endpoint de demonstration.

API FastAPI de triage medical. Backend d'inference:
- vLLM si USE_VLLM=1 (GPU, prod / HF Space GPU)
- transformers sinon (CPU, fallback / tests CI)
- mode echo si USE_VLLM=0 ET MODEL_ID vide (tests unitaires)

Variables d'environnement:
- MODEL_ID        : id du modele HF Hub (ex: Marintosti/chsa-triage-qwen3-1.7b)
- USE_VLLM        : "1" pour charger vLLM, "0" pour transformers/mock (default: "1")
- API_KEY         : si defini, exige le header X-API-Key
- PORT            : port d'ecoute (default: 7860 - requis par HF Spaces)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = Path(os.environ.get("DEPLOY_CONFIG", PROJECT_ROOT / "configs" / "deployment_config.yaml"))

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_ID = os.environ.get("MODEL_ID", "").strip()
USE_VLLM = os.environ.get("USE_VLLM", "1") == "1"
API_KEY = os.environ.get("API_KEY", "").strip()

# Etat global du backend
_backend: dict[str, Any] = {"kind": "mock", "engine": None, "tokenizer": None}
audit_log: list[dict] = []


def _load_vllm(model_id: str) -> None:
    from vllm import LLM  # type: ignore

    vllm_cfg = config["vllm"]
    _backend["engine"] = LLM(
        model=model_id,
        tensor_parallel_size=vllm_cfg["tensor_parallel_size"],
        max_model_len=vllm_cfg["max_model_len"],
        gpu_memory_utilization=vllm_cfg["gpu_memory_utilization"],
        dtype=vllm_cfg["dtype"],
    )
    _backend["kind"] = "vllm"
    logger.info("Backend vLLM charge: %s", model_id)


def _load_transformers(model_id: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    _backend["engine"] = model
    _backend["tokenizer"] = tok
    _backend["kind"] = "transformers"
    logger.info("Backend transformers (CPU) charge: %s", model_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_ID:
        logger.warning("MODEL_ID non defini - mode mock (echo) active")
    else:
        try:
            if USE_VLLM:
                _load_vllm(MODEL_ID)
            else:
                _load_transformers(MODEL_ID)
        except Exception as exc:  # noqa: BLE001
            logger.error("Echec chargement modele (%s), bascule en mode mock: %s", MODEL_ID, exc)
    yield
    _backend["engine"] = None
    _backend["tokenizer"] = None


app = FastAPI(
    title="CHSA - Agent IA Triage Medical",
    description="POC d'un agent IA de triage medical pour le Centre Hospitalier Saint-Aurelien",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------- Schemas ----------------------------


class TriageRequest(BaseModel):
    patient_id: str | None = None
    symptoms: str
    age: int | None = None
    sex: str | None = None
    medical_history: str | None = None
    vital_signs: dict | None = None


class TriageResponse(BaseModel):
    interaction_id: str
    timestamp: str
    patient_id: str | None
    priority_level: str
    explanation: str
    recommendations: str
    raw_response: str
    latency_ms: float
    backend: str


# ---------------------------- Securite ----------------------------


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


# ---------------------------- Prompt + parsing ----------------------------


def build_prompt(req: TriageRequest) -> str:
    system_prompt = config["triage"]["system_prompt"]
    parts = [f"Symptomes rapportes: {req.symptoms}"]
    if req.age is not None:
        parts.append(f"Age: {req.age} ans")
    if req.sex:
        parts.append(f"Sexe: {req.sex}")
    if req.medical_history:
        parts.append(f"Antecedents medicaux: {req.medical_history}")
    if req.vital_signs:
        vitals = ", ".join(f"{k}: {v}" for k, v in req.vital_signs.items())
        parts.append(f"Constantes vitales: {vitals}")
    parts.append("\nVeuillez evaluer le niveau de priorite et fournir vos recommandations.")
    return f"{system_prompt}\n\nPatient:\n" + "\n".join(parts)


def parse_triage_response(raw: str) -> tuple[str, str, str]:
    raw_upper = raw.upper()
    if "P1" in raw_upper or "URGENCE MAXIMALE" in raw_upper:
        priority = "P1 - URGENCE MAXIMALE"
    elif "P2" in raw_upper or "URGENCE MODEREE" in raw_upper or "URGENCE MODÉRÉE" in raw_upper:
        priority = "P2 - URGENCE MODEREE"
    elif "P3" in raw_upper or "URGENCE DIFFEREE" in raw_upper or "URGENCE DIFFÉRÉE" in raw_upper:
        priority = "P3 - URGENCE DIFFEREE"
    else:
        priority = "NON DETERMINE"
    return priority, raw.strip(), ""


# ---------------------------- Inference ----------------------------


def _generate(prompt: str) -> str:
    kind = _backend["kind"]
    inf = config["inference"]

    if kind == "vllm":
        from vllm import SamplingParams  # type: ignore

        params = SamplingParams(
            temperature=inf["temperature"],
            top_p=inf["top_p"],
            max_tokens=inf["max_tokens"],
            repetition_penalty=inf["repetition_penalty"],
        )
        out = _backend["engine"].generate([prompt], params)
        return out[0].outputs[0].text

    if kind == "transformers":
        import torch

        tok = _backend["tokenizer"]
        model = _backend["engine"]
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=config["vllm"]["max_model_len"])
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=inf["max_tokens"],
                temperature=inf["temperature"],
                top_p=inf["top_p"],
                repetition_penalty=inf["repetition_penalty"],
                do_sample=True,
                pad_token_id=tok.pad_token_id,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Mock: reponse deterministe pour tests/demo
    return (
        "P2 - URGENCE MODEREE\n"
        "Justification: reponse de test (mode mock, aucun modele charge).\n"
        "Recommandations: consulter un professionnel de sante."
    )


# ---------------------------- Endpoints ----------------------------


@app.get("/")
async def root():
    return {
        "service": "CHSA - Agent IA Triage Medical",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "backend": _backend["kind"],
        "model_id": MODEL_ID or None,
        "auth_enabled": bool(API_KEY),
    }


@app.post("/triage", response_model=TriageResponse, dependencies=[Depends(require_api_key)])
async def triage(req: TriageRequest):
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    prompt = build_prompt(req)

    t0 = time.perf_counter()
    try:
        raw = _generate(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erreur inference")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000.0

    priority, explanation, recommendations = parse_triage_response(raw)

    response = TriageResponse(
        interaction_id=interaction_id,
        timestamp=timestamp,
        patient_id=req.patient_id,
        priority_level=priority,
        explanation=explanation,
        recommendations=recommendations,
        raw_response=raw,
        latency_ms=round(latency_ms, 2),
        backend=_backend["kind"],
    )

    audit_log.append(
        {
            "interaction_id": interaction_id,
            "timestamp": timestamp,
            "patient_id": req.patient_id,
            "symptoms": req.symptoms,
            "priority_level": priority,
            "latency_ms": response.latency_ms,
            "backend": _backend["kind"],
        }
    )
    logger.info("Triage %s -> %s (%.0f ms)", interaction_id, priority, latency_ms)
    return response


@app.get("/audit", dependencies=[Depends(require_api_key)])
async def get_audit_log(limit: int = 100):
    return {
        "total_interactions": len(audit_log),
        "entries": audit_log[-limit:],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.deployment.serve:app",
        host="0.0.0.0",  # noqa: S104
        port=int(os.environ.get("PORT", "7860")),
        log_level="info",
    )
