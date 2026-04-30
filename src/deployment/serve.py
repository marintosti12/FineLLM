"""
Semaine 4 - Deploiement de l'endpoint de demonstration.

API FastAPI de triage medical. Backend d'inference:
- vLLM si USE_VLLM=1 (GPU, prod / HF Space GPU)
- transformers sinon (CPU, fallback / tests CI)
- mode echo si USE_VLLM=0 ET MODEL_ID vide (tests unitaires)

Support adapters LoRA (PEFT) via ADAPTER_ID si le MODEL_ID pointe sur le base
et qu'on souhaite appliquer un adapter LoRA.

Variables d'environnement:
- MODEL_ID        : id du modele HF Hub (ex: Qwen/Qwen3-1.7B-Base ou Marintosti/chsa-triage-merged)
- ADAPTER_ID      : chemin/id d'un adapter PEFT a appliquer sur MODEL_ID (optionnel)
- USE_VLLM        : "1" pour charger vLLM, "0" pour transformers/mock (default: "1")
- API_KEY         : si defini, exige le header X-API-Key
- PORT            : port d'ecoute (default: 7860 - requis par HF Spaces)
- AUDIT_LOG_PATH  : chemin du fichier JSONL append-only pour la tracabilite RGPD
                    (default: audit/audit.jsonl). Mettre vide pour desactiver.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = Path(os.environ.get("DEPLOY_CONFIG", PROJECT_ROOT / "configs" / "deployment_config.yaml"))

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_ID = os.environ.get("MODEL_ID", "").strip()
ADAPTER_ID = os.environ.get("ADAPTER_ID", "").strip()
USE_VLLM = os.environ.get("USE_VLLM", "1") == "1"
API_KEY = os.environ.get("API_KEY", "").strip()
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", str(PROJECT_ROOT / "audit" / "audit.jsonl")).strip()

# Etat global du backend
_backend: dict[str, Any] = {"kind": "mock", "engine": None, "tokenizer": None}
audit_log: list[dict] = []
_audit_lock = threading.Lock()


def _persist_audit_entry(entry: dict) -> None:
    """Append-only JSONL pour la tracabilite RGPD (relisible apres redemarrage).

    Sans persistance, l'audit log est perdu au redeploiement, ce qui n'est pas
    acceptable pour un audit medical.
    """
    if not AUDIT_LOG_PATH:
        return
    try:
        path = Path(AUDIT_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False)
        with _audit_lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as exc:
        logger.warning("Echec persistance audit (%s): %s", AUDIT_LOG_PATH, exc)


def _load_audit_history() -> None:
    """Recharge l'historique au demarrage pour exposer un audit complet via /audit."""
    if not AUDIT_LOG_PATH:
        return
    path = Path(AUDIT_LOG_PATH)
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    audit_log.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        logger.info("Audit log charge: %d entrees depuis %s", len(audit_log), path)
    except OSError as exc:
        logger.warning("Echec lecture audit (%s): %s", path, exc)


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


def _load_transformers(model_id: str, adapter_id: str = "") -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Tokenizer : si un adapter est fourni, priviliegier son tokenizer (contient le bon chat_template)
    tok_src = adapter_id or model_id
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    if adapter_id:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, adapter_id)
        logger.info("Adapter LoRA applique: %s", adapter_id)

    model.eval()
    _backend["engine"] = model
    _backend["tokenizer"] = tok
    _backend["kind"] = "transformers"
    logger.info("Backend transformers (%s) charge: %s", device, model_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_audit_history()
    if not MODEL_ID:
        logger.warning("MODEL_ID non defini - mode mock (echo) active")
    else:
        try:
            if USE_VLLM:
                _load_vllm(MODEL_ID)
            else:
                _load_transformers(MODEL_ID, ADAPTER_ID)
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

# CORS : autoriser les appels depuis la page de demo (file://, github pages, etc.)
# Pour un POC de demonstration, on autorise toutes les origines. En production,
# restreindre a la liste des domaines hospitaliers autorises.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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
    """Construit la requete patient en texte brut (pour vLLM)."""
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
    parts.append("Veuillez evaluer le niveau de priorite (P1, P2 ou P3) et fournir vos recommandations.")
    return "\n".join(parts)


def build_chat_messages(req: TriageRequest) -> list[dict]:
    """Format ChatML (aligne avec le SFT/DPO)."""
    return [
        {"role": "system", "content": config["triage"]["system_prompt"]},
        {"role": "user", "content": build_prompt(req)},
    ]


# --- Nettoyage des artefacts de generation (preifixe multilingue, leak MCQ) ---

# Caracteres autorises en sortie : latin + ponctuation + espaces + chiffres + accents
# Tout ce qui n'est pas dans cet intervalle est considere comme un artefact de generation
# (chinois, thai, arabe, etc. que Qwen3-Base peut generer en premier token).
_NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ɏ -⁯₠-⃏∀-⋿]+")
_STOP_PATTERNS = [
    "\nQuestion :",
    "\nQuestion:",
    "\nChoix possibles",
    "\nCas clinique :",
    "\n<|im_end|>",
    "<|im_end|>",
    "<|im_start|>",
    "\nuser\n",
    "\nUSER\n",
]


def clean_response(raw: str) -> str:
    """Strip les artefacts du modele Base + coupe au premier leak de format MCQ."""
    # 1. Strip les caracteres non-latins au debut (prefixe thai/chinois/arabe)
    raw = _NON_LATIN.sub("", raw)
    # 2. Couper au premier "leak" du format training (nouvelle question, tokens chat, etc.)
    for stop in _STOP_PATTERNS:
        idx = raw.find(stop)
        if idx > 0:
            raw = raw[:idx]
    return raw.strip()


_PRIORITY_LABELS = {
    "P1": "P1 - URGENCE MAXIMALE",
    "P2": "P2 - URGENCE MODEREE",
    "P3": "P3 - URGENCE DIFFEREE",
}

# Mention explicite emise par le modele apres SFT/DPO. On match en priorite
# ces patterns pour eviter les faux positifs sur des mentions hypothetiques
# (ex : "qui mettraient en danger le pronostic vital").
_EXPLICIT_PRIORITY_RE = re.compile(
    r"(?:NIVEAU\s+DE\s+)?PRIORIT[EÉ]\s*[:\-]?\s*\**\s*(P[123])\b",
    re.IGNORECASE,
)
# Fallback : premiere occurrence isolee de Px dans la reponse.
_PX_RE = re.compile(r"\bP([123])\b")


def parse_triage_response(raw: str) -> tuple[str, str, str]:
    """Nettoie puis detecte le niveau de priorite dans la reponse.

    Strategie :
    1. Mention explicite "Niveau de Priorite : Px" / "Priorite : Px" (sortie SFT/DPO standard).
    2. Premiere occurrence isolee de P1 / P2 / P3 dans la reponse.
    3. Fallback heuristique sur des mots-cles non ambigus (uniquement si aucune mention explicite).
    """
    cleaned = clean_response(raw)

    # 1. Mention explicite (case-insensitive)
    m = _EXPLICIT_PRIORITY_RE.search(cleaned)
    if m:
        return _PRIORITY_LABELS[m.group(1).upper()], cleaned, ""

    # 2. Premiere occurrence isolee de Px
    m = _PX_RE.search(cleaned)
    if m:
        return _PRIORITY_LABELS["P" + m.group(1)], cleaned, ""

    # 3. Fallback mots-cles (utilise uniquement si le modele n'a pas formate la priorite).
    #    On retire "PRONOSTIC VITAL" qui genere des faux positifs en contexte hypothetique
    #    ("qui mettrait en danger le pronostic vital").
    upper = cleaned.upper()
    if "URGENCE MAXIMALE" in upper:
        priority = "P1 - URGENCE MAXIMALE"
    elif "URGENCE MODEREE" in upper or "URGENCE MODÉRÉE" in upper:
        priority = "P2 - URGENCE MODEREE"
    elif "URGENCE DIFFEREE" in upper or "URGENCE DIFFÉRÉE" in upper:
        priority = "P3 - URGENCE DIFFEREE"
    else:
        priority = "NON DETERMINE"

    return priority, cleaned, ""


# ---------------------------- Inference ----------------------------


def _generate(req: TriageRequest) -> str:
    kind = _backend["kind"]
    inf = config["inference"]
    messages = build_chat_messages(req)

    if kind == "vllm":
        from vllm import SamplingParams  # type: ignore

        # Applique le chat template Qwen3 via le tokenizer de vLLM
        llm = _backend["engine"]
        tok = llm.get_tokenizer()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        params = SamplingParams(
            temperature=inf["temperature"],
            top_p=inf["top_p"],
            max_tokens=inf["max_tokens"],
            repetition_penalty=inf["repetition_penalty"],
            stop=["<|im_end|>", "\nQuestion :", "\nChoix possibles"],
        )
        out = llm.generate([prompt], params)
        return out[0].outputs[0].text

    if kind == "transformers":
        import torch

        tok = _backend["tokenizer"]
        model = _backend["engine"]

        # Chat template ChatML (aligne avec l'entrainement SFT/DPO)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=inf["max_tokens"],
                temperature=inf["temperature"],
                top_p=inf["top_p"],
                top_k=50,
                repetition_penalty=inf["repetition_penalty"],
                no_repeat_ngram_size=4,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
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

    t0 = time.perf_counter()
    try:
        raw = _generate(req)
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

    entry = {
        "interaction_id": interaction_id,
        "timestamp": timestamp,
        "patient_id": req.patient_id,
        "symptoms": req.symptoms,
        "priority_level": priority,
        "latency_ms": response.latency_ms,
        "backend": _backend["kind"],
    }
    audit_log.append(entry)
    _persist_audit_entry(entry)
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
