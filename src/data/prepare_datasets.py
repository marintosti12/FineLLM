"""
Semaine 1 - Preparation et structuration des datasets medicaux.

Script CLI pour reproduire la preparation des donnees.
Le notebook 01_data_preparation.ipynb est la reference pour le detail des etapes.

Datasets sources :
- MediQAl (QCM medicaux francophones - ANR-MALADES)
- FrenchMedMCQA (QCM medicaux francophones - qanastek)
- MedQuAD (Medical Question Answering Dataset - lavita)
- UltraMedical-Preference (paires de preferences pour DPO - TsinghuaC3I)

Produit :
- Dataset SFT : ~5000 paires instruction-reponse (data/sft/)
- Dataset DPO : paires chosen/rejected (data/dpo/)
- Splits : train (80%) / eval (10%) / test (10%)
"""

import hashlib
import json
import logging
import random
from pathlib import Path

import yaml
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ANSWER_KEYS = ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"]
ANSWER_LABELS = ["A", "B", "C", "D", "E"]

SYSTEM_PROMPT = "Vous etes un assistant medical specialise dans le triage des urgences."
SEED = 42


def load_config(config_path: str = "configs/sft_config.yaml") -> dict:
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def save_jsonl(data: list[dict], output_path: Path) -> None:
    """Sauvegarde les donnees au format JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Sauvegarde de {len(data)} exemples dans {output_path}")


def clean_text(text: str) -> str:
    """Nettoyage basique du texte."""
    if not text:
        return ""
    text = " ".join(text.split())
    return text.strip()


def _format_mcq_instruction(item: dict, with_clinical_case: bool = False) -> str:
    """Formate une question MCQ avec ses choix en texte."""
    parts = []
    if with_clinical_case and item.get("clinical_case"):
        parts.append(f"Cas clinique : {item['clinical_case']}")
    parts.append(f"Question : {item['question']}")
    parts.append("\nChoix possibles :")
    for key, label in zip(ANSWER_KEYS, ANSWER_LABELS):
        answer = item.get(key)
        if answer:
            parts.append(f"  {label}. {answer}")
    return "\n".join(parts)


def _format_mcq_response(item: dict) -> str:
    """Formate la reponse correcte d'un MCQ."""
    correct = item.get("correct_answers", "")
    if isinstance(correct, list):
        labels = [ANSWER_LABELS[i] for i in correct if i < len(ANSWER_LABELS)]
        correct_labels = labels
    else:
        correct_labels = [c.strip() for c in str(correct).split(",")]

    response_parts = []
    for label in correct_labels:
        idx = ANSWER_LABELS.index(label) if label in ANSWER_LABELS else -1
        if idx >= 0:
            key = ANSWER_KEYS[idx]
            answer_text = item.get(key, "")
            response_parts.append(f"{label}. {answer_text}")

    if response_parts:
        return "La reponse correcte est : " + " ; ".join(response_parts)
    return f"La reponse correcte est : {correct}"


def format_sft_example(item_id: str, source: str, langue: str, type_question: str,
                       sujet_medical: str, niveau_confiance: float,
                       instruction: str, response: str) -> dict:
    """Formate un exemple SFT avec metadonnees."""
    return {
        "id": item_id,
        "source": source,
        "langue": langue,
        "type_question": type_question,
        "sujet_medical": sujet_medical,
        "niveau_confiance": niveau_confiance,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
    }


def deduplicate(data: list[dict], key_field: str = "messages") -> list[dict]:
    """Supprime les doublons exacts par hash."""
    seen = set()
    unique = []
    for item in data:
        content = json.dumps(item[key_field], ensure_ascii=False)
        h = hashlib.md5(content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(item)
    removed = len(data) - len(unique)
    if removed:
        logger.info(f"Deduplication: {removed} doublons supprimes")
    return unique


def split_dataset(data: list[dict], train_ratio: float = 0.8,
                  val_ratio: float = 0.1, seed: int = SEED) -> tuple:
    """Split un dataset en train/val/test (80/10/10)."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def load_mediqal() -> list[dict]:
    """Charge et formate le dataset MediQAl."""
    logger.info("Chargement de MediQAl...")
    examples = []
    try:
        for cfg in ["mcqu", "mcqm"]:
            qtype = "mcq_unique" if cfg == "mcqu" else "mcq_multiple"
            ds = load_dataset("ANR-MALADES/MediQAl", cfg, trust_remote_code=True, split="train")
            for i, item in enumerate(ds):
                instruction = _format_mcq_instruction(item, with_clinical_case=True)
                response = _format_mcq_response(item)
                subject = item.get("subject", "medecine generale")
                examples.append(format_sft_example(
                    item_id=f"mediqal_{cfg}_{i:05d}",
                    source="ANR-MALADES/MediQAl",
                    langue="fr",
                    type_question=qtype,
                    sujet_medical=subject,
                    niveau_confiance=1.0,
                    instruction=instruction,
                    response=response,
                ))
            logger.info(f"MediQAl/{cfg}: {len(ds)} exemples")

        ds_oeq = load_dataset("ANR-MALADES/MediQAl", "oeq", trust_remote_code=True, split="test")
        for i, item in enumerate(ds_oeq):
            question = item.get("question", "")
            answer = item.get("answer", "")
            if question and answer:
                instruction = question
                if item.get("clinical_case"):
                    instruction = f"Cas clinique : {item['clinical_case']}\n\nQuestion : {question}"
                subject = item.get("subject", "medecine generale")
                examples.append(format_sft_example(
                    item_id=f"mediqal_oeq_{i:05d}",
                    source="ANR-MALADES/MediQAl",
                    langue="fr",
                    type_question="open_question",
                    sujet_medical=subject,
                    niveau_confiance=0.9,
                    instruction=instruction,
                    response=answer,
                ))
        logger.info(f"MediQAl/oeq: {len(ds_oeq)} exemples")
        logger.info(f"MediQAl total: {len(examples)} exemples")
        return examples
    except Exception as e:
        logger.warning(f"Erreur chargement MediQAl: {e}")
        return examples


def load_french_med_mcqa() -> list[dict]:
    """Charge et formate le dataset FrenchMedMCQA."""
    logger.info("Chargement de FrenchMedMCQA...")
    try:
        ds = load_dataset("qanastek/frenchmedmcqa", trust_remote_code=True, split="train")
        examples = []
        for i, item in enumerate(ds):
            instruction = _format_mcq_instruction(item)
            response = _format_mcq_response(item)
            n_correct = item.get("number_correct_answers", 0)
            qtype = "mcq_unique" if n_correct == 0 else "mcq_multiple"
            examples.append(format_sft_example(
                item_id=f"frenchmedmcqa_{i:05d}",
                source="qanastek/frenchmedmcqa",
                langue="fr",
                type_question=qtype,
                sujet_medical="pharmacie",
                niveau_confiance=1.0,
                instruction=instruction,
                response=response,
            ))
        logger.info(f"FrenchMedMCQA: {len(examples)} exemples")
        return examples
    except Exception as e:
        logger.warning(f"Erreur chargement FrenchMedMCQA: {e}")
        return []


def load_medquad() -> list[dict]:
    """Charge et formate le dataset MedQuAD."""
    logger.info("Chargement de MedQuAD...")
    try:
        ds = load_dataset("lavita/MedQuAD", trust_remote_code=True, split="train")
        examples = []
        for i, item in enumerate(ds):
            question = clean_text(item.get("question", ""))
            answer = clean_text(item.get("answer", ""))
            if question and answer:
                subject = item.get("focus_area", item.get("source", "general"))
                examples.append(format_sft_example(
                    item_id=f"medquad_{i:05d}",
                    source="lavita/MedQuAD",
                    langue="en",
                    type_question="open_question",
                    sujet_medical=subject,
                    niveau_confiance=0.9,
                    instruction=question,
                    response=answer,
                ))
        logger.info(f"MedQuAD: {len(examples)} exemples")
        return examples
    except Exception as e:
        logger.warning(f"Erreur chargement MedQuAD: {e}")
        return []


def load_ultramedical_preference() -> list[dict]:
    """Charge le dataset UltraMedical-Preference pour le DPO."""
    logger.info("Chargement de UltraMedical-Preference...")
    try:
        ds = load_dataset("TsinghuaC3I/UltraMedical-Preference", trust_remote_code=True, split="train")
        examples = []
        for i, item in enumerate(ds):
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", [])
            rejected = item.get("rejected", [])
            if prompt and chosen and rejected:
                examples.append({
                    "id": f"ultramedical_{i:05d}",
                    "source": "TsinghuaC3I/UltraMedical-Preference",
                    "langue": "en",
                    "niveau_confiance": 0.8,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })
        logger.info(f"UltraMedical-Preference: {len(examples)} exemples")
        return examples
    except Exception as e:
        logger.warning(f"Erreur chargement UltraMedical-Preference: {e}")
        return []


def prepare_sft_dataset(max_samples: int = 5000) -> None:
    """Prepare le dataset SFT en agregeant les sources medicales."""
    logger.info("=== Preparation du dataset SFT ===")

    all_examples = []
    all_examples.extend(load_mediqal())
    all_examples.extend(load_french_med_mcqa())
    all_examples.extend(load_medquad())

    logger.info(f"Total exemples bruts: {len(all_examples)}")

    # Deduplication
    all_examples = deduplicate(all_examples, key_field="messages")

    # Echantillonnage
    if len(all_examples) > max_samples:
        random.seed(SEED)
        all_examples = random.sample(all_examples, max_samples)
        logger.info(f"Echantillonnage: {max_samples} exemples retenus")

    # Split train/val/test (80/10/10)
    train_data, val_data, test_data = split_dataset(all_examples)

    save_jsonl(train_data, PROJECT_ROOT / "data" / "sft" / "train.jsonl")
    save_jsonl(val_data, PROJECT_ROOT / "data" / "sft" / "eval.jsonl")
    save_jsonl(test_data, PROJECT_ROOT / "data" / "sft" / "test.jsonl")

    logger.info(f"SFT dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")


def prepare_dpo_dataset() -> None:
    """Prepare le dataset DPO."""
    logger.info("=== Preparation du dataset DPO ===")

    examples = load_ultramedical_preference()

    # Split train/val/test (80/10/10)
    train_data, val_data, test_data = split_dataset(examples)

    save_jsonl(train_data, PROJECT_ROOT / "data" / "dpo" / "train.jsonl")
    save_jsonl(val_data, PROJECT_ROOT / "data" / "dpo" / "eval.jsonl")
    save_jsonl(test_data, PROJECT_ROOT / "data" / "dpo" / "test.jsonl")

    logger.info(f"DPO dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")


if __name__ == "__main__":
    config = load_config()
    max_samples = config.get("data", {}).get("max_samples", 5000)

    prepare_sft_dataset(max_samples=max_samples)
    prepare_dpo_dataset()

    logger.info("Preparation des datasets terminee.")
