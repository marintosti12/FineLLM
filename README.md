# FineLLM - Agent IA de Triage Medical

POC d'un agent IA de triage medical pour le Centre Hospitalier Saint-Aurelien (CHSA).

## Objectif

Developper un agent intelligent qui assiste le personnel soignant dans le triage initial des patients aux urgences :
- Collecte adaptive des symptomes
- Evaluation du niveau de priorite (P1/P2/P3)
- Explications et recommandations cliniques
- Tracabilite des interactions pour audit

## Architecture

```
FineLLM/
├── configs/                 # Configurations YAML (SFT, DPO, deploiement)
├── data/                    # Datasets medicaux
│   ├── sft/                 # Paires instruction-reponse (5000 exemples)
│   └── dpo/                 # Paires chosen/rejected
├── src/
│   ├── data/                # Preparation des datasets
│   ├── training/            # Scripts SFT et DPO
│   ├── evaluation/          # Metriques et benchmarks
│   ├── deployment/          # API FastAPI + vLLM
│   └── utils/               # Utilitaires
├── notebooks/               # Exploration et analyse
├── tests/                   # Tests unitaires
├── scripts/                 # Scripts utilitaires
└── docs/                    # Documentation technique
```

## Modele

**Base** : Qwen3-1.7B-Base
- Phase 1 : Fine-tuning supervise (SFT) avec LoRA
- Phase 2 : Alignement par preferences (DPO)
- Deploiement : vLLM pour inference optimisee

## Datasets

| Source | Langue | Usage |
|--------|--------|-------|
| MediQA | EN | SFT |
| FrenchMedMCQA | FR | SFT |
| MedQuAD | EN | SFT |
| UltraMedical-Preference | EN | DPO |

## Installation

```bash
poetry install
```

## Utilisation

### 1. Preparation des donnees
```bash
python -m src.data.prepare_datasets
```

### 2. Fine-tuning SFT
```bash
python -m src.training.sft_trainer
```

### 3. Alignement DPO
```bash
python -m src.training.dpo_trainer
```

### 4. Evaluation
```bash
python -m src.evaluation.evaluate_model --model_path outputs/dpo/final
```

### 5. Deploiement
```bash
uvicorn src.deployment.serve:app --host 0.0.0.0 --port 8000
```

## Planning

| Semaine | Objectif |
|---------|----------|
| S1 | Preparation et structuration des donnees |
| S2 | Fine-tuning SFT avec LoRA |
| S3 | Alignement DPO |
| S4 | Deploiement vLLM et evaluation finale |
