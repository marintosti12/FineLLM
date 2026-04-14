---
title: CHSA Triage API
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Agent IA de triage medical - POC CHSA
---

# CHSA - Agent IA Triage Medical

POC d'un agent de triage medical pour le Centre Hospitalier Saint-Aurelien (CHSA).
Modele Qwen3-1.7B fine-tune (SFT + LoRA) puis aligne par DPO sur corpus medical FR/EN.

## Endpoints

| Methode | Route      | Description                                  |
|---------|------------|----------------------------------------------|
| GET     | `/`        | Infos service                                |
| GET     | `/health`  | Etat du backend (vllm / transformers / mock) |
| POST    | `/triage`  | Triage a partir des symptomes                |
| GET     | `/audit`   | Journal d'audit des interactions             |
| GET     | `/docs`    | Swagger UI                                   |

Si `API_KEY` est defini (secret du Space), les routes `/triage` et `/audit`
exigent le header `X-API-Key`.

## Exemple

```bash
curl -X POST https://marintosti-chsa-triage-api.hf.space/triage \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "symptoms": "Douleur thoracique irradiant au bras gauche, sueurs",
    "age": 58,
    "sex": "M",
    "vital_signs": {"fc": 112, "ta": "150/95", "spo2": 94}
  }'
```

## Variables d'environnement

| Var          | Defaut                                   | Role                               |
|--------------|------------------------------------------|------------------------------------|
| `MODEL_ID`   | *(vide -> mode mock)*                    | Repo HF du modele fine-tune        |
| `USE_VLLM`   | `1`                                      | `0` pour fallback transformers CPU |
| `API_KEY`    | *(vide -> auth desactivee)*              | Header `X-API-Key` exige si defini |
| `PORT`       | `7860`                                   | Port d'ecoute (fixe par HF Spaces) |

## Limites d'usage

Outil d'aide a la decision uniquement. Ne remplace pas un jugement clinique.
Resultats indicatifs, a valider systematiquement par un professionnel de sante.
