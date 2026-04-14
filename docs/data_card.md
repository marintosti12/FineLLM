# Data Card — Corpus Médical Bilingue CHSA

## Vue d'ensemble

Corpus médical bilingue (français/anglais) destiné au fine-tuning SFT et à l'alignement DPO d'un modèle de triage médical pour le Centre Hospitalier Saint-Aurélien.

## Sources de données

| Dataset | Éditeur | Langue | Type | Exemples | Licence | Usage |
|---------|---------|--------|------|----------|---------|-------|
| [MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | ANR-MALADES | FR | QCM + questions ouvertes | ~20 849 | Apache 2.0 | SFT |
| [FrenchMedMCQA](https://huggingface.co/datasets/qanastek/frenchmedmcqa) | qanastek | FR | QCM pharmacie | ~2 171 | Apache 2.0 | SFT |
| [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) | lavita | EN | Q&A médicales | ~47 441 | - | SFT |
| [UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | TsinghuaC3I | EN | Paires préférentielles | ~109 353 | Apache 2.0 | DPO |

## Schéma des métadonnées

Chaque exemple du corpus processed contient :

| Champ | Type | Description |
|-------|------|-------------|
| `id` | string | Identifiant unique (`{source}_{config}_{index}`) |
| `source` | string | Nom HuggingFace du dataset d'origine |
| `langue` | string | `fr` ou `en` |
| `type_question` | string | `mcq_unique`, `mcq_multiple`, `open_question`, `preference` |
| `sujet_medical` | string | Spécialité médicale (ex: "Infectious Diseases", "pharmacie") |
| `niveau_confiance` | float | 1.0 (QCM), 0.9 (question ouverte), 0.8 (préférence DPO) |
| `instruction` / `response` | string | Paire SFT formatée |
| `prompt` / `chosen` / `rejected` | string/list | Triplet DPO |

## Processus RGPD et anonymisation

### Justification

Bien que les datasets utilisés soient publics et ne contiennent pas de données patients réelles (QCM d'examens, Q&A médicales génériques), une anonymisation systématique est appliquée pour :

1. **Conformité RGPD** : démontrer le processus d'anonymisation dans le POC
2. **Précaution** : les cas cliniques de MediQAl peuvent contenir des noms fictifs de patients
3. **Reproductibilité** : établir le pipeline pour les futures données réelles du CHSA

### Outil utilisé

**Microsoft Presidio** (v2.2+) — outil open source de détection et masquage de données sensibles.

- **AnalyzerEngine** : détection d'entités sensibles via NLP (spaCy)
- **AnonymizerEngine** : remplacement par des placeholders

### Modèles linguistiques

| Langue | Modèle spaCy | Usage |
|--------|-------------|-------|
| Français | `fr_core_news_md` | MediQAl, FrenchMedMCQA |
| Anglais | `en_core_web_sm` | MedQuAD, UltraMedical |

### Entités détectées et stratégie de masquage

| Entité | Placeholder | Stratégie |
|--------|-------------|-----------|
| PERSON | `<PERSONNE>` | replace |
| PHONE_NUMBER | `<TELEPHONE>` | replace |
| EMAIL_ADDRESS | `<EMAIL>` | replace |
| LOCATION | `<LIEU>` | replace |
| DATE_TIME | `<DATE>` | replace |
| NRP | `<NRP>` | replace |
| Autres | `<PII>` | replace |

### Rapport d'anonymisation

Un rapport JSON (`data/raw/anonymization_report.json`) est généré automatiquement avec :
- Nombre d'entités détectées par type et par fichier
- Statistiques globales
- Ce rapport sert de trace d'audit pour la conformité RGPD.

### Données personnelles résiduelles

Après anonymisation, un contrôle qualité est effectué pour vérifier qu'aucune donnée personnelle identifiable ne subsiste. Le seuil de détection est fixé à 0.4 (score Presidio).

## Splits

| Split | Ratio | Usage |
|-------|-------|-------|
| Train | 80% | Entraînement SFT/DPO |
| Validation | 10% | Évaluation pendant l'entraînement |
| Test | 10% | Évaluation finale (jamais vu pendant l'entraînement) |

Seed de randomisation : `42` (reproductibilité garantie).

## Transformations appliquées

1. **Téléchargement brut** → `data/raw/` (JSONL, données originales)
2. **Anonymisation RGPD** → `data/raw/` (écrasement, entités masquées)
3. **Nettoyage + métadonnées** → `data/processed/` (format unifié, déduplication)
4. **Formatage + splits** → `data/sft/` et `data/dpo/` (format ChatML / DPO)

Chaque étape est documentée et reproductible via le notebook `notebooks/01_data_preparation.ipynb`.
