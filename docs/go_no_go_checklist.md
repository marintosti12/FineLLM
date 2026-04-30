# Checklist Go / No-Go — Mise en production conditionnelle

**Projet :** CHSA — Agent IA de Triage Médical
**Phase :** Passage du POC à un déploiement pilote (Phase 3)
**Statut :** À valider par la Direction Innovation Médicale + DSI + DPO + équipe urgences

> **Règle de décision :** un seul item bloquant en **NO-GO** suffit à reporter la mise en production.
> Les items **WARN** doivent faire l'objet d'un plan d'atténuation documenté avant le démarrage.

---

## 1. Modèle et qualité clinique

| # | Critère | Cible | Statut POC | Bloquant ? |
|---|---|---|---|---|
| 1.1 | Accuracy triage P1/P2/P3 sur jeu de test interne CHSA | ≥ 80 % | À mesurer en Phase 3 | 🔴 NO-GO si < 70 % |
| 1.2 | Sensibilité sur cas P1 (rappel) | ≥ 95 % | À mesurer en Phase 3 | 🔴 NO-GO si < 90 % |
| 1.3 | Taux de sous-priorisation (P1 → P2/P3) | < 2 % | À mesurer en Phase 3 | 🔴 NO-GO sinon |
| 1.4 | Taux d'hallucination clinique (cas adverses) | < 1 % | À mesurer | 🟡 WARN |
| 1.5 | Validation par panel médical (≥ 3 urgentistes) | OK sur 200 cas | Non fait (POC) | 🔴 NO-GO |
| 1.6 | Comparaison Base / SFT / DPO disponible | Documentée | Section 5.5 du rapport | 🟢 |

---

## 2. Performance et infrastructure

| # | Critère | Cible | Statut POC | Bloquant ? |
|---|---|---|---|---|
| 2.1 | Latence P95 (inférence serveur) | < 1500 ms | À mesurer en GPU réel | 🔴 NO-GO si > 3000 ms |
| 2.2 | Throughput soutenu | ≥ 5 req/s | À mesurer | 🟡 WARN |
| 2.3 | Cold start | < 30 s | À mesurer | 🟡 WARN |
| 2.4 | Haute disponibilité (≥ 2 répliques + LB) | OK | 1 instance HF Space (POC) | 🔴 NO-GO en prod |
| 2.5 | Plan de capacité dimensionné (pic d'urgences) | Documenté | À faire | 🔴 NO-GO |
| 2.6 | Bench de charge réaliste exécuté | OK, archivé | `scripts/bench_latency.py` | 🟢 |

---

## 3. Sécurité et conformité RGPD

| # | Critère | Cible | Statut POC | Bloquant ? |
|---|---|---|---|---|
| 3.1 | Authentification API (clé / token) | Activée | `X-API-Key` (env var) | 🟢 |
| 3.2 | Rotation des clés API (procédure documentée) | OK | À faire | 🟡 WARN |
| 3.3 | Rate limiting / anti-DoS | Activé | Non implémenté | 🟡 WARN |
| 3.4 | TLS 1.2+ obligatoire | OK | Géré par HF Spaces / reverse proxy | 🟢 |
| 3.5 | Audit log persistant (RGPD art. 30) | OK | JSONL append-only (`AUDIT_LOG_PATH`) | 🟢 |
| 3.6 | Anonymisation des données d'entraînement | Documentée | `data/raw/anonymization_report.json` | 🟢 |
| 3.7 | AIPD (Analyse d'Impact RGPD) validée | OK | À conduire avant prod | 🔴 NO-GO |
| 3.8 | Déclaration CNIL (donnée de santé) | Faite | À faire | 🔴 NO-GO |
| 3.9 | Hébergement HDS (Hébergeur Données Santé) | Certifié | HF Spaces ≠ HDS — migration on-premise / OVH HDS requise | 🔴 NO-GO en prod |
| 3.10 | Pas de logs PII en clair (en dehors de l'audit) | OK | À auditer | 🟡 WARN |

---

## 4. Observabilité et exploitation

| # | Critère | Cible | Statut POC | Bloquant ? |
|---|---|---|---|---|
| 4.1 | Healthcheck `/health` | OK | Présent | 🟢 |
| 4.2 | Métriques exportées (Prometheus / OpenTelemetry) | OK | À ajouter | 🟡 WARN |
| 4.3 | Alerting sur erreurs ≥ 1 % / 5 min | Configuré | À faire | 🔴 NO-GO |
| 4.4 | Alerting sur latence p95 > seuil | Configuré | À faire | 🟡 WARN |
| 4.5 | Tableau de bord opérationnel (Grafana ou équivalent) | Disponible | À créer | 🟡 WARN |
| 4.6 | Logs centralisés (ELK / Loki) | OK | À mettre en place | 🟡 WARN |
| 4.7 | Procédure d'astreinte définie | Documentée | À écrire avec la DSI | 🔴 NO-GO |

---

## 5. CI/CD et reproductibilité

| # | Critère | Cible | Statut POC | Bloquant ? |
|---|---|---|---|---|
| 5.1 | Build Docker reproductible | OK | `Dockerfile` + CI sanity build | 🟢 |
| 5.2 | Tests automatisés (CI verte) | OK | `tests/test_serve.py` (8 tests) | 🟢 |
| 5.3 | Pipeline de déploiement sans intervention manuelle | OK | `.github/workflows/deploy-space.yml` | 🟢 |
| 5.4 | Rollback automatique en cas d'échec healthcheck | OK | À implémenter (HF Space ≠ rollback natif) | 🟡 WARN |
| 5.5 | Versioning du modèle (HF Hub avec tag) | OK | À formaliser | 🟢 |
| 5.6 | Secrets gérés hors du code | OK | `secrets.HF_TOKEN`, `API_KEY` env | 🟢 |

---

## 6. Validation clinique et gouvernance

| # | Critère | Cible | Statut POC | Bloquant ? |
|---|---|---|---|---|
| 6.1 | Mode "assistance uniquement" (pas d'autonomie) | Garanti dans l'UI | À valider en intégration SIH | 🔴 NO-GO |
| 6.2 | Validation humaine systématique sur P1 | Workflow garanti | À cabler dans le SIH | 🔴 NO-GO |
| 6.3 | Comité d'éthique interne consulté | Avis favorable | À organiser | 🔴 NO-GO |
| 6.4 | Limites d'usage documentées pour les soignants | Disponibles | `docs/usage_limits.md` (à créer) | 🔴 NO-GO |
| 6.5 | Procédure de signalement d'erreur clinique | Définie | À écrire | 🔴 NO-GO |
| 6.6 | Plan de formation utilisateurs (IDE, médecins) | Préparé | À faire | 🔴 NO-GO |
| 6.7 | Marquage CE-MD classe IIa lancé | OK | À initier | 🟡 WARN (non bloquant pour pilote interne) |

---

## 7. Plan de rollback et continuité de service

| # | Critère | Cible | Statut POC | Bloquant ? |
|---|---|---|---|---|
| 7.1 | Procédure de désactivation rapide (< 5 min) | OK | Variable d'env / kill switch | 🟡 WARN |
| 7.2 | Mode dégradé (sans agent IA) testé | OK | Tri humain seul = état actuel | 🟢 |
| 7.3 | Sauvegarde de l'audit log (off-site) | OK | À mettre en place | 🔴 NO-GO |
| 7.4 | RPO / RTO définis | Documentés | À chiffrer avec la DSI | 🟡 WARN |

---

## Synthèse de décision

| Catégorie | Items 🟢 | Items 🟡 | Items 🔴 |
|---|---|---|---|
| 1. Modèle / clinique | 1 | 1 | 4 |
| 2. Performance / infra | 1 | 3 | 2 |
| 3. Sécurité / RGPD | 5 | 3 | 3 |
| 4. Observabilité | 1 | 4 | 2 |
| 5. CI/CD | 5 | 1 | 0 |
| 6. Validation clinique | 0 | 1 | 6 |
| 7. Rollback | 1 | 2 | 1 |
| **TOTAL** | **14** | **15** | **18** |

> **Conclusion POC (avril 2026) :**
> Le POC démontre la **faisabilité technique** (les items 🟢 couvrent l'essentiel de la stack data → API → CI/CD).
> En revanche, **18 items en NO-GO** restent à traiter avant tout déploiement clinique réel — principalement
> sur la **validation clinique**, la **conformité hébergement HDS / CNIL**, et la **gouvernance**.
> Recommandation : passer en **Phase 3 pilote** uniquement après validation de la roadmap §9 du rapport
> technique, en commençant par la mise en place de l'AIPD et du panel de validation clinique.

---

*Document à réviser avant chaque jalon de mise en production. Version 1.0 — avril 2026.*
