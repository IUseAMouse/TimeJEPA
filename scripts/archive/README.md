# Archive — scripts de rounds clos

Déplacés ici le 2026-08-19 (audit d'hygiène), jamais supprimés (doctrine P2.9).
Chaque script a répondu à une question aujourd'hui tranchée par le registre
expérimental (`docs/EXPERIMENTAL_LOG.md`) ; ils ne sont plus branchés sur le
code vivant et NE DOIVENT PAS être importés — `reevaluate_checkpoints.py`
contient notamment une copie divergente du chargement de checkpoints
(sans `filter_loadable`), remplacée par `src/timejepa/evaluation/loading.py`.

| script | question posée | tranchée par |
|---|---|---|
| `reevaluate_checkpoints.py` | replay des checkpoints legacy vs RevIN corrigé (P0.7) | round P0 clos ; pointait vers `../TimeJEPA_2ndbatch_results/`, hors du repo |
| `report_reevaluation.py` | rapport avant/après de la re-éval (P0.8) | idem |
| `diagnose_ettm.py` | l'échec ETTm vient-il du patch ou de la fréquence ? | E14/E17 : c'est le corpus et la fréquence |
| `probe_uncertainty.py` | l'incertitude est-elle encodée dans les représentations ? | P2.1 : tête quantile livrée et mesurée |
| `compute_model_config.py` | dimensionnement par lois d'échelle | auto-déclaré deprecated, zéro référence |
