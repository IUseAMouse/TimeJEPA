# Archive - scripts from closed rounds

Moved here on 2026-08-19 (hygiene audit), never deleted (no-delete policy,
P2.9). Each script answered a question that the experiment registry
(`docs/EXPERIMENTAL_LOG.md`) has since settled; they are no longer wired to
live code and MUST NOT be imported. In particular,
`reevaluate_checkpoints.py` contains a divergent copy of checkpoint loading
(without `filter_loadable`), superseded by
`src/timejepa/evaluation/loading.py`.

| script | question it asked | settled by |
|---|---|---|
| `reevaluate_checkpoints.py` | replay legacy checkpoints against the fixed RevIN (P0.7) | round P0 closed; pointed at `../TimeJEPA_2ndbatch_results/`, outside the repo |
| `report_reevaluation.py` | before/after report of the re-evaluation (P0.8) | same |
| `diagnose_ettm.py` | does the ETTm failure come from the patch size or the frequency? | E14/E17: it is the corpus and the frequency |
| `probe_uncertainty.py` | is uncertainty encoded in the representations? | P2.1: quantile head shipped and measured |
| `compute_model_config.py` | sizing by scaling laws | self-declared deprecated, zero references |
