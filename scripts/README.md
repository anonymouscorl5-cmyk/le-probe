# Utility Scripts

This directory contains one-off helpers used during dataset audits, artifact generation, and reproducibility checks.

## Typical Use Cases

- feature trigger mining and activation sanity checks,
- canonical state extraction for figure reproduction,
- reward-model comparison and validation,
- dataset conversion/compression and integrity scans.

## Representative Scripts

- `find_feature_triggers.py`
- `generate_canonical_triptychs.py`
- `reproduce_canonical_states.py`
- `check_target_activations.py`
- `compare_reward_models.py`
- `convert_to_parquet.py`
- `compress_dataset.py`

## Notes

- Most scripts are maintenance utilities, not part of the primary training/inference path.
- Prefer module-level READMEs (`dataset`, `lewm`, `interpretability`) for paper-facing workflows.
