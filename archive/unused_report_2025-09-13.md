# Unused Files Audit Report

Generated: 2025-09-13 07:41:34

## Summary
- Unimported Python modules (src/gogooku3): 9
- Maybe-unreferenced non-Python files: 19

## Unimported Python Modules (candidates)
- `src/gogooku3/compat/aliases.py`  (module: `gogooku3.compat.aliases`)
- `src/gogooku3/compat/script_wrappers.py`  (module: `gogooku3.compat.script_wrappers`)
- `src/gogooku3/contracts/schemas.py`  (module: `gogooku3.contracts.schemas`)
- `src/gogooku3/features/cross_features.py`  (module: `gogooku3.features.cross_features`)
- `src/gogooku3/features/financial_features.py`  (module: `gogooku3.features.financial_features`)
- `src/gogooku3/features/flow_features.py`  (module: `gogooku3.features.flow_features`)
- `src/gogooku3/features/ta_core.py`  (module: `gogooku3.features.ta_core`)
- `src/gogooku3/joins/intervals.py`  (module: `gogooku3.joins.intervals`)
- `src/gogooku3/utils/calendar_utils.py`  (module: `gogooku3.utils.calendar_utils`)

## Maybe-unreferenced Non-Python Files (candidates)
- `configs/atft/best_hyperparameters.json`
- `configs/atft/config_phase1.yaml`
- `configs/atft/config_phase2.yaml`
- `configs/atft/config_phase3.yaml`
- `configs/atft/hydra/job_logging/jst.yaml`
- `configs/atft/inference/streaming.yaml`
- `configs/atft/train/profiles/robust.yaml`
- `configs/atft/train/safe_production.yaml`
- `configs/atft/variance_fix_config.json`
- `configs/atft_success_env.sh`
- `configs/experiment/baseline.yaml`
- `configs/features/market.yaml`
- `configs/inference/batch.yaml`
- `configs/joins/settings.yaml`
- `configs/jquants_api_config.yaml`
- `configs/market_code_config.yaml`
- `configs/sector_mappings/sector17_map.example.json`
- `configs/sector_mappings/sector33_map.example.json`
- `scripts/_archive/complete_atft_training.sh`

## Notes
- Static analysis; confirm by quarantining and running tests/pipelines.
- Dynamic imports, CLI entrypoints, and DAG tools may hide usage.
