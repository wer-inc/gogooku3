# Forward Return Labels

## Definitions
- Horizons: 1, 5, 10, 20 trading days (extendable).
- Return type: simple return consistent with current `returns_*` features.
  - `feat_ret_{h}d(Code, Date=t) = Close_{t+h} / Close_{t} - 1`
- Alignment: labels are forward-looking; features at `t` are used to predict `feat_ret_*` at `t`.

## Masking & Validity
- Per Code, the last `h` rows have `feat_ret_{h}d = NULL` and must be excluded from training/eval.
- Rows with missing `Close` at `t` or `t+h` are set to `NULL` for the corresponding horizon.
- Evaluation uses only rows where both features (at `t`) and labels (at `t+h`) are valid.

## Calendar & Business Days
- Horizons are in trading days. If the dataset has gaps (non-trading days), the `h`-step shift is done over the per-Code sorted trading calendar.
- EOD (as-of) features that become effective `T+1` remain valid for predicting `feat_ret_1d` and beyond, since labels reference `t+h`.

## Dividend/Corporate Actions
- Returns assume the `Close` series is split-adjusted; if dividend-adjusted series is available, prefer that for label computation to avoid dividend jumps.
- Document the series used in experiment metadata (split-only vs total-return proxy).

## Leakage Guards
- Do not forward-fill labels; only compute via forward shift.
- Ensure any EOD-disclosed features use `effective_date <= Date_t` (already enforced via as-of joins).

## Implementation Notes
- Computation (per Code, date-sorted):
  - `feat_ret_1d = (Close.shift(-1) / Close) - 1`
  - `feat_ret_5d = (Close.shift(-5) / Close) - 1`
  - etc.
- Storage: optional; labels can be materialized into the parquet or computed on-the-fly at training.
- Quality check: verify `feat_ret_1d(Code,t) == returns_1d(Code,t+1)` for consistency (already in `data_checks.py`).

