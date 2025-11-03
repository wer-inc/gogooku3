"""Dataset loading helpers for backtesting and inference."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import yaml

_logger = logging.getLogger(__name__)


def apply_feature_aliases(
    frame: pl.DataFrame,
    aliases_yaml: Path | str | None = None,
) -> pl.DataFrame:
    """Apply feature aliases from YAML configuration to create missing columns.

    Args:
        frame: Input dataframe
        aliases_yaml: Path to YAML file with alias definitions. If None, returns frame unchanged.

    Returns:
        DataFrame with alias columns added

    Raises:
        ValueError: If alias expression is invalid or unsupported
        FileNotFoundError: If aliases_yaml path doesn't exist
    """
    if aliases_yaml is None:
        return frame

    aliases_path = Path(aliases_yaml)
    if not aliases_path.exists():
        raise FileNotFoundError(f"Aliases config not found: {aliases_yaml}")

    with aliases_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)

    if not isinstance(data, dict) or "aliases" not in data:
        raise ValueError(f"Invalid aliases config (missing 'aliases' key): {aliases_yaml}")

    alias_map = data["aliases"]
    if not alias_map:
        return frame

    exprs = []
    for new_col, expr_str in alias_map.items():
        expr_str = expr_str.strip()

        # Parse safe arithmetic expressions
        if " - " in expr_str:
            parts = expr_str.split(" - ")
            if len(parts) != 2:
                raise ValueError(f"Alias '{new_col}': subtraction must have exactly 2 operands, got: {expr_str}")
            a, b = parts[0].strip(), parts[1].strip()
            exprs.append((pl.col(a) - pl.col(b)).alias(new_col))

        elif " + " in expr_str:
            parts = expr_str.split(" + ")
            if len(parts) != 2:
                raise ValueError(f"Alias '{new_col}': addition must have exactly 2 operands, got: {expr_str}")
            a, b = parts[0].strip(), parts[1].strip()
            exprs.append((pl.col(a) + pl.col(b)).alias(new_col))

        else:
            # Fix ③: Simple column copying (e.g., "alpha_5d" -> "source_col")
            # If expression is just a column name (no operators), copy it
            if expr_str in frame.columns:
                exprs.append(pl.col(expr_str).alias(new_col))
            else:
                raise ValueError(
                    f"Alias '{new_col}': unsupported expression '{expr_str}'. "
                    f"Only addition (+), subtraction (-), and simple column copying are supported."
                )

    if exprs:
        print(f"[Loader] Applying {len(exprs)} feature aliases from {aliases_path.name}")
        frame = frame.with_columns(exprs)

    return frame


def _resolve_alias(name: str, df_cols: set[str]) -> str | None:
    """Resolve column name with flexible matching (Phase 2 compatibility).

    Args:
        name: Requested column name
        df_cols: Available columns in dataframe

    Returns:
        Matched column name or None if not found

    Resolution strategy:
    1. Exact match
    2. Case-insensitive match
    3. returns_* → ret_prev_* name transition
    4. TurnoverValue ↔ turnovervalue case normalization
    """
    # 1) Exact match
    if name in df_cols:
        return name

    # 2) Case-insensitive match
    low = {c.lower(): c for c in df_cols}
    if name.lower() in low:
        return low[name.lower()]

    # 3) Phase 2 name transition: returns_* → ret_prev_*
    if name.startswith("returns_"):
        alt = name.replace("returns_", "ret_prev_")
        if alt in df_cols:
            return alt
        if alt.lower() in low:
            return low[alt.lower()]

    # 4) TurnoverValue case normalization
    if name == "TurnoverValue" and "turnovervalue" in low:
        return low["turnovervalue"]

    return None


def load_backtest_frame(
    data_path: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    feature_cols: list[str] | None = None,
    lookback: int = 0,
    aliases_yaml: Path | str | None = None,
    features_mode: str = "drop-missing",
    passthrough_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Load parquet dataset with required columns for inference/backtests.

    Args:
        data_path: Path to parquet dataset
        start_date: Start date filter (ISO format)
        end_date: End date filter (ISO format)
        feature_cols: List of feature columns to load
        lookback: Number of days to include before start_date for windowing
        aliases_yaml: Optional path to feature aliases YAML for compatibility
        features_mode: Feature validation mode - "strict" (error on missing),
                      "fill-zero" (fill with 0.0), or "drop-missing" (continue with available)
        passthrough_cols: Additional columns to preserve (e.g., risk factors for EI)

    Returns:
        Loaded and filtered dataframe with aliases applied
    """
    print(f"[Backtest] Loading dataset: {data_path}")

    required_cols: set[str] = {
        "Date",
        "Code",
        "Close",
        "Volume",
        "TurnoverValue",
        "returns_1d",
        "returns_5d",
        "returns_10d",
        "returns_20d",
    }

    if feature_cols:
        required_cols.update(feature_cols)

    if passthrough_cols:
        required_cols.update(passthrough_cols)
        print(f"[Loader] Adding {len(passthrough_cols)} passthrough columns: {passthrough_cols}")

    # Pre-load alias definitions to identify source columns and remove alias targets
    alias_targets: set[str] = set()
    if aliases_yaml:
        aliases_path = Path(aliases_yaml)
        if aliases_path.exists():
            with aliases_path.open("r", encoding="utf-8") as fp:
                alias_data = yaml.safe_load(fp)
            if isinstance(alias_data, dict) and "aliases" in alias_data:
                # Extract source columns from alias expressions
                for target_col, expr_str in alias_data["aliases"].items():
                    alias_targets.add(target_col)
                    expr_str = expr_str.strip()
                    # Parse addition/subtraction to get operands
                    if " - " in expr_str or " + " in expr_str:
                        operands = expr_str.replace(" - ", " ").replace(" + ", " ").split()
                        required_cols.update(operands)

    # Remove alias target columns from required_cols (they don't exist in parquet yet)
    cols_to_load = required_cols - alias_targets

    # Phase 2 Bug #14-30 fix: Use _resolve_alias for flexible column matching
    # First, peek at the dataset schema to get available columns
    dataset_cols = set(pl.read_parquet_schema(str(data_path)).names())

    # Apply alias resolution to requested columns
    available_cols = []
    missing_cols = []
    alias_map = {}  # Maps requested name -> actual column name

    for req_col in cols_to_load:
        resolved = _resolve_alias(req_col, dataset_cols)
        if resolved:
            available_cols.append(resolved)
            if resolved != req_col:
                alias_map[req_col] = resolved
        else:
            missing_cols.append(req_col)

    if alias_map:
        print(f"[Loader] Resolved {len(alias_map)} column aliases (e.g., {list(alias_map.items())[:3]})")

    if missing_cols:
        _logger.warning(
            f"[Loader] Phase 2 compatibility: {len(missing_cols)} requested features "
            f"not in dataset (will be skipped). Examples: {sorted(missing_cols)[:10]}"
        )
        print(
            f"⚠️  [Phase 2 Compat] {len(missing_cols)} features missing from dataset "
            f"(e.g., {', '.join(sorted(missing_cols)[:5])})"
        )

    frame = pl.read_parquet(str(data_path), columns=list(available_cols))
    frame = frame.sort(["Date", "Code"])

    # Apply reverse alias mapping to restore requested column names
    if alias_map:
        reverse_map = {v: k for k, v in alias_map.items()}
        frame = frame.rename(reverse_map)

    # Apply feature aliases for backward compatibility
    if aliases_yaml:
        frame = apply_feature_aliases(frame, aliases_yaml)

    # Fix ②: Feature validation with strict/fill-zero modes
    # After aliases, check if all requested feature_cols are now available
    if feature_cols:
        final_missing = set(feature_cols) - set(frame.columns)
        if final_missing:
            # Exclude target columns from fill-zero mode
            target_patterns = ["target_", "returns_", "label_"]
            safe_missing = [col for col in final_missing if not any(pat in col for pat in target_patterns)]

            if features_mode == "strict":
                raise ValueError(
                    f"[Strict mode] Missing {len(final_missing)} required features after alias application: "
                    f"{sorted(final_missing)[:10]}"
                )
            elif features_mode == "fill-zero" and safe_missing:
                print(
                    f"[Fill-zero mode] Creating {len(safe_missing)} missing features with 0.0: "
                    f"{', '.join(sorted(safe_missing)[:5])}"
                )
                frame = frame.with_columns([pl.lit(0.0).alias(col) for col in safe_missing])
            # drop-missing mode: continue with available features (default behavior)

    start_dt = np.datetime64(start_date) if start_date else None
    end_dt = np.datetime64(end_date) if end_date else None

    buffer_start = None
    if lookback > 0 and start_dt is not None:
        all_dates = frame["Date"].unique().sort().to_numpy()
        idx = np.searchsorted(all_dates, start_dt)
        if idx == 0:
            buffer_idx = 0
        else:
            buffer_idx = max(0, idx - lookback)
        if all_dates.size > 0:
            buffer_start = all_dates[buffer_idx]

    lower_bound = buffer_start if buffer_start is not None else start_dt
    if lower_bound is not None:
        frame = frame.filter(pl.col("Date") >= lower_bound)
    if end_dt is not None:
        frame = frame.filter(pl.col("Date") <= end_dt)

    print(f"[Backtest] Loaded {len(frame):,} rows")
    print(
        "[Backtest] Date span:",
        frame["Date"].min(),
        "→",
        frame["Date"].max(),
    )
    print(f"[Backtest] Unique stocks: {frame['Code'].n_unique()}")

    return frame
