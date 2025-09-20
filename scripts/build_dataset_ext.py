from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from gogooku3.features_ext.interactions import add_interactions
from gogooku3.features_ext.outliers import winsorize
from gogooku3.features_ext.scale_unify import add_ratio_adv_z
from gogooku3.features_ext.sector_loo import add_sector_loo
from gogooku3.features_ext.cache_utils import cache_parquet
from gogooku3.features_ext.audit import run_basic_audits, AuditError
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build extended dataset (non-breaking)")
    p.add_argument("--input", type=Path, required=True, help="Input panel parquet")
    p.add_argument("--output", type=Path, required=True, help="Output parquet path")
    p.add_argument("--adv-col", type=str, default="dollar_volume_ma20", help="ADV column name")
    p.add_argument("--config", type=Path, default=None, help="YAML listing ratio/adv pairs and winsor targets")
    p.add_argument("--apply-winsor-in-builder", action="store_true", help="Apply winsorization at build-time (use with care; fold-safe winsor runs in training)")
    p.add_argument("--cache-dir", type=Path, default=None, help="Optional cache directory for intermediate outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pl.read_parquet(str(args.input))

    cfg = {
        "ratio_adv_z": [
            {"value": "margin_long_tot", "adv": args.adv_col, "prefix": "margin_long"},
            {"value": "dmi_long", "adv": args.adv_col, "prefix": "dmi_long"},
        ],
        "winsor": {"cols": ["returns_1d", "returns_5d", "rel_to_sec_5d", "dmi_long_to_adv20", "stmt_rev_fore_op"]},
    }
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        cfg.update(y)

    # 1) Add sector LOO for 1D returns (cacheable)
    def _stage1() -> pl.DataFrame:
        _df = df
        if "returns_1d" in _df.columns and "sector33_id" in _df.columns:
            _df = add_sector_loo(_df, ret_col="returns_1d", sec_col="sector33_id")
        return _df
    df = cache_parquet(args.cache_dir or "cache", name="stage1_loo", kwargs={"input": str(args.input)}, builder=_stage1)

    # 2) Add Ratio/ADV/Z for a small starter list (extend via config as needed)
    def _stage2() -> pl.DataFrame:
        _df = df
        for item in cfg.get("ratio_adv_z", []):
            col = item.get("value"); adv = item.get("adv"); pre = item.get("prefix")
            if col in _df.columns and adv in _df.columns:
                _df = add_ratio_adv_z(_df, value_col=col, adv_col=adv, prefix=pre)
        return _df
    df = cache_parquet(args.cache_dir or "cache", name="stage2_ratio_adv_z", kwargs={"config": str(args.config or "")}, builder=_stage2)

    # 3) Winsorize selected columns if present
    winsor_targets = cfg.get("winsor", {}).get("cols", [])
    present = [c for c in winsor_targets if c in df.columns]
    if present and args.apply_winsor_in_builder:
        def _stage3() -> pl.DataFrame:
            return winsorize(df, present, k=5.0)
        df = cache_parquet(args.cache_dir or "cache", name="stage3_winsor", kwargs={"cols": ",".join(present)}, builder=_stage3)

    # 4) Add compact interactions if base columns exist
    try:
        def _stage4() -> pl.DataFrame:
            return add_interactions(df)
        df = cache_parquet(args.cache_dir or "cache", name="stage4_interactions", kwargs={}, builder=_stage4)
    except pl.ColumnNotFoundError:
        pass

    # Audits
    try:
        notes = run_basic_audits(df)
        if notes:
            print("⚠️ Audit notes:")
            for n in notes:
                print(" - ", n)
    except AuditError as e:
        print(f"❌ Audit failed: {e}")
        raise

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(args.output))


if __name__ == "__main__":
    main()
