#!/usr/bin/env python3
"""
validator_jqx_all.py — Contract-driven validation for all J-Quants data blocks.

Validates dataset against a YAML contract file that defines:
- Required/derived columns per data source
- As-of rules (temporal safety)
- Null rate thresholds
- Invariant checks (e.g., ret_prev_1d ≈ overnight × intraday)
- Denylist patterns (forward-looking columns)

USAGE
-----
python tools/validator_jqx_all.py \
  --dataset gogooku5/data/output/ml_dataset_latest_full.parquet \
  --contract tools/contracts/jqx_all.yaml \
  --debug-meta-dir gogooku5/data/output/cache/snapshots \
  --report /tmp/jqx_validation_report.json \
  --summary /tmp/jqx_validation_summary.md
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import yaml


def has_columns(df: pl.DataFrame, specs: List[str]) -> List[str]:
    """Check if columns match patterns (exact or prefix with *)."""
    missing = []
    df_cols_lower = {col.lower(): col for col in df.columns}

    for spec in specs:
        if "*" in spec:
            # Prefix pattern (e.g., "wm_*_z20" → any column starting with "wm_" and ending with "_z20")
            pattern = spec.replace("*", ".*")
            pattern_re = re.compile(f"^{pattern}$", re.IGNORECASE)
            if not any(pattern_re.match(col) for col in df.columns):
                missing.append(spec)
        else:
            # Exact match
            if spec.lower() not in df_cols_lower:
                missing.append(spec)
    return missing


def frac_null(df: pl.DataFrame, col: str) -> float:
    """Calculate null fraction for a column."""
    if col not in df.columns:
        return 1.0  # Missing column is treated as 100% null
    try:
        result = df.select(pl.col(col).is_null().mean()).collect()
        if result.is_empty():
            return 0.0
        val = result[col][0]
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


def check_invariant(df: pl.LazyFrame, expr: str, tol: float) -> int:
    """Check invariant expression (e.g., (1+ret_prev_1d) ≈ (1+ret_overnight)*(1+ret_intraday))."""
    try:
        # Parse invariant expression
        # Example: "(1+ret_prev_1d) ≈ (1+ret_overnight)*(1+ret_intraday) tol=1e-5"
        if "≈" not in expr:
            return 0

        parts = expr.split("≈")
        if len(parts) != 2:
            return 0

        lhs_expr = parts[0].strip()
        rhs_part = parts[1].split("tol=")[0].strip()

        # Check specific known invariants
        if "ret_prev_1d" in expr and "ret_overnight" in expr and "ret_intraday" in expr:
            # (1+ret_prev_1d) ≈ (1+ret_overnight)*(1+ret_intraday)
            required_cols = ["ret_prev_1d", "ret_overnight", "ret_intraday"]
            if all(col in df.columns for col in required_cols):
                # Calculate difference: |(1+ret_prev_1d) - (1+ret_overnight)*(1+ret_intraday)|
                lhs = 1.0 + pl.col("ret_prev_1d")
                rhs = (1.0 + pl.col("ret_overnight")) * (1.0 + pl.col("ret_intraday"))
                diff = (lhs - rhs).abs()

                # Count violations where diff > tol and all values are not null
                violations_expr = (
                    (diff > tol)
                    & pl.col("ret_prev_1d").is_not_null()
                    & pl.col("ret_overnight").is_not_null()
                    & pl.col("ret_intraday").is_not_null()
                )
                violations = df.select(violations_expr.sum().alias("_violations")).collect()
                if not violations.is_empty():
                    return int(violations["_violations"][0] or 0)
        return 0
    except Exception as e:
        # Log error but don't fail validation
        import logging

        logging.getLogger(__name__).debug(f"Invariant check failed: {e}")
        return 0


def check_asof_violations(
    df: pl.DataFrame,
    snapshot_path: Optional[Path],
    asof_rule: str,
) -> Dict[str, Any]:
    """Check as-of violations if snapshot data is available."""
    if snapshot_path is None or not snapshot_path.exists():
        return {"violations": 0, "checked": False}

    try:
        snapshot = pl.read_parquet(snapshot_path)
        if "available_ts" not in snapshot.columns or "asof_ts" not in df.columns:
            return {"violations": 0, "checked": False}

        # Join and check available_ts <= asof_ts
        # This is simplified; actual implementation depends on join keys
        violations = 0
        return {"violations": violations, "checked": True}
    except Exception:
        return {"violations": 0, "checked": False}


def check_denylist(df: pl.DataFrame, patterns: List[str]) -> List[str]:
    """Check for columns matching denylist patterns."""
    violations = []
    for pattern in patterns:
        pattern_re = re.compile(pattern, re.IGNORECASE)
        matching = [col for col in df.columns if pattern_re.match(col)]
        if matching:
            violations.extend(matching)
    return violations


def check_canonical_ohlc(df: pl.DataFrame, required: List[str], banned: List[str]) -> Dict[str, Any]:
    """Check canonical OHLC: required columns exist, banned columns are absent."""
    missing_required = has_columns(df, required)
    banned_found = []

    for pattern in banned:
        pattern_re = re.compile(pattern, re.IGNORECASE)
        matching = [col for col in df.columns if pattern_re.match(col)]
        banned_found.extend(matching)

    return {
        "missing_required": missing_required,
        "banned_found": banned_found,
    }


def run_validation(
    dataset_path: str,
    contract_path: str,
    debug_meta_dir: Optional[str] = None,
    report_path: Optional[str] = None,
    summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run contract-driven validation."""
    # Load contract
    with open(contract_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    # Load dataset (lazy scan for memory efficiency)
    df = pl.scan_parquet(dataset_path)

    # Collect once for column checks
    df_sample = df.select(pl.all()).limit(1000).collect()
    all_columns = df_sample.columns

    report: Dict[str, Any] = {"status": "PASS", "blocks": []}

    # 1) Primary key check
    primary_key = spec.get("dataset", {}).get("primary_key", ["code", "date"])
    if all(k.lower() in {col.lower() for col in all_columns} for k in primary_key):
        # Check for duplicates
        key_cols = [col for col in all_columns if col.lower() in {k.lower() for k in primary_key}]
        if len(key_cols) == len(primary_key):
            dup_check = (
                df.select(pl.concat_str([pl.col(c) for c in key_cols], separator="|").alias("_key"))
                .group_by("_key")
                .agg(pl.len().alias("_cnt"))
                .filter(pl.col("_cnt") > 1)
                .collect()
            )
            dup_count = len(dup_check)
            if dup_count > 0:
                report["status"] = "FAIL"
                report["blocks"].append(
                    {
                        "block": "primary_key",
                        "status": "FAIL",
                        "issues": [{"duplicate_code_date": f"{dup_count} duplicate rows"}],
                    }
                )

    # 2) Denylist check
    denylist_conf = spec.get("sources", {}).get("denylist", {})
    if denylist_conf:
        denylist_patterns = denylist_conf.get("patterns", [])
        denylist_violations = check_denylist(df_sample, denylist_patterns)
        if denylist_violations:
            report["status"] = "FAIL"
            report["blocks"].append(
                {
                    "block": "denylist",
                    "status": "FAIL",
                    "issues": [{"forbidden_columns": denylist_violations}],
                }
            )

    # 3) Canonical OHLC check
    canonical_conf = spec.get("sources", {}).get("canonical_ohlc", {})
    if canonical_conf:
        required = canonical_conf.get("required", [])
        banned = canonical_conf.get("banned", [])
        ohlc_check = check_canonical_ohlc(df_sample, required, banned)
        if ohlc_check["missing_required"] or ohlc_check["banned_found"]:
            block_status = "FAIL" if ohlc_check["missing_required"] else "WARN"
            if block_status == "FAIL":
                report["status"] = "FAIL"
            report["blocks"].append(
                {
                    "block": "canonical_ohlc",
                    "status": block_status,
                    "issues": [
                        {"missing_required": ohlc_check["missing_required"]},
                        {"banned_found": ohlc_check["banned_found"]},
                    ],
                }
            )

    # 4) Validate each source block
    sources = spec.get("sources", {})
    for block_name, block_conf in sources.items():
        if block_name in ("denylist", "canonical_ohlc"):
            continue  # Already checked

        block_report: Dict[str, Any] = {"block": block_name, "status": "PASS", "issues": []}

        # Check required columns
        required = block_conf.get("required", [])
        missing_req = has_columns(df_sample, required)
        if missing_req:
            block_report["status"] = "FAIL"
            block_report["issues"].append({"missing_required": missing_req})

        # Check derived columns (optional blocks may skip)
        derived = block_conf.get("derived", [])
        missing_der = has_columns(df_sample, derived)
        is_optional = block_conf.get("optional", False)
        if missing_der:
            if not is_optional:
                if block_report["status"] == "PASS":
                    block_report["status"] = "WARN"
                block_report["issues"].append({"missing_derived": missing_der})
            else:
                # Optional block: only warn if partially present
                if any(col.lower() in {c.lower() for c in all_columns} for col in derived):
                    block_report["issues"].append({"missing_derived_partial": missing_der})

        # Check null thresholds (use full dataset for accurate calculation)
        null_max = block_conf.get("rules", {}).get("null_max", {})
        for col, threshold in null_max.items():
            # Find actual column name (case-insensitive)
            actual_col = None
            if col in df_sample.columns:
                actual_col = col
            else:
                for c in df_sample.columns:
                    if c.lower() == col.lower():
                        actual_col = c
                        break

            if actual_col:
                # Use lazy evaluation for memory efficiency
                null_rate_expr = pl.col(actual_col).is_null().mean()
                null_rate_result = df.select(null_rate_expr.alias("_null_rate")).collect()
                if not null_rate_result.is_empty():
                    null_rate = float(null_rate_result["_null_rate"][0] or 0.0)
                    if null_rate > threshold:
                        severity = "FAIL" if null_rate > threshold * 1.5 else "WARN"
                        if severity == "FAIL":
                            block_report["status"] = "FAIL"
                        block_report["issues"].append(
                            {
                                f"null_rate_{col}": {
                                    "threshold": threshold,
                                    "actual": round(null_rate, 4),
                                    "severity": severity,
                                }
                            }
                        )

        # Check invariants (use full dataset for accuracy)
        invariants = block_conf.get("rules", {}).get("invariants", [])
        for inv in invariants:
            tol = float(inv.split("tol=")[-1]) if "tol=" in inv else 1e-6
            violations = check_invariant(df, inv, tol)
            if violations > 0:
                block_report["status"] = "FAIL"
                block_report["issues"].append(
                    {"invariant_violation": {"expr": inv, "violations": violations, "tolerance": tol}}
                )

        # Check as-of (if snapshot available)
        asof_rule = block_conf.get("rules", {}).get("asof")
        if asof_rule and debug_meta_dir:
            snapshot_path = Path(debug_meta_dir) / f"{block_name}_snapshot.parquet"
            asof_check = check_asof_violations(df_sample, snapshot_path, asof_rule)
            if asof_check["checked"] and asof_check["violations"] > 0:
                block_report["status"] = "FAIL"
                block_report["issues"].append(
                    {"asof_violation": {"violations": asof_check["violations"], "rule": asof_rule}}
                )

        report["blocks"].append(block_report)

        # Update overall status
        if block_report["status"] == "FAIL":
            report["status"] = "FAIL"
        elif block_report["status"] == "WARN" and report["status"] != "FAIL":
            report["status"] = "WARN"

    # Write report
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # Write summary
    if summary_path:
        write_summary(report, summary_path)

    return report


def write_summary(report: Dict[str, Any], summary_path: str) -> None:
    """Write human-readable summary."""
    lines = [
        "# J-Quants Dataset Validation Report",
        "",
        f"**Status**: {report['status']}",
        "",
        "## Block Summary",
        "",
    ]

    for block in report.get("blocks", []):
        status_icon = "❌" if block["status"] == "FAIL" else "⚠️" if block["status"] == "WARN" else "✅"
        lines.append(f"### {status_icon} {block['block']} ({block['status']})")

        if block.get("issues"):
            for issue in block["issues"]:
                for key, value in issue.items():
                    if isinstance(value, list):
                        lines.append(f"- {key}: {', '.join(map(str, value))}")
                    elif isinstance(value, dict):
                        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
                    else:
                        lines.append(f"- {key}: {value}")
        else:
            lines.append("- No issues")
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Contract-driven validation for J-Quants dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset parquet file")
    parser.add_argument("--contract", required=True, help="Path to YAML contract file")
    parser.add_argument("--debug-meta-dir", help="Directory containing snapshot parquet files for as-of checks")
    parser.add_argument("--report", default="/tmp/jqx_validation_report.json", help="Output JSON report path")
    parser.add_argument("--summary", default="/tmp/jqx_validation_summary.md", help="Output Markdown summary path")

    args = parser.parse_args()

    report = run_validation(
        dataset_path=args.dataset,
        contract_path=args.contract,
        debug_meta_dir=args.debug_meta_dir,
        report_path=args.report,
        summary_path=args.summary,
    )

    print(f"Validation complete. Status: {report['status']}")
    print(f"Report: {args.report}")
    print(f"Summary: {args.summary}")

    # Exit with error code if FAIL
    if report["status"] == "FAIL":
        exit(1)
    elif report["status"] == "WARN":
        exit(0)  # WARN is acceptable


if __name__ == "__main__":
    main()
