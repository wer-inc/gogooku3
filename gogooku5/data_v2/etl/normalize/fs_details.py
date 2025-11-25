"""Normalize /fins/fs_details response."""

from __future__ import annotations

import json
import re

import polars as pl

_FIELD_MAP = {
    "DisclosedDate": "disclosed_date",
    "DisclosedTime": "disclosed_time",
    "LocalCode": "local_code",
    "DisclosureNumber": "disclosure_number",
    "TypeOfDocument": "type_of_document",
}

_DATE_FIELDS = {"disclosed_date"}


def _sanitize_fs_key(key: str) -> str:
    """
    Convert FinancialStatement JSON keys into stable snake_case column names.

    Examples:
    - \"Cash and cash equivalents (IFRS)\" -> \"cash_and_cash_equivalents_ifrs\"
    - \"Other non-current assets - NCA (IFRS)\" -> \"other_non_current_assets_nca_ifrs\"
    """
    k = key.strip().lower()
    # Normalise common suffixes first for readability.
    k = k.replace("(ifrs)", " ifrs")
    k = k.replace("(dei)", " dei")
    # Replace any sequence of non-alphanumeric characters with underscores.
    k = re.sub(r"[^a-z0-9]+", "_", k)
    k = k.strip("_")
    return k


def normalize_fs_details(records: list[dict]) -> pl.DataFrame:
    """
    Normalize /fins/fs_details response into a wide, columnar form.

    - Meta fields (DisclosedDate, LocalCode, ...) are mapped via _FIELD_MAP.
    - FinancialStatement is:
        * preserved as JSON string in financial_statement_json
        * fully flattened into individual columns derived from keys.
    """
    if not records:
        return pl.DataFrame()

    mapped: list[dict] = []
    for rec in records:
        out: dict = {v: rec.get(k) for k, v in _FIELD_MAP.items()}
        fs = rec.get("FinancialStatement") or {}
        # Keep raw JSON for full fidelity.
        out["financial_statement_json"] = json.dumps(fs, ensure_ascii=False)
        # Flatten all FinancialStatement keys into columns.
        if isinstance(fs, dict):
            for raw_key, value in fs.items():
                col = _sanitize_fs_key(str(raw_key))
                # Avoid clobbering meta fields / JSON column if names collide.
                if col in out:
                    # In extremely rare collisions, prefer existing meta/json.
                    continue
                out[col] = value
        mapped.append(out)

    df = pl.DataFrame(mapped)

    # Normalize empty strings to null for all string-typed columns.
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col))

    # Parse date fields where present.
    for col in _DATE_FIELDS:
        if col in df.columns and df.select(pl.col(col).is_not_null().any()).item():
            df = df.with_columns(pl.col(col).str.strptime(pl.Date, strict=False, exact=False))

    return df
