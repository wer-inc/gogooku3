"""
Static sector code → name mappings for JPX 17/33 sectors.

This module is optional and only used to backfill names when the API payload
does not include `Sector17Name` / `Sector33Name`.
"""

from __future__ import annotations

from typing import Dict
from pathlib import Path
import json
import os

# Minimal, extendable maps. Keys are strings to preserve leading zeros.
SECTOR17_NAME_MAP: Dict[str, str] = {
    # Common examples (accept both zero-padded and non-padded keys)
    "01": "食品", "1": "食品",
    "02": "繊維製品", "2": "繊維製品",
    "03": "パルプ・紙", "3": "パルプ・紙",
    "04": "化学", "4": "化学",
    "05": "医薬品", "5": "医薬品",
    "06": "石油・石炭製品", "6": "石油・石炭製品",
    "07": "ゴム製品", "7": "ゴム製品",
    "08": "ガラス・土石製品", "8": "ガラス・土石製品",
    "09": "鉄鋼", "9": "鉄鋼",
    "10": "非鉄金属",
    "11": "金属製品",
    "12": "機械",
    "13": "電気機器",
    "14": "輸送用機器",
    "15": "精密機器",
    "16": "その他製品",
    "17": "情報・通信",  # 一部データでは「情報通信」と表記
}

SECTOR33_NAME_MAP: Dict[str, str] = {
    # Frequent examples in our dataset/tests; extend as needed
    "3200": "化学",
    "3300": "医薬品",
    "3400": "石油・石炭製品",
    "4200": "電気機器",
    "4300": "輸送用機器",
    "6050": "小売業",
    "7050": "銀行業",
    "7100": "証券、商品先物取引業",
    "9999": "その他",
}


def get_sector17_name(code: str | None) -> str:
    if code is None:
        return ""
    name = SECTOR17_NAME_MAP.get(code)
    if name:
        return name
    # try without leading zeros
    name = SECTOR17_NAME_MAP.get(code.lstrip("0"))
    return name or ""


def get_sector33_name(code: str | None) -> str:
    if code is None:
        return ""
    return SECTOR33_NAME_MAP.get(code, "")


def _load_json(path: Path) -> Dict[str, str]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Normalize keys to strings
                    return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def load_overrides() -> None:
    """
    Load optional overrides from JSON into the in-memory maps.

    Search order:
    - Env vars `SECTOR17_MAP_JSON`, `SECTOR33_MAP_JSON`
    - Default: `configs/sector_mappings/sector17_map.json` and `sector33_map.json`
    """
    root = Path(__file__).resolve().parents[2]
    default_dir = root / "configs" / "sector_mappings"
    s17_path = Path(os.getenv("SECTOR17_MAP_JSON", str(default_dir / "sector17_map.json")))
    s33_path = Path(os.getenv("SECTOR33_MAP_JSON", str(default_dir / "sector33_map.json")))

    o17 = _load_json(s17_path)
    o33 = _load_json(s33_path)
    if o17:
        SECTOR17_NAME_MAP.update(o17)
    if o33:
        SECTOR33_NAME_MAP.update(o33)


# Load overrides at import time (no-op if files do not exist)
load_overrides()
