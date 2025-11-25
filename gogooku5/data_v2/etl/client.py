"""HTTP client helpers for J-Quants."""

from __future__ import annotations

from typing import Any

import requests

from .config import (
    BREAKDOWN_URL,
    DAILY_QUOTES_URL,
    FS_DETAILS_URL,
    HTTP_TIMEOUT,
    LISTED_URL,
    STATEMENTS_URL,
    TRADES_SPEC_URL,
    TRADING_CAL_URL,
)


def fetch_calendar_records(id_token: str, *, from_date: str, to_date: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {id_token}"}
    params: dict[str, str] = {"from": from_date, "to": to_date}
    resp = requests.get(TRADING_CAL_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    records = resp.json().get("trading_calendar") or resp.json().get("data")
    if not records:
        raise RuntimeError(f"No trading_calendar data returned for {from_date}->{to_date}")
    return records


def fetch_listed_info_for_date(id_token: str, day: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {id_token}"}
    resp = requests.get(LISTED_URL, headers=headers, params={"date": day}, timeout=HTTP_TIMEOUT)
    if resp.status_code == 413:
        raise RuntimeError("Payload too large (413). Consider narrowing date range.")
    resp.raise_for_status()
    data = resp.json()
    info = data.get("info")
    if not info:
        return []
    return info


def fetch_daily_quotes_for_date(id_token: str, day: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {id_token}"}
    resp = requests.get(DAILY_QUOTES_URL, headers=headers, params={"date": day}, timeout=HTTP_TIMEOUT)
    if resp.status_code == 413:
        raise RuntimeError("Payload too large (413). Narrow date range or split.")
    resp.raise_for_status()
    data = resp.json()
    quotes = data.get("daily_quotes") or data.get("data")
    if not quotes:
        return []
    return quotes


def fetch_breakdown_for_date(id_token: str, day: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {id_token}"}
    resp = requests.get(BREAKDOWN_URL, headers=headers, params={"date": day}, timeout=HTTP_TIMEOUT)
    if resp.status_code == 413:
        raise RuntimeError("Payload too large (413). Narrow date range or split.")
    resp.raise_for_status()
    data = resp.json()
    records = data.get("breakdown") or data.get("data")
    if not records:
        return []
    return records


def fetch_statements_for_date(id_token: str, day: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {id_token}"}
    all_records: list[dict[str, Any]] = []
    pagination_key: str | None = None
    while True:
        params: dict[str, str] = {"date": day}
        if pagination_key:
            params["pagination_key"] = pagination_key
        resp = requests.get(STATEMENTS_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
        if resp.status_code == 413:
            raise RuntimeError("Payload too large (413). Narrow date range or split.")
        resp.raise_for_status()
        data = resp.json()
        recs = data.get("statements") or data.get("data") or []
        all_records.extend(recs)
        pagination_key = data.get("pagination_key")
        if not pagination_key:
            break
    return all_records


def fetch_fs_details_for_date(id_token: str, day: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {id_token}"}
    all_records: list[dict[str, Any]] = []
    pagination_key: str | None = None
    while True:
        params: dict[str, str] = {"date": day}
        if pagination_key:
            params["pagination_key"] = pagination_key
        resp = requests.get(FS_DETAILS_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
        if resp.status_code == 413:
            raise RuntimeError("Payload too large (413). Narrow date range or split.")
        resp.raise_for_status()
        data = resp.json()
        recs = data.get("fs_details") or data.get("data") or []
        all_records.extend(recs)
        pagination_key = data.get("pagination_key")
        if not pagination_key:
            break
    return all_records


def fetch_trades_spec(
    id_token: str,
    *,
    start: str | None = None,
    end: str | None = None,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch markets/trades_spec with optional section and from/to filters."""

    headers = {"Authorization": f"Bearer {id_token}"}
    all_records: list[dict[str, Any]] = []
    pagination_key: str | None = None
    while True:
        params: dict[str, str] = {}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        if section:
            params["section"] = section
        if pagination_key:
            params["pagination_key"] = pagination_key
        resp = requests.get(TRADES_SPEC_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        recs = data.get("trades_spec") or data.get("data") or []
        all_records.extend(recs)
        pagination_key = data.get("pagination_key")
        if not pagination_key:
            break
    return all_records
