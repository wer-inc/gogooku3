"""HTTP client helpers for J-Quants."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import requests

from .config import (
    BREAKDOWN_URL,
    DAILY_MARGIN_INTEREST_URL,
    DAILY_QUOTES_URL,
    FS_DETAILS_URL,
    HTTP_TIMEOUT,
    LISTED_URL,
    SHORT_SELLING_POSITIONS_URL,
    SHORT_SELLING_URL,
    STATEMENTS_URL,
    TRADES_SPEC_URL,
    TRADING_CAL_URL,
    WEEKLY_MARGIN_INTEREST_URL,
)

# 33 sector codes (JPX)
SECTOR33_CODES: list[str] = [
    "0050",
    "1050",
    "2050",
    "3050",
    "3100",
    "3150",
    "3200",
    "3250",
    "3300",
    "3350",
    "3400",
    "3450",
    "3500",
    "3550",
    "3600",
    "3650",
    "3700",
    "3750",
    "3800",
    "4050",
    "5050",
    "5100",
    "5150",
    "5200",
    "5250",
    "6050",
    "6100",
    "7050",
    "7100",
    "7150",
    "7200",
    "8050",
    "9050",
    "9999",
]


def _date_iter(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    step = timedelta(days=1)
    cur = s
    out: list[str] = []
    while cur <= e:
        out.append(cur.isoformat())
        cur += step
    return out


def fetch_calendar_records(
    id_token: str,
    *,
    from_date: str | None,
    to_date: str | None,
) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {id_token}"}
    params: dict[str, str] = {}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
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


def fetch_weekly_margin_interest(
    id_token: str,
    *,
    start: str,
    end: str,
    code: str | None = None,
) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {id_token}"}
    all_records: list[dict[str, Any]] = []
    # API requires code or date. If code is provided, use from/to; else loop dates with ?date=.
    if code:
        pagination_key: str | None = None
        while True:
            params: dict[str, str] = {"from": start, "to": end, "code": code}
            if pagination_key:
                params["pagination_key"] = pagination_key
            resp = requests.get(WEEKLY_MARGIN_INTEREST_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            recs = data.get("weekly_margin_interest") or data.get("data") or []
            all_records.extend(recs)
            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
    else:
        # Use trading_calendar to pick business days; weekly_margin_interest is weekly so hitting all biz days
        # still respects API's date requirement but limits to trading days.
        cal_dates = requests.get(
            TRADING_CAL_URL,
            headers=headers,
            params={"from": start, "to": end},
            timeout=HTTP_TIMEOUT,
        )
        cal_dates.raise_for_status()
        cal_data = cal_dates.json()
        cal_records = cal_data.get("trading_calendar") or cal_data.get("data") or []
        # filter to HolidayDivision in ('1','2') => trading days
        days = [
            r["Date"] if "Date" in r else r.get("date")
            for r in cal_records
            if (r.get("HolidayDivision") or r.get("holiday_division")) in ("1", "2")
        ]
        for day in days:
            if not day:
                continue
            pagination_key: str | None = None
            while True:
                params: dict[str, str] = {"date": day}
                if pagination_key:
                    params["pagination_key"] = pagination_key
                resp = requests.get(WEEKLY_MARGIN_INTEREST_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                recs = data.get("weekly_margin_interest") or data.get("data") or []
                all_records.extend(recs)
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
    return all_records


def fetch_daily_margin_interest(
    id_token: str,
    *,
    start: str,
    end: str,
    code: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch /markets/daily_margin_interest."""
    headers = {"Authorization": f"Bearer {id_token}"}
    all_records: list[dict[str, Any]] = []

    if code:
        pagination_key: str | None = None
        while True:
            params: dict[str, str] = {"from": start, "to": end, "code": code}
            if pagination_key:
                params["pagination_key"] = pagination_key
            resp = requests.get(DAILY_MARGIN_INTEREST_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            recs = data.get("daily_margin_interest") or data.get("data") or []
            all_records.extend(recs)
            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
    else:
        # code指定なし → trading_calendar から取引日を取得して date でループ
        cal_resp = requests.get(
            TRADING_CAL_URL,
            headers=headers,
            params={"from": start, "to": end},
            timeout=HTTP_TIMEOUT,
        )
        cal_resp.raise_for_status()
        cal_data = cal_resp.json()
        cal_records = cal_data.get("trading_calendar") or cal_data.get("data") or []
        # filter to HolidayDivision in ('1','2') => trading days
        days = [
            r["Date"] if "Date" in r else r.get("date")
            for r in cal_records
            if (r.get("HolidayDivision") or r.get("holiday_division")) in ("1", "2")
        ]
        for day in days:
            if not day:
                continue
            pagination_key: str | None = None
            while True:
                params: dict[str, str] = {"date": day}
                if pagination_key:
                    params["pagination_key"] = pagination_key
                resp = requests.get(DAILY_MARGIN_INTEREST_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                recs = data.get("daily_margin_interest") or data.get("data") or []
                all_records.extend(recs)
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
    return all_records


def fetch_short_selling(
    id_token: str,
    *,
    start: str,
    end: str,
    sector33code: str | None = None,
) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {id_token}"}
    all_records: list[dict[str, Any]] = []
    # API requires sector33code or date. Prefer sector33code loops to reduce call count.
    codes = [sector33code] if sector33code else SECTOR33_CODES
    for sec in codes:
        pagination_key: str | None = None
        while True:
            params: dict[str, str] = {"from": start, "to": end, "sector33code": sec}
            if pagination_key:
                params["pagination_key"] = pagination_key
            resp = requests.get(SHORT_SELLING_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            recs = data.get("short_selling") or data.get("data") or []
            all_records.extend(recs)
            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
    return all_records


def fetch_short_selling_positions(
    id_token: str,
    *,
    start: str,
    end: str,
    code: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch short selling positions data.

    API requires code, disclosed_date, or calculated_date.
    - If code is provided: use disclosed_date_from/to for bulk fetch
    - If code is not provided: iterate through dates using disclosed_date
    """
    headers = {"Authorization": f"Bearer {id_token}"}
    all_records: list[dict[str, Any]] = []

    if code:
        # code指定あり → from/to でまとめ取得
        pagination_key: str | None = None
        while True:
            params: dict[str, str] = {
                "code": code,
                "disclosed_date_from": start,
                "disclosed_date_to": end,
            }
            if pagination_key:
                params["pagination_key"] = pagination_key
            resp = requests.get(SHORT_SELLING_POSITIONS_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            recs = data.get("short_selling_positions") or data.get("data") or []
            all_records.extend(recs)
            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
    else:
        # code指定なし → trading_calendar から取引日を取得して disclosed_date でループ
        cal_resp = requests.get(
            TRADING_CAL_URL,
            headers=headers,
            params={"from": start, "to": end},
            timeout=HTTP_TIMEOUT,
        )
        cal_resp.raise_for_status()
        cal_data = cal_resp.json()
        cal_records = cal_data.get("trading_calendar") or cal_data.get("data") or []
        # filter to HolidayDivision in ('1','2') => trading days
        days = [
            r["Date"] if "Date" in r else r.get("date")
            for r in cal_records
            if (r.get("HolidayDivision") or r.get("holiday_division")) in ("1", "2")
        ]
        for day in days:
            if not day:
                continue
            pagination_key: str | None = None
            while True:
                params: dict[str, str] = {"disclosed_date": day}
                if pagination_key:
                    params["pagination_key"] = pagination_key
                resp = requests.get(SHORT_SELLING_POSITIONS_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                recs = data.get("short_selling_positions") or data.get("data") or []
                all_records.extend(recs)
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
    return all_records
