from __future__ import annotations

"""
Earnings event features with strict as-of controls.

Implements the minimal P0 feature set:
  - e_days_to / e_days_since: distance to next / previous earnings (business days)
  - e_win_pre{1,3,5} / e_win_post{1,3,5}: proximity flags
  - e_is_E0: event-day indicator

Data is sourced from J-Quants `/fins/announcement` and guarded so that announcements
only become visible from the evening (default 19:00 JST) of the previous business day.
"""

import bisect
import logging
from datetime import date, datetime, time
from typing import Iterable, Sequence

import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_WINDOWS: tuple[int, ...] = (1, 3, 5)
SCHEDULE_REFRESH_HOUR: int = 19


def _coerce_date(value: object) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def _coerce_time(value: object) -> time:
    if value in (None, ""):
        return time(0, 0)
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return time(0, 0)
        parts = value.split(":")
        try:
            hour = int(parts[0]) if parts else 0
            minute = int(parts[1]) if len(parts) > 1 else 0
            second = int(parts[2]) if len(parts) > 2 else 0
            return time(hour % 24, minute % 60, second % 60)
        except ValueError:
            return time(0, 0)
    if isinstance(value, (int, float)):
        hour = int(value) % 24
        return time(hour, 0)
    return time(0, 0)


def _coerce_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, time(0, 0))
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        token = token.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(token)
            return dt.replace(tzinfo=None)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(token, fmt)
            except ValueError:
                continue
    return None


def _parse_business_days(
    quotes: pl.DataFrame, business_days: Iterable[str] | None
) -> tuple[list[date], dict[date, int], pl.DataFrame]:
    """Build business-day index mapping used for distance calculations."""
    if business_days:
        try:
            bd_dates = sorted(datetime.strptime(str(d), "%Y-%m-%d").date() for d in business_days)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid business day token: {exc}") from exc
    else:
        bd_dates = quotes.select("Date").unique().sort("Date")["Date"].to_list()
        bd_dates = [d if isinstance(d, date) else d.date() for d in bd_dates]

    if not bd_dates:
        raise ValueError("Business day calendar cannot be empty")

    date_to_idx = {d: idx for idx, d in enumerate(bd_dates)}
    calendar_df = (
        pl.DataFrame({"Date": bd_dates})
        .with_row_count("day_index", offset=0)
        .with_columns(pl.col("day_index").cast(pl.Int32))
    )
    return bd_dates, date_to_idx, calendar_df


def _bisect_index(target: date, bd_dates: Sequence[date]) -> int | None:
    """Return insertion index of target within business-day list or None if beyond."""
    pos = bisect.bisect_left(bd_dates, target)
    if pos >= len(bd_dates):
        return None
    return pos


def _announcement_day_index(target: date, bd_dates: Sequence[date], date_to_idx: dict[date, int]) -> int | None:
    """Map announcement date to business-day index (nearest forward business day)."""
    if target in date_to_idx:
        return date_to_idx[target]
    return _bisect_index(target, bd_dates)


def _prev_business_day_index(target: date, bd_dates: Sequence[date], date_to_idx: dict[date, int]) -> int:
    """Return index of previous business day (inclusive of first day)."""
    if target in date_to_idx:
        idx = date_to_idx[target]
        return max(idx - 1, 0)

    pos = bisect.bisect_left(bd_dates, target)
    if pos <= 0:
        return 0
    return min(pos - 1, len(bd_dates) - 1)


def _resolve_effective_day_index(
    available_ts: datetime,
    bd_dates: Sequence[date],
    asof_hour: int,
) -> tuple[int, date]:
    """Return first business-day index whose as-of timestamp covers available_ts."""
    candidate_idx = bisect.bisect_left(bd_dates, available_ts.date())
    if candidate_idx >= len(bd_dates):
        return len(bd_dates) - 1, bd_dates[-1]

    start_idx = max(candidate_idx - 1, 0)
    asof_cutoff = time(hour=asof_hour, minute=0)
    for idx in range(start_idx, len(bd_dates)):
        candidate_date = bd_dates[idx]
        candidate_ts = datetime.combine(candidate_date, asof_cutoff)
        if candidate_ts >= available_ts:
            return idx, candidate_date

    return len(bd_dates) - 1, bd_dates[-1]


def _build_snapshot(
    announcement_df: pl.DataFrame,
    *,
    bd_dates: Sequence[date],
    date_to_idx: dict[date, int],
    asof_hour: int,
) -> pl.DataFrame:
    """Normalize announcement schedule and attach availability metadata."""
    if announcement_df.is_empty():
        return pl.DataFrame(
            {
                "Code": pl.Series([], dtype=pl.Utf8),
                "announcement_date": pl.Series([], dtype=pl.Date),
                "announcement_day_index": pl.Series([], dtype=pl.Int32),
                "effective_day_index": pl.Series([], dtype=pl.Int32),
                "effective_date": pl.Series([], dtype=pl.Date),
                "available_ts": pl.Series([], dtype=pl.Datetime("us")),
            }
        )

    df = announcement_df.rename({"Date": "announcement_date"})
    if "announcement_date" not in df.columns:
        raise ValueError("Earnings announcements require a 'Date' column")

    df = df.with_columns(
        [
            pl.col("announcement_date").cast(pl.Date),
            pl.col("Code").cast(pl.Utf8),
        ]
    )
    df = df.unique(["Code", "announcement_date"])

    ann_dates = df["announcement_date"].to_list()

    published_ts_list: list[datetime | None]
    if "PublishedDateTime" in df.columns:
        published_ts_list = [_coerce_datetime(val) for val in df["PublishedDateTime"].to_list()]
    elif "PublishedAt" in df.columns:
        published_ts_list = [_coerce_datetime(val) for val in df["PublishedAt"].to_list()]
    elif "published_at" in df.columns:
        published_ts_list = [_coerce_datetime(val) for val in df["published_at"].to_list()]
    elif "PublishedDate" in df.columns:
        dates_raw = df["PublishedDate"].to_list()
        times_raw = df["PublishedTime"].to_list() if "PublishedTime" in df.columns else [None] * len(dates_raw)
        published_ts_list = []
        for raw_date, raw_time in zip(dates_raw, times_raw):
            parsed_date = _coerce_date(raw_date)
            parsed_time = _coerce_time(raw_time)
            if parsed_date is None:
                published_ts_list.append(None)
            else:
                published_ts_list.append(datetime.combine(parsed_date, parsed_time))
    else:
        published_ts_list = [None] * len(ann_dates)

    ann_indices: list[int | None] = []
    eff_indices: list[int] = []
    eff_dates: list[date] = []
    avail_ts: list[datetime] = []

    refresh_cutoff = time(hour=SCHEDULE_REFRESH_HOUR, minute=0)
    for d, published_ts in zip(ann_dates, published_ts_list):
        idx = _announcement_day_index(d, bd_dates, date_to_idx)
        ann_indices.append(idx)

        prev_idx = _prev_business_day_index(d, bd_dates, date_to_idx)
        fallback_date = bd_dates[prev_idx]
        fallback_ts = datetime.combine(fallback_date, refresh_cutoff)

        if published_ts is not None:
            published_ts = published_ts.replace(microsecond=0)
            available_ts = published_ts if published_ts > fallback_ts else fallback_ts
        else:
            available_ts = fallback_ts

        eff_idx, eff_date = _resolve_effective_day_index(available_ts, bd_dates, asof_hour)
        eff_indices.append(eff_idx)
        eff_dates.append(eff_date)
        avail_ts.append(available_ts)

    snapshot = df.with_columns(
        [
            pl.Series("announcement_day_index", ann_indices, dtype=pl.Int32),
            pl.Series("effective_day_index", eff_indices, dtype=pl.Int32),
            pl.Series("effective_date", eff_dates, dtype=pl.Date),
            pl.Series("available_ts", avail_ts, dtype=pl.Datetime("us")),
        ]
    )

    snapshot = snapshot.filter(pl.col("announcement_day_index").is_not_null())
    if snapshot.is_empty():
        return snapshot

    snapshot = snapshot.sort(["Code", "announcement_day_index", "available_ts"]).unique(
        ["Code", "announcement_day_index"], keep="last"
    )
    return snapshot


def _join_asof_forward(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    by: str,
    left_on: str,
    right_on: str,
    suffix: str,
) -> pl.DataFrame:
    """Emulate forward as-of join via sign inversion (Polars only exposes backward)."""
    if left.is_empty() or right.is_empty():
        return left

    right_columns = list(right.columns)

    left_augmented = (
        left.with_row_count("_join_order")
        .with_columns((-pl.col(left_on)).alias("_neg_idx_left"))
        .sort([by, "_neg_idx_left", "_join_order"])
    )
    right_augmented = right.with_columns((-pl.col(right_on)).alias("_neg_idx_right")).sort([by, "_neg_idx_right"])

    joined = left_augmented.join_asof(
        right_augmented,
        left_on="_neg_idx_left",
        right_on="_neg_idx_right",
        by=by,
        strategy="backward",
    )

    joined = joined.drop(["_neg_idx_left", "_neg_idx_right"], strict=False)

    rename_map: dict[str, str] = {}
    for col in right_columns:
        if col == by:
            continue
        candidate = f"{col}{suffix}"
        if candidate in joined.columns:
            continue
        if col in joined.columns:
            rename_map[col] = candidate
    if rename_map:
        joined = joined.rename(rename_map)

    joined = joined.sort("_join_order").drop("_join_order")
    return joined


def _ensure_windows(windows: Sequence[int] | None) -> list[int]:
    if not windows:
        return list(DEFAULT_WINDOWS)
    uniq = sorted(set(int(w) for w in windows if int(w) > 0))
    if not uniq:
        return list(DEFAULT_WINDOWS)
    return uniq


def add_earnings_event_block(  # noqa: PLR0912
    quotes: pl.DataFrame,
    announcement_df: pl.DataFrame | None,
    statements_df: pl.DataFrame | None = None,
    *,
    business_days: Iterable[str] | None = None,
    windows: Sequence[int] | None = None,
    asof_hour: int = 15,
    enable_pead: bool = False,  # retained for compatibility (no-op in P0)
    enable_volatility: bool = False,  # retained for compatibility (no-op in P0)
) -> pl.DataFrame:
    """
    Attach earnings proximity features to the quotes dataframe.

    Args:
        quotes: Base panel (expects Code/Date columns).
        announcement_df: Normalized earnings announcements.
        statements_df: Unused in P0 (placeholder for forward compatibility).
        business_days: Optional explicit business-day list (YYYY-MM-DD strings).
        windows: Proximity windows (business days) e.g. [1, 3, 5].
        asof_hour: Dataset as-of hour (JST, 0-23) for availability checks.
    """
    _ = statements_df  # deliberately unused in P0

    if asof_hour < 0 or asof_hour > 23:
        raise ValueError("asof_hour must be between 0 and 23")

    win = _ensure_windows(windows)

    if announcement_df is None or announcement_df.is_empty():
        logger.info("Earnings announcements missing; injecting null proximity features")
        null_cols = [
            pl.lit(None).cast(pl.Int32).alias("e_days_to"),
            pl.lit(None).cast(pl.Int32).alias("e_days_since"),
            pl.lit(0).cast(pl.Int8).alias("e_is_E0"),
        ]
        for w in win:
            null_cols.append(pl.lit(0).cast(pl.Int8).alias(f"e_win_pre{w}"))
            null_cols.append(pl.lit(0).cast(pl.Int8).alias(f"e_win_post{w}"))
        return quotes.with_columns(null_cols)

    bd_dates, date_to_idx, calendar_df = _parse_business_days(quotes, business_days)
    snapshot = _build_snapshot(
        announcement_df,
        bd_dates=bd_dates,
        date_to_idx=date_to_idx,
        asof_hour=asof_hour,
    )

    if snapshot.is_empty():
        logger.info("No announcements within business-day range; features remain null")
        return add_earnings_event_block(
            quotes,
            None,
            business_days=business_days,
            windows=win,
            asof_hour=asof_hour,
        )

    base = quotes.with_row_count("_row_nr")

    working = base.join(calendar_df, on="Date", how="left").with_columns(
        [
            pl.col("day_index").cast(pl.Int32),
            (pl.col("Date").cast(pl.Datetime("us")) + pl.duration(hours=asof_hour)).alias("_asof_ts"),
        ]
    )

    required = {"Code", "day_index"}
    if not required.issubset(set(working.columns)):
        missing = required - set(working.columns)
        raise ValueError(f"Quotes dataframe missing required columns: {missing}")

    working = working.sort(["Code", "day_index", "_row_nr"])
    snapshot = snapshot.sort(["Code", "announcement_day_index"])

    with_prev = working.join_asof(
        snapshot,
        left_on="day_index",
        right_on="announcement_day_index",
        by="Code",
        strategy="backward",
    )
    prev_rename = {col: f"{col}_prev" for col in snapshot.columns if col not in {"Code"}}
    with_prev = with_prev.rename(prev_rename)
    with_prev_next = _join_asof_forward(
        with_prev,
        snapshot,
        by="Code",
        left_on="day_index",
        right_on="announcement_day_index",
        suffix="_next",
    )

    conflict_cols = [col for col in with_prev_next.columns if col.endswith("_right")]
    if conflict_cols:
        with_prev_next = with_prev_next.drop(conflict_cols)

    # Distance calculations (business-day indexes already aligned)
    features = with_prev_next.with_columns(
        [
            pl.when(
                (pl.col("announcement_day_index_next").is_not_null())
                & (pl.col("effective_day_index_next").is_not_null())
                & (pl.col("day_index") >= pl.col("effective_day_index_next"))
            )
            .then((pl.col("announcement_day_index_next") - pl.col("day_index")).cast(pl.Int32))
            .otherwise(None)
            .alias("e_days_to"),
            pl.when(
                (pl.col("announcement_day_index_prev").is_not_null())
                & (pl.col("day_index") >= pl.col("effective_day_index_prev"))
            )
            .then((pl.col("day_index") - pl.col("announcement_day_index_prev")).cast(pl.Int32))
            .otherwise(None)
            .alias("e_days_since"),
        ]
    )

    # Event-day flag
    features = features.with_columns(
        [
            pl.when((pl.col("e_days_to") == 0) | (pl.col("e_days_since") == 0))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("e_is_E0")
        ]
    )

    # Proximity flags
    for w in win:
        pre_cond = pl.col("e_days_to").is_not_null() & (pl.col("e_days_to") >= 1) & (pl.col("e_days_to") <= w)
        post_cond = pl.col("e_days_since").is_not_null() & (pl.col("e_days_since") >= 1) & (pl.col("e_days_since") <= w)
        if w == 1:
            pre_cond = pl.col("e_days_to") == 1
            post_cond = pl.col("e_days_since") == 1

        features = features.with_columns(
            [
                pl.when(pre_cond).then(1).otherwise(0).cast(pl.Int8).alias(f"e_win_pre{w}"),
                pl.when(post_cond).then(1).otherwise(0).cast(pl.Int8).alias(f"e_win_post{w}"),
            ]
        )

    # Leak detection: future availability must never exceed as-of timestamp
    leak_next = features.filter(
        pl.col("e_days_to").is_not_null()
        & pl.col("available_ts_next").is_not_null()
        & (pl.col("available_ts_next") > pl.col("_asof_ts"))
    ).height
    leak_prev = features.filter(
        pl.col("e_days_since").is_not_null()
        & pl.col("available_ts_prev").is_not_null()
        & (pl.col("available_ts_prev") > pl.col("_asof_ts"))
    ).height
    leak_count = leak_next + leak_prev
    if leak_count > 0:
        raise RuntimeError(f"Earnings feature leak detected: {leak_count} rows expose future data")

    # Restore original order and drop helper columns
    drop_cols = {
        c
        for c in features.columns
        if c.endswith("_prev") or c.endswith("_next") or c in {"day_index", "_asof_ts", "available_ts"}
    }
    # Keep availability timestamps for diagnostics before drop-list filtering
    diagnostic_cols = {
        "available_ts_prev": "e_prev_available_ts",
        "available_ts_next": "e_next_available_ts",
    }
    features = features.rename({k: v for k, v in diagnostic_cols.items() if k in features.columns})

    # Prepare final column order
    feature_cols = ["e_days_to", "e_days_since", "e_is_E0"]
    for w in win:
        feature_cols.extend([f"e_win_pre{w}", f"e_win_post{w}"])

    features = features.sort("_row_nr")
    retain_cols = [col for col in features.columns if col not in drop_cols]
    features = features.select(retain_cols)

    # Ensure diagnostic columns exist (may be absent if not joined)
    for col in ("e_prev_available_ts", "e_next_available_ts"):
        if col not in features.columns:
            features = features.with_columns(pl.lit(None).alias(col))

    base_cols = [c for c in base.columns if c != "_row_nr"]
    ordered = features.sort("_row_nr").select(
        base_cols + feature_cols + ["e_prev_available_ts", "e_next_available_ts", "_row_nr"]
    )
    ordered = ordered.drop("_row_nr")

    return ordered
