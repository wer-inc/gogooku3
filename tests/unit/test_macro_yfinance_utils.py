import pandas as pd
import polars as pl
import pytest
from pathlib import Path
from types import SimpleNamespace

from src.features.macro.btc import load_btc_history
from src.features.macro.fx import load_fx_history
from src.features.macro.vix import load_vix_history, shift_to_next_business_day
from src.features.macro.yfinance_utils import (
    ensure_yfinance_available,
    get_yfinance_module,
    flatten_yfinance_columns,
    is_yfinance_available,
    resolve_cached_parquet,
)


def _make_yfinance_frame(ticker: str) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, name="Date")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    data = {
        (field, ticker): [float(i + idx) for idx in range(len(index))]
        for i, field in enumerate(fields, start=1)
    }
    return pd.DataFrame(data, index=index)


def test_flatten_yfinance_columns_drops_single_ticker_level() -> None:
    raw = _make_yfinance_frame("^VIX")
    flattened = flatten_yfinance_columns(raw, ticker="^VIX")
    assert list(flattened.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    reset = flattened.reset_index()
    reset = flatten_yfinance_columns(reset, ticker="^VIX")
    assert "Date" in reset.columns


@pytest.fixture(autouse=True)
def reset_yfinance_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.features.macro.yfinance_utils._YFINANCE_CACHE",
        None,
        raising=False,
    )


@pytest.mark.parametrize(
    ("loader", "ticker"),
    [
        (load_vix_history, "^VIX"),
        (load_fx_history, "JPY=X"),
        (load_btc_history, "BTC-USD"),
    ],
)
def test_load_history_handles_multiindex_columns(
    loader, ticker: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = loader.__module__
    target_module = __import__(module, fromlist=["dummy"])

    def fake_download(*args, **kwargs) -> pd.DataFrame:
        requested = args[0] if args else kwargs.get("tickers", ticker)
        return _make_yfinance_frame(str(requested))

    monkeypatch.setattr(
        target_module,
        "get_yfinance_module",
        lambda **kwargs: SimpleNamespace(download=fake_download),
    )

    if loader is load_fx_history:
        result = loader("2024-01-01", "2024-01-05", ticker=ticker, force_refresh=True)
    elif loader is load_btc_history:
        result = loader("2024-01-01", "2024-01-05", ticker=ticker, force_refresh=True)
    else:
        result = loader("2024-01-01", "2024-01-05", force_refresh=True)

    assert isinstance(result, pl.DataFrame)
    assert not result.is_empty()
    assert "Date" in result.columns
    assert "Close" in result.columns


def test_ensure_yfinance_available_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "yfinance":
            raise ImportError("yfinance missing")
        return original_import(name, package)

    monkeypatch.setattr(
        "src.features.macro.yfinance_utils.importlib.import_module",
        fake_import,
    )

    with pytest.raises(RuntimeError):
        ensure_yfinance_available()

    assert is_yfinance_available() is False


def test_shift_to_next_business_day_handles_callable_expr() -> None:
    import datetime as dt

    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(5)]
    df = pl.DataFrame({"Date": dates})

    shifted = shift_to_next_business_day(
        df,
        business_days=["2024-01-01", "2024-01-04", "2024-01-05", "2024-01-08"],
    )

    assert "effective_date" in shifted.columns
    assert shifted["effective_date"].dtype == pl.Date


def test_get_yfinance_module_configures_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import importlib

    class DummyYF:
        def __init__(self) -> None:
            self.cache_location: str | None = None
            self.tz_location: str | None = None

        def download(self, *args, **kwargs):  # pragma: no cover - unused in test
            raise RuntimeError("not implemented")

        def set_cache_location(self, path: str | None) -> None:
            self.cache_location = path

        def set_tz_cache_location(self, path: str | None) -> None:
            self.tz_location = path

    dummy = DummyYF()

    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "yfinance":
            return dummy
        return real_import(name, package)

    monkeypatch.setenv("YFINANCE_CACHE_DIR", str(tmp_path / "yf_cache"))
    monkeypatch.setattr(
        "src.features.macro.yfinance_utils._YFINANCE_CACHE", None, raising=False
    )
    monkeypatch.setattr(
        "src.features.macro.yfinance_utils._CACHE_CONFIGURED", False, raising=False
    )
    monkeypatch.setattr("importlib.import_module", fake_import)

    get_yfinance_module()

    assert dummy.cache_location is not None
    assert dummy.tz_location is not None


def test_resolve_cached_parquet_finds_best_match(tmp_path: Path) -> None:
    target = tmp_path / "fx_usdjpy_history_20210923_20251027.parquet"
    existing = tmp_path / "fx_usdjpy_history_20210922_20251027.parquet"
    existing.touch()

    resolved = resolve_cached_parquet(
        target,
        prefix="fx_usdjpy",
        start="2022-10-27",
        end="2025-10-27",
    )
    assert resolved == existing
