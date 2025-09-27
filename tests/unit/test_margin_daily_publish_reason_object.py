import polars as pl

from src.gogooku3.features.margin_daily import add_publish_reason_flags


def test_publish_reason_object_dict_handling():
    # PublishReason as dict-like (Polars Object)
    df = pl.DataFrame(
        {
            "PublishReason": [
                {"Restricted": "1", "DailyPublication": "0", "Monitoring": "0", "RestrictedByJSF": "0", "PrecautionByJSF": "0", "UnclearOrSecOnAlert": "0"},
                {"Restricted": "0", "DailyPublication": "1", "Monitoring": "0", "RestrictedByJSF": "0", "PrecautionByJSF": "0", "UnclearOrSecOnAlert": "0"},
            ]
        }
    )

    out = add_publish_reason_flags(df)
    cols = [c for c in out.columns if c.startswith("dmi_reason_")]
    assert "dmi_reason_restricted" in cols
    assert out.select(pl.col("dmi_reason_restricted")).to_series().to_list() == [1, 0]
    assert out.select(pl.col("dmi_reason_dailypublication")).to_series().to_list() == [0, 1]
    # Count should reflect sum of flags
    assert out.select(pl.col("dmi_reason_count")).to_series().to_list() == [1, 1]

