# Forecast × Detect × Decide — Minimal Wiring

This repo now includes a minimal, testable implementation of the three-layer pipeline:

- Forecast: TimesFM-style zero-shot adapter (fallback baseline)
- Detect: Residual-Q, Change-Point, Spectral Residual → ensemble → ranges
- Decide: Pseudo-VIX style 30D volatility index aggregation

## Flat I/O Schema

- Observations: id, ts, y
- Known future/static: placeholders (future TFT integration)
- Forecast output: id, ts, h, y_hat[, p10, p50, p90]
- Anomaly ranges: id, start, end, score[, type]
- Index (ESVI): ts, index, level[, p10, p90]

## Quickstart

- Forecast (baseline TimesFM style):
  `gogooku3 forecast --obs data/obs.json --horizons 1,5,20,30 --out output/forecast.json`

- Forecast (TFT-like minimal, trains then predicts):
  `gogooku3 forecast-tft --obs data/obs.json --horizons 1,5,20,30 --out output/forecast_tft.json`

- Train TFT-like (Purged CV + WQL) with feature config:
  `gogooku3 tft-train --obs data/obs.parquet --known data/known_future.json --static data/static.csv \
    --horizons 1,5,20,30 --n-splits 3 --embargo 20 --feature-config configs/pipeline/features/default.yaml`

- Detect (ensemble → ranges → VUS-PR):
  `gogooku3 detect --obs data/obs.json --fcst output/forecast.json \
    --h 1 --out-ranges output/ranges.json --min-len 2 --perc 0.95 \
    [--labels data/labels.json] [--events data/events.json --event-pre 0 --event-post 1] [--eval-iou 0.25]`

- Index (Pseudo-VIX / ESVI):
  `gogooku3 index esvi --fcst output/forecast.json --out output/esvi.json --name ESVI_JP [--weights data/weights.csv]`

- Labels (store ops):
  - `gogooku3 labels ls --path data/labels.json`
  - `gogooku3 labels merge --a data/labels_manual.json --b data/labels_auto.json --out data/labels_merged.json`

- Compare forecasts (Champion/Challenger):
  `gogooku3 compare --obs data/obs.json --a output/forecast_tft.json --b output/forecast.json --name-a tft --name-b timesfm --h 5`

- Promote winner (auto decision by KPI):
  `gogooku3 promote --obs data/obs.json --a output/forecast_tft.json --b output/forecast.json --name-a tft --name-b timesfm --h 5 --metric WQL --delta 0.0`

## Notes

- The TFT adapter is a placeholder; it proxies to the baseline so the full wiring works today.
- Residual-Q uses rolling median/MAD (window=63) and squashes |z| via logistic.
- Change-point score is a windowed mean-shift statistic; SR is an FFT-based saliency score.
- VUS-PR here integrates PR over score thresholds with greedy 1–1 range matching.

## Config

See `configs/pipeline/three_layer_minimal.yaml` for default parameters.
Feature parameters: `configs/pipeline/features/default.yaml` (KAMA/VIDYA multi-window, fractional diff, rolling quantiles, CS features).

HPO (minimize avg WQL):
`gogooku3 tft-hpo --obs data/obs.parquet --known data/known_future.json --static data/static.csv \
  --horizons 1,5,10,20 --n-splits 3 --embargo 20 --trials 20`

Gate check:
`gogooku3 gate --baseline baseline_metrics.json --candidate cand_metrics.json --key WQL_h5 --direction min --max-regress 0.005`
