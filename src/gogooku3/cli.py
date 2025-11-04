"""Command-line interface for gogooku3."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from gogooku3.decide import build_pseudo_vix
from gogooku3.detect import (
    change_point_score,
    evaluate_vus_pr,
    residual_q_score,
    score_to_ranges,
    spectral_residual_score,
    stack_and_score,
)
from gogooku3.detect.label_store import load_labels, merge_labels, save_labels
from gogooku3.forecast import TFTAdapter, timesfm_predict
from gogooku3.metrics.forecast_metrics import mae, smape, weighted_quantile_loss
from gogooku3.training.tft_hpo import run_tft_hpo
from gogooku3.training.tft_trainer import TFTTrainerConfig, train_tft_quantile
from gogooku3.utils.settings import settings


def cmd_data(args: argparse.Namespace) -> int:
    """Data validation and preparation commands."""
    root = Path(settings.project_root)
    required = [
        root / "configs" / "train" / "adaptive.yaml",
        root / "data",
        root / "src" / "gogooku3",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("❌ Missing files/directories:")
        for m in missing:
            print(f" - {m}")
        return 1
    print("✅ data validate: OK")
    print(f"📂 Project root: {settings.project_root}")
    print(f"📁 Config directory: {root / 'configs'}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Training command with dry-run capability."""
    if args.dry_run:
        print("🧪 [dry-run] Safe Training Pipeline Steps:")
        print(" 1) Load dataset (ProductionDatasetV3)")
        print(" 2) Generate quality features")
        print(" 3) Walk-Forward split (with embargo>=20 days)")
        print(" 4) Cross-sectional normalization (fit on train only)")
        print(" 5) LightGBM baseline training")
        print(" 6) Graph construction")
        print(" 7) Performance report generation")
        print("")
        print("✅ Pipeline order prevents data leakage:")
        print("   - Split BEFORE normalization")
        print("   - Fit normalizer on train fold only")
        print("   - Transform train/test with same statistics")
        return 0

    print("⚠️ Full training pipeline not yet wired to CLI.")
    print("💡 Use --dry-run to see pipeline steps.")
    print("💡 For now, use: python scripts/run_safe_training.py")
    return 2


def cmd_infer(args: argparse.Namespace) -> int:
    """Inference command with TENT support."""
    print("🔮 Inference configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Input: {args.input_path}")
    print(f"  Output: {args.output_path}")

    if args.tta == "tent":
        print("  🧠 TENT adaptation enabled:")
        print(f"    Steps per batch: {args.tta_steps}")
        print(f"    Learning rate: {args.tta_lr}")
        print("    Target: BatchNorm parameters only")
        print("    Method: Entropy minimization")

        # Run TENT inference
        try:
            from ..inference.tent_inference_runner import run_tent_inference

            result = run_tent_inference(
                model_path=args.model_path,
                input_path=args.input_path,
                output_path=args.output_path,
                tent_steps=args.tta_steps,
                tent_lr=args.tta_lr,
            )
            if result["success"]:
                print("✅ TENT inference completed successfully")
                print(f"   Processed: {result.get('batches_processed', 0)} batches")
                print(
                    f"   Avg entropy improvement: {result.get('avg_entropy_improvement', 0):.4f}"
                )
                print(f"   Final confidence: {result.get('avg_confidence', 0):.3f}")
                return 0
            else:
                print(
                    f"❌ TENT inference failed: {result.get('error', 'Unknown error')}"
                )
                return 1

        except ImportError:
            print("⚠️ TENT inference runner not available.")
            print("💡 Run: python -m src.inference.tent_inference_runner --help")
            return 2
        except Exception as e:
            print(f"❌ TENT inference error: {e}")
            return 1

    elif args.tta == "off":
        print("  Standard inference (no adaptation)")
        print("⚠️ Standard inference pipeline not yet wired to CLI.")
        return 2

    else:
        print(f"❌ Unknown TTA method: {args.tta}")
        return 1


def main() -> None:
    """Main CLI entry point for gogooku3."""
    parser = argparse.ArgumentParser(
        prog="gogooku3", description="Gogooku3 – 金融MLシステム（最小CLI）"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data command
    p_data = subparsers.add_parser("data", help="データ検証/準備")
    p_data.set_defaults(func=cmd_data)

    # Training command
    p_train = subparsers.add_parser("train", help="学習の実行/ドライラン")
    p_train.add_argument("--dry-run", action="store_true", help="配線確認のみ")
    p_train.add_argument("--config", type=Path, help="設定ファイルパス")
    p_train.set_defaults(func=cmd_train)

    # Inference command
    p_infer = subparsers.add_parser("infer", help="推論")
    p_infer.add_argument("--model-path", required=True, help="モデルファイルパス")
    p_infer.add_argument("--input-path", required=True, help="入力データパス")
    p_infer.add_argument("--output-path", required=True, help="出力パス")
    p_infer.add_argument(
        "--tta", choices=["off", "tent"], default="off", help="推論時適応(TTA)方式"
    )
    p_infer.add_argument(
        "--tta-steps", type=int, default=2, help="TTAステップ数（各バッチ）"
    )
    p_infer.add_argument("--tta-lr", type=float, default=1e-4, help="TTA学習率")
    p_infer.set_defaults(func=cmd_infer)

    # Forecast (TimesFM-style) command
    p_fc = subparsers.add_parser("forecast", help="ゼロショット予測 (TimesFMスタイル)")
    p_fc.add_argument("--obs", required=True, help="観測データ (json/csv/parquet)")
    p_fc.add_argument("--horizons", default="1,5,10,20,30", help="予測ホライズンCSV")
    p_fc.add_argument("--out", required=True, help="出力先 (json)")

    def _cmd_fc(args: argparse.Namespace) -> int:
        def _read(path: str) -> pd.DataFrame:
            if path.endswith(".json") or path.endswith(".jsonl"):
                return pd.read_json(path)
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)

        df_obs = _read(args.obs)
        horizons = [int(x) for x in args.horizons.split(",") if x]
        fc = timesfm_predict(df_obs, horizons=horizons)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                json.loads(fc.to_json(orient="records", date_format="iso")),
                f,
                ensure_ascii=False,
            )
        print(f"✅ Forecasts saved: {args.out} ({len(fc)} rows)")
        return 0

    p_fc.set_defaults(func=_cmd_fc)

    # Forecast (TFT-like minimal quantile forecaster)
    p_tft = subparsers.add_parser(
        "forecast-tft", help="最小TFT風 量子回帰 (学習→即時予測)"
    )
    p_tft.add_argument(
        "--obs", required=True, help="観測データ (json/csv/parquet; id,ts,y,features)"
    )
    p_tft.add_argument("--horizons", default="1,5,10,20,30")
    p_tft.add_argument("--out", required=True, help="出力先 (json)")

    def _cmd_tft(args: argparse.Namespace) -> int:
        def _read(path: str) -> pd.DataFrame:
            if path.endswith(".json") or path.endswith(".jsonl"):
                return pd.read_json(path)
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)

        df_obs = _read(args.obs)
        horizons = [int(x) for x in args.horizons.split(",") if x]
        model = TFTAdapter(horizons=horizons)
        try:
            model.fit(df_obs)
            fc = model.predict(df_obs)
        except Exception:
            # Fallback if fit fails
            fc = timesfm_predict(df_obs, horizons=horizons)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                json.loads(fc.to_json(orient="records", date_format="iso")),
                f,
                ensure_ascii=False,
            )
        print(f"✅ TFT-like forecasts saved: {args.out} ({len(fc)} rows)")
        return 0

    p_tft.set_defaults(func=_cmd_tft)

    # Detect command (ensemble → ranges → VUS-PR)
    p_dt = subparsers.add_parser("detect", help="残差×CP×SR アンサンブル→範囲化→VUS-PR")
    p_dt.add_argument("--obs", required=True, help="観測データ (json/csv/parquet)")
    p_dt.add_argument("--fcst", required=True, help="予測データ (json/csv/parquet)")
    p_dt.add_argument("--h", type=int, default=1, help="残差のホライズン")
    p_dt.add_argument("--out-ranges", required=True, help="異常レンジ出力先 (json)")
    p_dt.add_argument("--labels", default=None, help="区間ラベル (json, optional)")
    p_dt.add_argument("--events", default=None, help="イベント点 (json/csv/parquet)")
    p_dt.add_argument("--event-pre", type=int, default=0)
    p_dt.add_argument("--event-post", type=int, default=0)
    p_dt.add_argument(
        "--eval-iou", type=float, default=None, help="IoUしきい値 (VUS-PR IoU版)"
    )
    p_dt.add_argument("--min-len", type=int, default=2)
    p_dt.add_argument("--perc", type=float, default=0.95)
    p_dt.add_argument("--metrics-out", default=None, help="VUS-PR結果をJSON保存")

    def _cmd_dt(args: argparse.Namespace) -> int:
        def _read(path: str) -> pd.DataFrame:
            if path.endswith(".json") or path.endswith(".jsonl"):
                return pd.read_json(path)
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)

        df_obs = _read(args.obs)
        df_fc = _read(args.fcst)
        r = residual_q_score(df_obs, df_fc, horizon=args.h)
        cp = change_point_score(df_obs)
        sr = spectral_residual_score(df_obs)
        ens = stack_and_score([r, cp, sr])
        ranges = score_to_ranges(ens, min_len=args.min_len, perc=args.perc)
        payload = [
            {
                "id": rg.id,
                "start": rg.start.strftime("%Y-%m-%d"),
                "end": rg.end.strftime("%Y-%m-%d"),
                "score": rg.score,
                "type": rg.type,
            }
            for rg in ranges
        ]
        with open(args.out_ranges, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"✅ Ranges saved: {args.out_ranges} ({len(ranges)} ranges)")
        if args.labels or args.events:
            from gogooku3.detect import RangeLabel

            gold: list[RangeLabel] = []
            if args.labels:
                labels_df = _read(args.labels)
                label_ranges = [
                    {
                        "id": str(row["id"]),
                        "start": pd.to_datetime(row["start"]),
                        "end": pd.to_datetime(row["end"]),
                    }
                    for _, row in labels_df.iterrows()
                ]
                gold.extend(
                    [
                        RangeLabel(id=lr["id"], start=lr["start"], end=lr["end"])
                        for lr in label_ranges
                    ]
                )
            if args.events:
                events_df = _read(args.events)
                from gogooku3.detect.label_generators import events_to_ranges

                ids = df_obs["id"].astype(str).unique().tolist()
                gold.extend(
                    events_to_ranges(
                        events_df,
                        ids,
                        pre_days=args.event_pre,
                        post_days=args.event_post,
                    )
                )
            if args.eval_iou is not None:
                from gogooku3.detect.ranges import evaluate_vus_pr_iou

                vs = evaluate_vus_pr_iou(ranges, gold, min_iou=args.eval_iou)
            else:
                vs = evaluate_vus_pr(ranges, gold)
            print(f"📈 VUS-PR: {vs['vus_pr']:.4f}")
            if args.metrics_out:
                with open(args.metrics_out, "w", encoding="utf-8") as f:
                    json.dump({"VUS_PR": vs.get("vus_pr", 0.0)}, f, ensure_ascii=False)
        return 0

    p_dt.set_defaults(func=_cmd_dt)

    # Index command (Pseudo-VIX)
    p_ix = subparsers.add_parser("index", help="指数系ユーティリティ")
    sp = p_ix.add_subparsers(dest="index_cmd", required=True)
    p_esvi = sp.add_parser("esvi", help="30日先ボラ指数 (Pseudo-VIX)")
    p_esvi.add_argument("--fcst", required=True, help="予測データ (json/csv/parquet)")
    p_esvi.add_argument("--out", required=True, help="出力先 (json)")
    p_esvi.add_argument("--name", default="ESVI_JP", help="指数名")
    p_esvi.add_argument("--weights", default=None, help="重みファイル (id,weight)")

    def _cmd_esvi(args: argparse.Namespace) -> int:
        def _read(path: str) -> pd.DataFrame:
            if path.endswith(".json") or path.endswith(".jsonl"):
                return pd.read_json(path)
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)

        df_fc = _read(args.fcst)
        wdf = None
        if args.weights:
            wdf = _read(args.weights)
        out = build_pseudo_vix(df_fc, index_name=args.name, weights=wdf)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                json.loads(out.to_json(orient="records", date_format="iso")),
                f,
                ensure_ascii=False,
            )
        print(f"✅ Index saved: {args.out} ({len(out)} rows)")
        return 0

    p_esvi.set_defaults(func=_cmd_esvi)

    # Label Store commands
    p_labels = subparsers.add_parser("labels", help="ラベルストア操作")
    sp_l = p_labels.add_subparsers(dest="labels_cmd", required=True)

    p_ls = sp_l.add_parser("ls", help="ラベルファイル情報")
    p_ls.add_argument("--path", required=True)

    def _cmd_labels_ls(args: argparse.Namespace) -> int:
        labels = load_labels(args.path)
        print(f"📦 Labels: {len(labels)} records from {args.path}")
        if labels:
            ids = sorted({l.id for l in labels})
            print(f"  IDs: {len(ids)} unique")
        return 0

    p_ls.set_defaults(func=_cmd_labels_ls)

    p_merge = sp_l.add_parser("merge", help="2つのラベルをマージ")
    p_merge.add_argument("--a", required=True)
    p_merge.add_argument("--b", required=True)
    p_merge.add_argument("--out", required=True)

    def _cmd_labels_merge(args: argparse.Namespace) -> int:
        la = load_labels(args.a)
        lb = load_labels(args.b)
        merged = merge_labels(la, lb)
        save_labels(args.out, merged)
        print(f"✅ Merged {len(la)}+{len(lb)} → {len(merged)} into {args.out}")
        return 0

    p_merge.set_defaults(func=_cmd_labels_merge)

    # Compare forecasts (Champion/Challenger)
    p_cmp = subparsers.add_parser("compare", help="予測のKPI比較 (sMAPE/MAE/WQL)")
    p_cmp.add_argument("--obs", required=True, help="観測データ (id,ts,y)")
    p_cmp.add_argument("--a", required=True, help="予測A (json/csv/parquet)")
    p_cmp.add_argument("--b", required=True, help="予測B (json/csv/parquet)")
    p_cmp.add_argument("--name-a", default="A")
    p_cmp.add_argument("--name-b", default="B")
    p_cmp.add_argument("--h", type=int, default=1, help="評価ホライズン")

    def _cmd_compare(args: argparse.Namespace) -> int:
        def _read(path: str) -> pd.DataFrame:
            if path.endswith(".json") or path.endswith(".jsonl"):
                return pd.read_json(path)
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)

        obs = _read(args.obs)
        fa = _read(args.a)
        fb = _read(args.b)
        # filter by horizon and join on id+ts
        fa = fa[fa["h"] == args.h]
        fb = fb[fb["h"] == args.h]
        key = ["id", "ts"]
        da = obs.merge(fa, on=key, how="inner", suffixes=(None, None))
        db = obs.merge(fb, on=key, how="inner", suffixes=(None, None))
        if len(da) == 0 or len(db) == 0:
            print("❌ Not enough overlapping rows to compare. Check alignment.")
            return 2
        ya = da["y"].to_numpy()
        yha = da["y_hat"].to_numpy()
        yb = db["y"].to_numpy()
        yhb = db["y_hat"].to_numpy()
        metrics = {}
        metrics[args.name_a] = {
            "sMAPE": smape(ya, yha),
            "MAE": mae(ya, yha),
            "WQL": weighted_quantile_loss(ya, da),
        }
        metrics[args.name_b] = {
            "sMAPE": smape(yb, yhb),
            "MAE": mae(yb, yhb),
            "WQL": weighted_quantile_loss(yb, db),
        }
        print(json.dumps(metrics, indent=2))
        return 0

    p_cmp.set_defaults(func=_cmd_compare)

    # Promote winner (Champion/Challenger) by metric
    p_prom = subparsers.add_parser("promote", help="KPIで勝者選定し出力")
    p_prom.add_argument("--obs", required=True)
    p_prom.add_argument("--a", required=True)
    p_prom.add_argument("--b", required=True)
    p_prom.add_argument("--name-a", default="A")
    p_prom.add_argument("--name-b", default="B")
    p_prom.add_argument("--h", type=int, default=5)
    p_prom.add_argument("--metric", choices=["WQL", "MAE", "sMAPE"], default="WQL")
    p_prom.add_argument(
        "--delta", type=float, default=0.0, help="最小改善幅(>なら昇格)"
    )

    def _cmd_promote(args: argparse.Namespace) -> int:
        def _read(path: str) -> pd.DataFrame:
            if path.endswith(".json") or path.endswith(".jsonl"):
                return pd.read_json(path)
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)

        obs = _read(args.obs)
        fa = _read(args.a)
        fb = _read(args.b)
        fa = fa[fa["h"] == args.h]
        fb = fb[fb["h"] == args.h]
        key = ["id", "ts"]
        da = obs.merge(fa, on=key, how="inner")
        db = obs.merge(fb, on=key, how="inner")
        if len(da) == 0 or len(db) == 0:
            print("❌ Not enough overlap to decide")
            return 2
        ya = da["y"].to_numpy()
        yha = da["y_hat"].to_numpy()
        yb = db["y"].to_numpy()
        yhb = db["y_hat"].to_numpy()
        A = {
            "WQL": weighted_quantile_loss(ya, da),
            "MAE": mae(ya, yha),
            "sMAPE": smape(ya, yha),
        }
        B = {
            "WQL": weighted_quantile_loss(yb, db),
            "MAE": mae(yb, yhb),
            "sMAPE": smape(yb, yhb),
        }
        # Lower is better for all three defined metrics
        a_val = A[args.metric]
        b_val = B[args.metric]
        winner = args.name_a if (a_val + args.delta) < b_val else args.name_b
        print(
            json.dumps(
                {
                    "metric": args.metric,
                    "horizon": args.h,
                    args.name_a: A,
                    args.name_b: B,
                    "winner": winner,
                    "improvement": float(b_val - a_val)
                    if winner == args.name_a
                    else float(a_val - b_val),
                },
                indent=2,
            )
        )
        return 0

    p_prom.set_defaults(func=_cmd_promote)

    # TFT training (Purged CV + WQL)
    p_tfttrain = subparsers.add_parser(
        "tft-train", help="TFT最小学習 (Purged CV + WQL)"
    )
    p_tfttrain.add_argument("--obs", required=True)
    p_tfttrain.add_argument("--known", default=None)
    p_tfttrain.add_argument("--static", default=None)
    p_tfttrain.add_argument("--horizons", default="1,5,10,20,30")
    p_tfttrain.add_argument("--n-splits", type=int, default=3)
    p_tfttrain.add_argument("--embargo", type=int, default=20)
    p_tfttrain.add_argument("--feature-config", default=None, help="特徴量設定YAML")
    p_tfttrain.add_argument(
        "--cat-encode", choices=["codes", "hash", "target"], default="codes"
    )
    p_tfttrain.add_argument(
        "--cat-cols", default=None, help="カテゴリ列名CSV（未指定なら自動検出）"
    )
    p_tfttrain.add_argument("--hash-buckets", type=int, default=64)

    def _cmd_tfttrain(args: argparse.Namespace) -> int:
        def _read(path: str | None) -> pd.DataFrame | None:
            if not path:
                return None
            if path.endswith(".json") or path.endswith(".jsonl"):
                return pd.read_json(path)
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)

        obs = _read(args.obs)
        known = _read(args.known)
        static = _read(args.static)
        horizons = [int(x) for x in args.horizons.split(",") if x]
        from gogooku3.features.feature_params import FeatureParams

        feature_params = FeatureParams()
        if args.feature_config:
            import yaml

            with open(args.feature_config, encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            # load keys if present
            for k, v in y.get("features", {}).items():
                if hasattr(feature_params, k):
                    setattr(feature_params, k, v)
        cfg = TFTTrainerConfig(
            horizons=horizons,
            embargo_days=args.embargo,
            n_splits=args.n_splits,
            features=feature_params,
            cat_encode=args.cat_encode,
            cat_cols=[c.strip() for c in args.cat_cols.split(",")]
            if args.cat_cols
            else None,
            hash_buckets=args.hash_buckets,
        )
        out = train_tft_quantile(obs, known, static, cfg)
        print(json.dumps(out, indent=2))
        return 0

    p_tfttrain.set_defaults(func=_cmd_tfttrain)

    # TFT HPO
    p_thpo = subparsers.add_parser("tft-hpo", help="TFT最小HPO (Purged CV + WQL最小化)")
    p_thpo.add_argument("--obs", required=True)
    p_thpo.add_argument("--known", default=None)
    p_thpo.add_argument("--static", default=None)
    p_thpo.add_argument("--horizons", default="1,5,10,20")
    p_thpo.add_argument("--n-splits", type=int, default=3)
    p_thpo.add_argument("--embargo", type=int, default=20)
    p_thpo.add_argument("--trials", type=int, default=20)

    def _cmd_tft_hpo(args: argparse.Namespace) -> int:
        def _read(path: str | None) -> pd.DataFrame | None:
            if not path:
                return None
            if path.endswith(".json") or path.endswith(".jsonl"):
                return pd.read_json(path)
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)

        obs = _read(args.obs)
        known = _read(args.known)
        static = _read(args.static)
        from gogooku3.training.tft_hpo import HPOConfig

        cfg = HPOConfig(
            horizons=[int(x) for x in args.horizons.split(",") if x],
            n_splits=args.n_splits,
            embargo_days=args.embargo,
            trials=args.trials,
        )
        out = run_tft_hpo(obs, known, static, cfg)
        print(json.dumps(out, indent=2))
        return 0

    p_thpo.set_defaults(func=_cmd_tft_hpo)

    # Regression gate for metrics (e.g., VUS_PR or WQL)
    p_gate = subparsers.add_parser("gate", help="回帰ゲート（ベースライン vs 候補）")
    p_gate.add_argument("--baseline", required=True, help="baseline metrics.json")
    p_gate.add_argument("--candidate", required=True, help="candidate metrics.json")
    p_gate.add_argument(
        "--key", required=True, help="メトリクスキー（例: VUS_PR, WQL_h5）"
    )
    p_gate.add_argument("--direction", choices=["min", "max"], required=True)
    p_gate.add_argument("--max-regress", type=float, default=0.0)

    def _cmd_gate(args: argparse.Namespace) -> int:
        from gogooku3.quality.gate import GateRule, check_gate, load_json

        base = load_json(args.baseline)
        cand = load_json(args.candidate)
        res = check_gate(
            base,
            cand,
            GateRule(
                key=args.key, direction=args.direction, max_regress=args.max_regress
            ),
        )
        print(json.dumps(res, indent=2))
        return 0 if res["passed"] else 2

    p_gate.set_defaults(func=_cmd_gate)

    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()


# Note: Extended dataset building, multi-head training, and ablation commands
# were removed as dead code (140+ lines). If needed in the future, they can be
# restored from git history (commit prior to 2025-11-01) and properly integrated.
