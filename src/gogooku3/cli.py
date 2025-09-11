"""Command-line interface for gogooku3."""

import sys
import argparse
from pathlib import Path

from gogooku3.utils.settings import settings

def cmd_data(args: argparse.Namespace) -> int:
    """Data validation and preparation commands."""
    root = Path(settings.project_root)
    required = [
        root / "configs" / "train" / "adaptive.yaml",
        root / "data",
        root / "src" / "gogooku3"
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
    """Inference command."""
    print(f"🔮 Inference configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Input: {args.input_path}")
    print(f"  Output: {args.output_path}")
    print("⚠️ Inference pipeline not yet wired to CLI.")
    return 2

def main() -> None:
    """Main CLI entry point for gogooku3."""
    parser = argparse.ArgumentParser(
        prog="gogooku3",
        description="Gogooku3 – 金融MLシステム（最小CLI）"
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
    p_infer.set_defaults(func=cmd_infer)
    
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
