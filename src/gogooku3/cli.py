"""Command-line interface for gogooku3."""

import argparse
import sys
from pathlib import Path

from gogooku3 import __version__
from gogooku3.utils.settings import settings


def main() -> None:
    """Main CLI entry point for gogooku3."""
    parser = argparse.ArgumentParser(
        prog="gogooku3",
        description="壊れず・強く・速く 金融ML システム - Gogooku3 Financial ML System",
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"gogooku3 {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument(
        "--config", 
        type=Path, 
        help="Path to training configuration file"
    )
    
    # Data command
    data_parser = subparsers.add_parser("data", help="Data processing commands")
    data_parser.add_argument(
        "--build-dataset", 
        action="store_true", 
        help="Build ML dataset"
    )
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run model inference")
    infer_parser.add_argument(
        "--model-path", 
        type=Path, 
        help="Path to trained model"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    print(f"🚀 Gogooku3 v{__version__} - 壊れず・強く・速く")
    print(f"📂 Project root: {settings.project_root}")
    print(f"🔧 Command: {args.command}")
    
    if args.command == "train":
        print("🎯 Training functionality will be implemented in Phase 3")
    elif args.command == "data":
        print("📊 Data processing functionality will be implemented in Phase 3")  
    elif args.command == "infer":
        print("🔮 Inference functionality will be implemented in Phase 3")
    
    print("✅ CLI framework ready for migration of existing scripts")


if __name__ == "__main__":
    main()