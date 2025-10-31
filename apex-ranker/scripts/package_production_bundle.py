#!/usr/bin/env python3
"""Package trained model, config, and panel cache into a deployable bundle."""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package APEX-Ranker production bundle")
    parser.add_argument("--model", required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument(
        "--panel-cache",
        required=False,
        help="Panel cache directory to include (optional)",
    )
    parser.add_argument(
        "--output",
        default="production/apex_ranker_bundle.tar.gz",
        help="Output tar.gz path",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata JSON output path",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Bundle version tag (default: ISO timestamp)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    config_path = Path(args.config).expanduser()
    panel_cache_dir = Path(args.panel_cache).expanduser() if args.panel_cache else None
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    version = args.version or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    metadata = {
        "version": version,
        "model": str(model_path.resolve()),
        "config": str(config_path.resolve()),
        "panel_cache": str(panel_cache_dir.resolve()) if panel_cache_dir else None,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model_path, arcname="model/apex_ranker_model.pt")
        tar.add(config_path, arcname="config/model_config.yaml")
        if panel_cache_dir and panel_cache_dir.exists():
            tar.add(panel_cache_dir, arcname="panel_cache")
        info = tarfile.TarInfo("metadata.json")
        metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
        info.size = len(metadata_bytes)
        tar.addfile(info, fileobj=io.BytesIO(metadata_bytes))

    if args.metadata:
        metadata_path = Path(args.metadata).expanduser()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    print(f"[Package] Bundle written to {output_path} (version {version})")


if __name__ == "__main__":
    main()
