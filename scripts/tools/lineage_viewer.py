#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple lineage viewer")
    ap.add_argument("--dir", default="output", help="output directory")
    ap.add_argument("--file", default="lineage.jsonl", help="lineage file name")
    ap.add_argument("--tail", type=int, default=20, help="last N records to show")
    args = ap.parse_args()
    p = Path(args.dir) / args.file
    if not p.exists():
        print(f"No lineage file: {p}")
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    for line in lines[-args.tail:]:
        try:
            obj = json.loads(line)
            print(f"[{obj.get('ts')}] {obj.get('transformation')}\n  inputs: {obj.get('inputs')}\n  output: {obj.get('output')}\n  meta: {obj.get('metadata')}")
        except Exception:
            print(line)

if __name__ == "__main__":
    main()

