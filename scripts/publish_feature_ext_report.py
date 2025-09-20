from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Insert ablation results into the implementation report")
    p.add_argument("--template", type=Path, default=Path("reports/feature_ext_implementation_report.md"))
    p.add_argument("--ablation", type=Path, default=Path("output/ablation_report.md"))
    p.add_argument("--out", type=Path, default=Path("reports/feature_ext_implementation_report.md"))
    p.add_argument("--marker", type=str, default="[[ABLATION_TABLE]]")
    return p.parse_args()


def extract_table(md: str) -> str:
    lines = md.splitlines()
    start = None
    # Find the first markdown table (line starting with | Variant | ...)
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") and "Variant" in ln and "RankIC" in ln:
            start = i
            break
    if start is None:
        return "(No ablation table found)"
    # Collect contiguous table lines
    table: list[str] = []
    for ln in lines[start:]:
        if ln.strip().startswith("|"):
            table.append(ln)
        else:
            break
    return "\n".join(table)


def main() -> None:
    args = parse_args()
    tpl = args.template.read_text(encoding="utf-8")
    abl = args.ablation.read_text(encoding="utf-8") if args.ablation.exists() else ""
    table = extract_table(abl) if abl else "(Run 'gogooku3 ablation ...' to generate results)"
    out = tpl.replace(args.marker, table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out, encoding="utf-8")
    print(f"✅ Report published: {args.out}")


if __name__ == "__main__":
    main()

