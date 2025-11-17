#!/usr/bin/env python3
"""
Generate detailed null column report for gogooku5 datasets.
Saves results to a markdown file for easy review.
"""

import polars as pl
import sys
from pathlib import Path
from datetime import datetime

def analyze_null_columns(parquet_path: str, dataset_name: str):
    """Analyze a parquet file for all-null columns."""
    print(f"Analyzing: {dataset_name}...")

    try:
        df = pl.read_parquet(parquet_path)
        total_rows = len(df)
        null_counts = df.null_count()

        all_null_cols = []
        high_null_cols = []

        for col in df.columns:
            null_count = null_counts[col][0]
            null_pct = (null_count / total_rows) * 100

            if null_count == total_rows:
                all_null_cols.append(col)
            elif null_pct > 95:
                high_null_cols.append((col, null_pct))

        return {
            'dataset_name': dataset_name,
            'total_rows': total_rows,
            'total_cols': len(df.columns),
            'all_null': sorted(all_null_cols),
            'high_null': sorted(high_null_cols, key=lambda x: x[1], reverse=True)
        }
    except Exception as e:
        print(f"Error analyzing {dataset_name}: {e}")
        return None

def main():
    base_path = Path("/workspace/gogooku3/gogooku5/data/output/datasets")
    output_path = Path("/workspace/gogooku3/gogooku5/docs/NULL_COLUMNS_REPORT.md")

    datasets = [
        ("ml_dataset_2024_full.parquet", "2024 Dataset"),
        ("ml_dataset_2025_full.parquet", "2025 Dataset"),
        ("ml_dataset_2024_2025_full_for_apex.parquet", "2024-2025 Combined (APEX)"),
    ]

    results = []
    for filename, name in datasets:
        filepath = base_path / filename
        if filepath.exists():
            result = analyze_null_columns(str(filepath), name)
            if result:
                results.append(result)

    # Generate markdown report
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# NULL Columns Report - gogooku5\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")

        # Summary table
        f.write("| Dataset | Total Rows | Total Cols | All-NULL (100%) | High-NULL (>95%) |\n")
        f.write("|---------|------------|------------|-----------------|------------------|\n")
        for r in results:
            f.write(f"| {r['dataset_name']} | {r['total_rows']:,} | {r['total_cols']:,} | "
                   f"{len(r['all_null'])} | {len(r['high_null'])} |\n")

        # Find common all-null columns across all datasets
        if results:
            common_null = set(results[0]['all_null'])
            for r in results[1:]:
                common_null &= set(r['all_null'])

            f.write(f"\n## Common All-NULL Columns Across All Datasets\n\n")
            f.write(f"**Count**: {len(common_null)} columns\n\n")

            if common_null:
                # Group by prefix for better organization
                prefix_groups = {}
                for col in sorted(common_null):
                    prefix = col.split('_')[0]
                    if prefix not in prefix_groups:
                        prefix_groups[prefix] = []
                    prefix_groups[prefix].append(col)

                for prefix in sorted(prefix_groups.keys()):
                    f.write(f"\n### {prefix.upper()} Features ({len(prefix_groups[prefix])} columns)\n\n")
                    for col in prefix_groups[prefix]:
                        f.write(f"- `{col}`\n")
            else:
                f.write("✅ No columns are all-null across all datasets\n\n")

        # Detailed per-dataset analysis
        f.write("\n---\n\n")
        f.write("## Detailed Analysis by Dataset\n\n")

        for r in results:
            f.write(f"\n### {r['dataset_name']}\n\n")
            f.write(f"- **Total Rows**: {r['total_rows']:,}\n")
            f.write(f"- **Total Columns**: {r['total_cols']:,}\n")
            f.write(f"- **All-NULL Columns**: {len(r['all_null'])}\n")
            f.write(f"- **High-NULL Columns (>95%)**: {len(r['high_null'])}\n\n")

            # Group all-null columns by prefix
            prefix_groups = {}
            for col in r['all_null']:
                prefix = col.split('_')[0]
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(col)

            f.write(f"#### All-NULL Columns by Feature Group\n\n")
            for prefix in sorted(prefix_groups.keys()):
                f.write(f"**{prefix.upper()}** ({len(prefix_groups[prefix])} columns):\n")
                for col in sorted(prefix_groups[prefix]):
                    f.write(f"- `{col}`\n")
                f.write("\n")

            # High-null columns
            if r['high_null']:
                f.write(f"#### High-NULL Columns (>95% but <100%)\n\n")
                f.write("| Column | NULL % |\n")
                f.write("|--------|--------|\n")
                for col, pct in r['high_null']:
                    f.write(f"| `{col}` | {pct:.2f}% |\n")
                f.write("\n")

    print(f"\n✅ Report saved to: {output_path}")
    print(f"   View with: cat {output_path}")
    return output_path

if __name__ == "__main__":
    main()
