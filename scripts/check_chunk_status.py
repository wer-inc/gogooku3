#!/usr/bin/env python3
"""
Check chunk rebuild status
"""
import json
import sys
from pathlib import Path
from datetime import datetime

def main():
    chunks_dir = Path("/workspace/gogooku3/output_g5/chunks")

    print("=" * 80)
    print(f"📊 Chunk Rebuild Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if not chunks_dir.exists():
        print("❌ Chunks directory not found!")
        sys.exit(1)

    status_files = sorted(chunks_dir.glob("*/status.json"))

    if not status_files:
        print("⚠️  No chunks found yet. Build may still be starting...")
        return

    # Count by state
    states = {"completed": 0, "failed": 0, "in_progress": 0, "other": 0}
    chunks_info = []

    for status_file in status_files:
        chunk_name = status_file.parent.name
        try:
            with open(status_file) as f:
                data = json.load(f)

            state = data.get("state", "unknown")
            rows = data.get("rows", 0)
            schema_hash = data.get("feature_schema_hash", "N/A")
            duration = data.get("build_duration_seconds", 0)

            # Categorize
            if state == "completed":
                states["completed"] += 1
                status_icon = "✅"
            elif "failed" in state:
                states["failed"] += 1
                status_icon = "❌"
            elif state == "in_progress":
                states["in_progress"] += 1
                status_icon = "🔄"
            else:
                states["other"] += 1
                status_icon = "❓"

            chunks_info.append({
                "name": chunk_name,
                "state": state,
                "icon": status_icon,
                "rows": rows,
                "hash": schema_hash,
                "duration": duration,
            })
        except Exception as e:
            chunks_info.append({
                "name": chunk_name,
                "state": "error",
                "icon": "⚠️",
                "rows": 0,
                "hash": "N/A",
                "duration": 0,
                "error": str(e)
            })

    # Print summary
    total = len(chunks_info)
    print(f"\n📈 Summary: {total} chunks")
    print(f"   ✅ Completed: {states['completed']}")
    print(f"   ❌ Failed: {states['failed']}")
    print(f"   🔄 In Progress: {states['in_progress']}")
    print(f"   ❓ Other: {states['other']}")

    # Print details
    print(f"\n📋 Chunk Details:\n")
    print(f"{'Chunk':<12} {'State':<20} {'Rows':>10} {'Duration':>10} {'Hash':<20}")
    print("-" * 80)

    for chunk in chunks_info:
        duration_str = f"{chunk['duration']:.1f}s" if chunk['duration'] > 0 else "N/A"
        print(f"{chunk['icon']} {chunk['name']:<10} {chunk['state']:<20} {chunk['rows']:>10,} {duration_str:>10} {chunk['hash'][:16]}")

    print("\n" + "=" * 80)

    # Check for target schema hash
    target_hash = "f077a15d37e1157a"
    matching = sum(1 for c in chunks_info if c['hash'] == target_hash)
    if matching > 0:
        print(f"✅ {matching}/{total} chunks match target schema (Phase 3: {target_hash})")

    # Return exit code
    if states["failed"] > 0:
        print(f"⚠️  {states['failed']} chunk(s) failed!")
        sys.exit(1)
    elif states["completed"] == total and total > 0:
        print(f"🎉 All {total} chunks completed successfully!")
        sys.exit(0)
    else:
        print(f"🔄 Build in progress... ({states['completed']}/{total} completed)")
        sys.exit(0)

if __name__ == "__main__":
    main()
