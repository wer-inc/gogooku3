#!/usr/bin/env bash
set -euo pipefail

START_DATE=${1:-2020-01-01}
END_DATE=${2:-2020-12-31}
CHUNK_MONTHS=${CHUNK_MONTHS:-3}

echo "📦 Rebuilding chunks for ${START_DATE} → ${END_DATE} (CHUNK_MONTHS=${CHUNK_MONTHS})"
make -f Makefile.dataset build-chunks \
  START="${START_DATE}" \
  END="${END_DATE}" \
  FORCE=1 \
  CHUNK_MONTHS="${CHUNK_MONTHS}"

echo "🔗 Merging completed chunks (ALLOW_PARTIAL=1)..."
make -f Makefile.dataset merge-chunks ALLOW_PARTIAL=1

echo "✅ Rebuild + merge finished."
