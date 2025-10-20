# Working Tree WIP Summary — 2025-10-20

## Overview
- The 2025-10-20 health check warning stemmed from 55 tracked/untracked changes across documentation migration, short-selling data fixes, and tooling updates.
- A new allowlisted WIP registry maintains visibility while preventing noisy warnings during active large-scale refactors.
- `tools/project-health-check.sh` now integrates `tools/quality/worktree_audit.py` to validate the working tree against documented change sets.

## Documented Change Sets
1. **docs_migration_oct2025** — Legacy markdown files replaced by structured entries under `docs/reports/**`.
2. **short_selling_fix_oct2025** — In-progress fixes for J-Quants short-selling normalization and dataset rebuild scripts.
3. **tooling_updates_oct2025** — Codex CLI workflow improvements and CI/link-check refresh.

## Action Items (TODO)
- [ ] Complete documentation migration by reviewing new files under `docs/reports/**` and deleting superseded root-level markdown.
- [ ] Finalize short-selling normalization changes (`src/gogooku3/components/jquants_async_fetcher.py`, dataset scripts) and commit after regression tests.
- [ ] Run `make dataset-bg` post-fix to regenerate the ML dataset and confirm short-selling coverage.
- [ ] Review Codex/CI tooling updates and align team workflows before merging to `main`.

## Implementation Notes
- Allowlist stored at `configs/quality/worktree_allowlist.json` (glob patterns per change set).
- Audit tool: `python tools/quality/worktree_audit.py --config configs/quality/worktree_allowlist.json --pretty`.
- Health check surfaces documented WIP as recommendations instead of warnings, while unexpected paths still escalate.
- Update the allowlist once the WIP batches land or when new refactors begin.
