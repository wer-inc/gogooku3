# Markdown Files Cleanup Plan

**Date**: 2025-10-13
**Status**: 📋 Plan
**Issue**: 35 .md files scattered in root directory

## Current State

### Root Directory .md Files (35 files - PROBLEM)

```
/root/gogooku3/
├── AGENTS.md
├── ATFT‑GAT‑FAN_IMPL.md
├── CACHE_FIX_DOCUMENTATION.md
├── CHANGELOG.md
├── CHANGES_SUMMARY.md
├── CLAUDE.md
├── CLEAN_UP.md
├── CODEX_MCP.md
├── DATASET_GENERATION_GUIDE.md
├── DOCS_REORGANIZATION_COMPLETE.md
├── DOCS_REORGANIZATION_STATUS.md
├── EFFICIENCY_REPORT.md
├── FEATURE_DEFAULTS_UPDATE.md
├── FUTURES_INTEGRATION_COMPLETE.md
├── GPU_ETL_USAGE.md
├── GPU_ML_PIPELINE.md
├── GPU_TRAINING.md
├── MIGRATION.md
├── NK225_OPTION_INTEGRATION_STATUS.md
├── OPTIMIZATION_REPORT_20251001.md
├── QUICK_START.md
├── README.md
├── REFACTORING.md
├── ROOT_CAUSE_FIX_COMPLETE.md
├── SECTOR_SHORT_SELLING_INTEGRATION_COMPLETE.md
├── SETUP_IMPROVEMENTS.md
├── SHELL_CLEANUP_COMPLETE.md
├── SHELL_SCRIPTS_CLEANUP_PLAN.md
├── TEST_CLEANUP_COMPLETE.md
├── TEST_FILES_CLEANUP_PLAN.md
├── TODO.md
├── TRAINING_IMPROVEMENTS.md
├── github_issues.md
├── manual.md
└── memories.md
```

### Properly Organized (OK)
```
docs/
├── architecture/
├── deepresearch/
├── development/
├── feedback/
├── fixes/
├── governance/
├── guides/
├── ja/
├── ml/
├── operations/
├── releases/
└── reports/
(78 .md files properly organized)
```

## Analysis & Classification

### Category 1: **KEEP IN ROOT** (Essential Project Files - 4 files)

These are the core files that should remain in root:

- **README.md** - Project overview (standard location)
- **CLAUDE.md** - Claude Code project instructions (standard location)
- **CHANGELOG.md** - Version history (standard location)
- **TODO.md** - Current tasks (frequently accessed)

**Reason**: Standard project structure, frequently accessed

### Category 2: **Guides & Tutorials** → `docs/guides/` (7 files)

- QUICK_START.md → docs/guides/quick_start.md
- DATASET_GENERATION_GUIDE.md → docs/guides/dataset_generation.md
- GPU_TRAINING.md → docs/guides/gpu_training.md
- GPU_ETL_USAGE.md → docs/guides/gpu_etl_usage.md
- GPU_ML_PIPELINE.md → docs/guides/gpu_ml_pipeline.md
- manual.md → docs/guides/manual.md
- CODEX_MCP.md → docs/guides/codex_mcp.md

**Reason**: User-facing documentation and tutorials

### Category 3: **Completion Reports** → `docs/reports/completion/` (13 files)

Recent completion/status reports (including today's session):

- SETUP_IMPROVEMENTS.md → docs/reports/completion/setup_improvements.md
- SHELL_CLEANUP_COMPLETE.md → docs/reports/completion/shell_cleanup_complete.md
- SHELL_SCRIPTS_CLEANUP_PLAN.md → docs/reports/completion/shell_scripts_cleanup_plan.md
- TEST_CLEANUP_COMPLETE.md → docs/reports/completion/test_cleanup_complete.md
- TEST_FILES_CLEANUP_PLAN.md → docs/reports/completion/test_files_cleanup_plan.md
- CHANGES_SUMMARY.md → docs/reports/completion/changes_summary.md
- DOCS_REORGANIZATION_COMPLETE.md → docs/reports/completion/docs_reorganization_complete.md
- DOCS_REORGANIZATION_STATUS.md → docs/reports/completion/docs_reorganization_status.md
- ROOT_CAUSE_FIX_COMPLETE.md → docs/reports/completion/root_cause_fix_complete.md
- CACHE_FIX_DOCUMENTATION.md → docs/reports/completion/cache_fix_documentation.md
- FUTURES_INTEGRATION_COMPLETE.md → docs/reports/completion/futures_integration_complete.md
- SECTOR_SHORT_SELLING_INTEGRATION_COMPLETE.md → docs/reports/completion/sector_short_selling_integration_complete.md
- NK225_OPTION_INTEGRATION_STATUS.md → docs/reports/completion/nk225_option_integration_status.md

**Reason**: Historical completion reports, useful for reference

### Category 4: **Feature/Update Reports** → `docs/reports/features/` (1 file)

- FEATURE_DEFAULTS_UPDATE.md → docs/reports/features/feature_defaults_update.md

**Reason**: Feature-specific documentation

### Category 5: **Architecture & Implementation** → `docs/architecture/` (3 files)

- ATFT‑GAT‑FAN_IMPL.md → docs/architecture/atft_gat_fan_implementation.md
- MIGRATION.md → docs/architecture/migration.md
- REFACTORING.md → docs/architecture/refactoring.md

**Reason**: Technical architecture documentation

### Category 6: **Analysis & Optimization Reports** → `docs/reports/analysis/` (3 files)

- OPTIMIZATION_REPORT_20251001.md → docs/reports/analysis/optimization_report_20251001.md
- EFFICIENCY_REPORT.md → docs/reports/analysis/efficiency_report.md
- TRAINING_IMPROVEMENTS.md → docs/reports/analysis/training_improvements.md

**Reason**: Performance analysis and optimization reports

### Category 7: **Development/Meta** → `docs/development/` (4 files)

- AGENTS.md → docs/development/agents.md
- CLEAN_UP.md → docs/development/clean_up.md
- github_issues.md → docs/development/github_issues.md
- memories.md → docs/development/memories.md

**Reason**: Development notes and meta-information

## Proposed Directory Structure

```
/root/gogooku3/
├── README.md                    ✅ KEEP
├── CLAUDE.md                    ✅ KEEP
├── CHANGELOG.md                 ✅ KEEP
├── TODO.md                      ✅ KEEP
└── docs/
    ├── guides/                  📚 (7 new files)
    │   ├── quick_start.md
    │   ├── dataset_generation.md
    │   ├── gpu_training.md
    │   ├── gpu_etl_usage.md
    │   ├── gpu_ml_pipeline.md
    │   ├── manual.md
    │   └── codex_mcp.md
    │
    ├── architecture/            🏗️ (3 new files)
    │   ├── atft_gat_fan_implementation.md
    │   ├── migration.md
    │   └── refactoring.md
    │
    ├── reports/
    │   ├── completion/          📋 (13 new files)
    │   │   ├── setup_improvements.md
    │   │   ├── shell_cleanup_complete.md
    │   │   ├── test_cleanup_complete.md
    │   │   └── ...
    │   │
    │   ├── analysis/            📊 (3 new files)
    │   │   ├── optimization_report_20251001.md
    │   │   ├── efficiency_report.md
    │   │   └── training_improvements.md
    │   │
    │   └── features/            ✨ (1 new file)
    │       └── feature_defaults_update.md
    │
    └── development/             🔧 (4 new files)
        ├── agents.md
        ├── clean_up.md
        ├── github_issues.md
        └── memories.md
```

## Cleanup Actions

### Phase 1: Create necessary subdirectories
```bash
mkdir -p docs/guides
mkdir -p docs/reports/completion
mkdir -p docs/reports/analysis
mkdir -p docs/reports/features
# docs/architecture/ and docs/development/ already exist
```

### Phase 2: Move guides to docs/guides/
```bash
mv QUICK_START.md docs/guides/quick_start.md
mv DATASET_GENERATION_GUIDE.md docs/guides/dataset_generation.md
mv GPU_TRAINING.md docs/guides/gpu_training.md
mv GPU_ETL_USAGE.md docs/guides/gpu_etl_usage.md
mv GPU_ML_PIPELINE.md docs/guides/gpu_ml_pipeline.md
mv manual.md docs/guides/manual.md
mv CODEX_MCP.md docs/guides/codex_mcp.md
```

### Phase 3: Move completion reports to docs/reports/completion/
```bash
mv SETUP_IMPROVEMENTS.md docs/reports/completion/setup_improvements.md
mv SHELL_CLEANUP_COMPLETE.md docs/reports/completion/shell_cleanup_complete.md
mv SHELL_SCRIPTS_CLEANUP_PLAN.md docs/reports/completion/shell_scripts_cleanup_plan.md
mv TEST_CLEANUP_COMPLETE.md docs/reports/completion/test_cleanup_complete.md
mv TEST_FILES_CLEANUP_PLAN.md docs/reports/completion/test_files_cleanup_plan.md
mv CHANGES_SUMMARY.md docs/reports/completion/changes_summary.md
mv DOCS_REORGANIZATION_COMPLETE.md docs/reports/completion/docs_reorganization_complete.md
mv DOCS_REORGANIZATION_STATUS.md docs/reports/completion/docs_reorganization_status.md
mv ROOT_CAUSE_FIX_COMPLETE.md docs/reports/completion/root_cause_fix_complete.md
mv CACHE_FIX_DOCUMENTATION.md docs/reports/completion/cache_fix_documentation.md
mv FUTURES_INTEGRATION_COMPLETE.md docs/reports/completion/futures_integration_complete.md
mv SECTOR_SHORT_SELLING_INTEGRATION_COMPLETE.md docs/reports/completion/sector_short_selling_integration_complete.md
mv NK225_OPTION_INTEGRATION_STATUS.md docs/reports/completion/nk225_option_integration_status.md
```

### Phase 4: Move feature reports to docs/reports/features/
```bash
mv FEATURE_DEFAULTS_UPDATE.md docs/reports/features/feature_defaults_update.md
```

### Phase 5: Move architecture docs to docs/architecture/
```bash
mv ATFT‑GAT‑FAN_IMPL.md docs/architecture/atft_gat_fan_implementation.md
mv MIGRATION.md docs/architecture/migration.md
mv REFACTORING.md docs/architecture/refactoring.md
```

### Phase 6: Move analysis reports to docs/reports/analysis/
```bash
mv OPTIMIZATION_REPORT_20251001.md docs/reports/analysis/optimization_report_20251001.md
mv EFFICIENCY_REPORT.md docs/reports/analysis/efficiency_report.md
mv TRAINING_IMPROVEMENTS.md docs/reports/analysis/training_improvements.md
```

### Phase 7: Move development docs to docs/development/
```bash
mv AGENTS.md docs/development/agents.md
mv CLEAN_UP.md docs/development/clean_up.md
mv github_issues.md docs/development/github_issues.md
mv memories.md docs/development/memories.md
```

### Phase 8: Create documentation index
```bash
# Create docs/reports/completion/README.md
# Create docs/guides/README.md
# Update docs/INDEX.md with new structure
```

## Benefits

### Before (Current State)
- ❌ 35 .md files in root directory
- ❌ Unclear document organization
- ❌ Difficult to find specific documentation
- ❌ Root directory cluttered

### After (Proposed State)
- ✅ Only 4 essential .md files in root
- ✅ Clear categorization (guides/reports/architecture/development)
- ✅ Easy to find documentation by purpose
- ✅ Clean root directory
- ✅ Consistent with existing docs/ structure

## File Count Summary

| Category | Files | Destination |
|----------|-------|-------------|
| **Keep in root** | 4 | Root (no move) |
| **Guides** | 7 | `docs/guides/` |
| **Completion reports** | 13 | `docs/reports/completion/` |
| **Feature reports** | 1 | `docs/reports/features/` |
| **Architecture** | 3 | `docs/architecture/` |
| **Analysis reports** | 3 | `docs/reports/analysis/` |
| **Development** | 4 | `docs/development/` |
| **Total** | **35** | **31 moved, 4 kept** |

## Documentation Index

After cleanup, create/update:

### `docs/INDEX.md`
Update to include:
- Link to guides/
- Link to reports/completion/
- Link to reports/analysis/
- Link to reports/features/
- Link to architecture/
- Link to development/

### `docs/guides/README.md` (new)
List all guides with brief descriptions

### `docs/reports/completion/README.md` (new)
List all completion reports chronologically

## Verification

After cleanup, verify:

```bash
# 1. Check root directory is clean
ls -1 *.md
# Expected: README.md, CLAUDE.md, CHANGELOG.md, TODO.md only

# 2. Check docs/ organization
find docs/ -name "*.md" | wc -l
# Expected: 78 + 31 = 109 files

# 3. Check new subdirectories exist
ls docs/guides/
ls docs/reports/completion/
ls docs/reports/analysis/
ls docs/reports/features/
```

## Risks and Mitigation

### Risk 1: Links to moved files may break
**Mitigation**:
- Search for references: `grep -r "SETUP_IMPROVEMENTS.md" .`
- Update CLAUDE.md if it references moved files
- Create redirect notes in root if needed

### Risk 2: Some files may be referenced by CI/CD or scripts
**Mitigation**:
- Check scripts/ for references
- Check .github/ workflows for references
- Update paths in any automation

### Risk 3: Users may have bookmarked old paths
**Mitigation**:
- Update README.md with "Documentation moved to docs/"
- Add note to CLAUDE.md about new structure

## Status: Awaiting Approval

This is a proposed plan. Execute with:
```bash
bash scripts/maintenance/cleanup_markdown_files.sh
```

Or execute manually following Phase 1-8 above.
