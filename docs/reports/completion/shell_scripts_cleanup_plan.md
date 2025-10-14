# Shell Scripts Cleanup Plan

**Date**: 2025-10-13
**Status**: 📋 Plan
**Issue**: 19 .sh files scattered in root directory

## Current State

### Root Directory .sh Files (19 files - PROBLEM)
```
/root/gogooku3/
├── codex-mcp-max.sh
├── codex-mcp.sh
├── fix_zero_loss.sh
├── generate_sector_dataset.sh
├── monitor_training.sh
├── organize_outputs.sh
├── run_optimized.sh
├── run_production_training.sh
├── run_single_process.sh
├── run_stable_simple.sh
├── run_stable_training.sh
├── smoke_test.sh
├── test_student_t.sh
├── test_student_t_direct.sh
├── train_improved.sh
├── train_optimized.sh
├── train_optimized_final.sh
├── train_optimized_rankic.sh
└── train_with_graph.sh
```

### Properly Organized (OK)
```
scripts/
├── generate_full_dataset.sh
├── launch_train_gpu_latest.sh
├── run_dataset_gpu.sh
├── run_improved_training.sh
├── setup_env.sh
├── train_gpu_latest.sh
├── maintenance/
│   ├── cleanup_cache.sh
│   ├── cleanup_datasets.sh
│   ├── cleanup_raw_data.sh
│   └── sync_to_gcs.sh
└── monitoring/
    └── watch_dataset.sh
```

## Analysis

### Files Status Check

#### 1. **Training Wrappers** (Redundant - Makefile targets exist)
- `train_optimized.sh` → `make train-optimized` exists
- `train_improved.sh` → `make train-improved` exists
- `train_optimized_final.sh` → Redundant
- `train_optimized_rankic.sh` → `make train-rankic-boost` exists
- `train_with_graph.sh` → `--adv-graph-train` flag exists
- `run_optimized.sh` → Redundant
- `run_production_training.sh` → `make train-stable` exists
- `run_stable_simple.sh` → Redundant
- `run_stable_training.sh` → `make train-stable` exists
- `run_single_process.sh` → Redundant

**Recommendation**: Archive or delete (Makefile is the single source of truth)

#### 2. **Testing Scripts** (Redundant - Python scripts exist)
- `smoke_test.sh` → `python scripts/smoke_test.py` exists
- `test_student_t.sh` → Experimental, not used
- `test_student_t_direct.sh` → Experimental, not used

**Recommendation**: Archive (Python scripts preferred)

#### 3. **Debugging/Development Tools**
- `fix_zero_loss.sh` → One-time fix script, likely obsolete
- `organize_outputs.sh` → Should be in scripts/maintenance/
- `monitor_training.sh` → Should be in scripts/monitoring/

**Recommendation**: Move to appropriate directories or archive

#### 4. **Data Generation**
- `generate_sector_dataset.sh` → Should be in scripts/data/

**Recommendation**: Move to scripts/data/

#### 5. **External Tools**
- `codex-mcp.sh` → Codex tool launcher (not Claude Code)
- `codex-mcp-max.sh` → Codex tool launcher

**Recommendation**: Move to tools/ directory or archive

## Proposed Directory Structure

```
/root/gogooku3/
├── scripts/
│   ├── data/
│   │   ├── generate_full_dataset.sh
│   │   ├── run_dataset_gpu.sh
│   │   └── generate_sector_dataset.sh (moved)
│   ├── training/
│   │   ├── launch_train_gpu_latest.sh
│   │   ├── train_gpu_latest.sh
│   │   └── run_improved_training.sh
│   ├── maintenance/
│   │   ├── cleanup_cache.sh
│   │   ├── cleanup_datasets.sh
│   │   ├── cleanup_raw_data.sh
│   │   ├── sync_to_gcs.sh
│   │   └── organize_outputs.sh (moved)
│   ├── monitoring/
│   │   ├── watch_dataset.sh
│   │   └── monitor_training.sh (moved)
│   ├── testing/
│   │   └── (Python scripts preferred)
│   └── setup_env.sh
├── tools/ (new)
│   ├── codex-mcp.sh (moved)
│   └── codex-mcp-max.sh (moved)
└── archive/
    └── shell_scripts_2025-10-13/
        ├── train_optimized.sh (archived)
        ├── train_improved.sh (archived)
        ├── smoke_test.sh (archived)
        └── ... (all redundant scripts)
```

## Cleanup Actions

### Phase 1: Move to Proper Locations
```bash
# Create new directories
mkdir -p scripts/data
mkdir -p scripts/training
mkdir -p tools
mkdir -p archive/shell_scripts_2025-10-13

# Move data scripts
mv generate_sector_dataset.sh scripts/data/

# Move monitoring scripts
mv monitor_training.sh scripts/monitoring/

# Move maintenance scripts
mv organize_outputs.sh scripts/maintenance/

# Move external tools
mv codex-mcp.sh codex-mcp-max.sh tools/
```

### Phase 2: Archive Redundant Scripts
```bash
# Archive all redundant training wrappers
mv train_*.sh archive/shell_scripts_2025-10-13/
mv run_*.sh archive/shell_scripts_2025-10-13/
mv smoke_test.sh archive/shell_scripts_2025-10-13/
mv test_student_*.sh archive/shell_scripts_2025-10-13/
mv fix_zero_loss.sh archive/shell_scripts_2025-10-13/
```

### Phase 3: Create Archive README
```bash
# Document archived scripts
cat > archive/shell_scripts_2025-10-13/README.md <<EOF
# Archived Shell Scripts

These scripts were archived on 2025-10-13 as part of project cleanup.

## Why Archived

All these scripts have been superseded by:
1. Makefile targets (preferred entry point)
2. Python scripts in scripts/ directory
3. Better organized shell scripts

## Migration Guide

### Training Commands
- OLD: ./train_optimized.sh → NEW: make train-optimized
- OLD: ./run_stable_training.sh → NEW: make train-stable
- OLD: ./train_improved.sh → NEW: make train-improved

### Testing Commands
- OLD: ./smoke_test.sh → NEW: python scripts/smoke_test.py --max-epochs 1

### Monitoring Commands
- OLD: ./monitor_training.sh → NEW: scripts/monitoring/monitor_training.sh

## If You Need These Scripts

If you need any of these scripts back:
1. Check if equivalent Makefile target exists
2. Check if Python script exists in scripts/
3. Copy from this archive if absolutely necessary

EOF
```

## Benefits

### Before (Current State)
- ❌ 19 .sh files in root directory
- ❌ Unclear which scripts are current
- ❌ Redundant with Makefile targets
- ❌ Difficult to find the right script
- ❌ No clear organization

### After (Proposed State)
- ✅ Clean root directory
- ✅ Organized scripts/ structure
- ✅ Makefile as single entry point
- ✅ Easy to find scripts by purpose
- ✅ Clear separation: data/training/monitoring/maintenance
- ✅ Archived scripts for historical reference

## Verification

After cleanup, verify:

```bash
# 1. Check root directory is clean
ls -la *.sh
# Expected: No .sh files

# 2. Check scripts/ organization
tree scripts/
# Expected: Organized subdirectories

# 3. Check Makefile targets work
make train-stable
make train-optimized
make smoke

# 4. Check archived scripts documented
cat archive/shell_scripts_2025-10-13/README.md
```

## Risks and Mitigation

### Risk 1: Some script might still be in use
**Mitigation**: Archive (don't delete), can restore if needed

### Risk 2: External tools depend on script paths
**Mitigation**: Check for references before moving

### Risk 3: Documentation might reference old paths
**Mitigation**: Update CLAUDE.md and README.md if needed

## Status: Awaiting Approval

This is a proposed plan. Execute with:
```bash
bash scripts/maintenance/cleanup_shell_scripts.sh
```

Or execute manually following Phase 1-3 above.
