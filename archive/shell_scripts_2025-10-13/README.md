# Archived Shell Scripts

**Date**: 2025-10-13
**Reason**: Project organization cleanup

## Why Archived

These scripts were archived because they have been superseded by:

1. **Makefile targets** (preferred entry point)
2. **Python scripts** in scripts/ directory
3. **Better organized shell scripts** in scripts/ subdirectories

## Migration Guide

### Training Commands

| Old Command | New Command |
|-------------|-------------|
| `./train_optimized.sh` | `make train-optimized` |
| `./train_improved.sh` | `make train-improved` |
| `./train_optimized_rankic.sh` | `make train-rankic-boost` |
| `./run_stable_training.sh` | `make train-stable` |
| `./run_production_training.sh` | `make train-stable` |
| `./train_with_graph.sh` | Use `--adv-graph-train` flag |

### Testing Commands

| Old Command | New Command |
|-------------|-------------|
| `./smoke_test.sh` | `python scripts/smoke_test.py --max-epochs 1` |
| `./test_student_t.sh` | Experimental, not needed |

### Monitoring Commands

| Old Command | New Command |
|-------------|-------------|
| `./monitor_training.sh` | `scripts/monitoring/monitor_training.sh` |

### Data Generation

| Old Command | New Command |
|-------------|-------------|
| `./generate_sector_dataset.sh` | `scripts/data/generate_sector_dataset.sh` |

### Maintenance

| Old Command | New Command |
|-------------|-------------|
| `./organize_outputs.sh` | `scripts/maintenance/organize_outputs.sh` |

## What's Better Now?

### Before Cleanup
- ❌ 19 .sh files in root directory
- ❌ Unclear which scripts are current
- ❌ Redundant with Makefile targets
- ❌ Difficult to find the right script

### After Cleanup
- ✅ Clean root directory
- ✅ Organized scripts/ structure
- ✅ Makefile as single entry point
- ✅ Easy to find scripts by purpose
- ✅ Clear separation: data/training/monitoring/maintenance

## If You Need These Scripts

If you absolutely need any of these scripts back:

1. **Check Makefile first**: `make help`
2. **Check scripts/ directory**: Organized by purpose
3. **Copy from this archive** if no alternative exists

Example:
```bash
# Copy from archive if needed
cp archive/shell_scripts_2025-10-13/train_optimized.sh ./
```

## Files in This Archive

### Training Wrappers (10 files)
- train_optimized.sh
- train_improved.sh
- train_optimized_final.sh
- train_optimized_rankic.sh
- train_with_graph.sh
- run_optimized.sh
- run_production_training.sh
- run_stable_simple.sh
- run_stable_training.sh
- run_single_process.sh

### Testing Scripts (3 files)
- smoke_test.sh
- test_student_t.sh
- test_student_t_direct.sh

### Debugging Scripts (1 file)
- fix_zero_loss.sh

**Total**: 14 archived files

## Related Documentation

- **SHELL_SCRIPTS_CLEANUP_PLAN.md** - Full cleanup plan and rationale
- **Makefile** - Current entry point for all commands
- **CLAUDE.md** - Project documentation
- **make help** - List all available commands
