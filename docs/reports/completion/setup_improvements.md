# Make Setup Improvements - Complete Documentation

**Date**: 2025-10-13
**Status**: ✅ Complete
**Author**: Claude (with user approval)

## Overview

This document describes the comprehensive improvements made to `make setup` to achieve a "perfect" setup experience that is idempotent, fail-fast, GPU-required, and uses modern Python packaging standards.

## Problems Solved

### 1. Redundant Dependency Management
**Before**: Two sources of truth for dependencies
- `requirements.txt` (79 lines)
- `pyproject.toml` dependencies section
- Risk of version conflicts and maintenance overhead

**After**: Single source of truth
- All dependencies in `pyproject.toml` (PEP 621 standard)
- `requirements.txt` → deprecation notice file
- `requirements.txt.deprecated` → historical reference

### 2. Weak Error Handling
**Before**: Steps failed but continued
```makefile
./venv/bin/pip install -r requirements.txt
# If this fails, setup continues anyway
```

**After**: Fail-fast on all steps
```makefile
./venv/bin/pip install -e . || { echo "❌ Dependency installation failed"; exit 1; }
```

### 3. Lack of Idempotency
**Before**: Unsafe to re-run
- Recreated venv every time (destroying existing environment)
- No protection for existing configurations

**After**: Safe re-runs
```makefile
@if [ -d venv ]; then
    echo "⚠️  venv already exists - skipping creation";
    echo "💡 To rebuild: rm -rf venv && make setup";
else
    python3 -m venv venv || { echo "❌ venv creation failed"; exit 1; };
fi
```

### 4. Weak Verification
**Before**: Soft warnings
```makefile
@./venv/bin/python -c "import torch; print(...)" || echo "⚠️ PyTorch check failed"
# Continues even if PyTorch or CUDA are broken
```

**After**: Strong assertions
```makefile
@./venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(...)" || { echo "❌ PyTorch/CUDA verification failed"; exit 1; }
```

### 5. Optional GPU (CPU Fallback)
**Before**: GPU was "nice to have"
```makefile
else
    echo "⚠️  No GPU detected - CPU-only mode";
    echo "⚠️  Dataset generation will be 10-100x slower without GPU";
fi
# Continues without GPU
```

**After**: GPU is required
```makefile
@if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "❌ GPU NOT detected - nvidia-smi not found";
    echo "❌ This project requires GPU for dataset generation and training";
    exit 1;
fi
```

### 6. Missing Dev Dependencies
**Before**: Only production dependencies installed
```makefile
./venv/bin/pip install -e .
# pytest, ruff, mypy, etc. NOT installed
```

**After**: Dev dependencies included
```makefile
./venv/bin/pip install -e .
./venv/bin/pip install -e ".[dev]"  # pytest, ruff, mypy, black, etc.
```

## Implementation Details

### New Setup Flow (6 Steps)

#### Step 1/6: Idempotent venv Creation
- Checks if venv exists before creating
- Upgrades pip, setuptools, wheel
- Fail-fast on errors
- Safe to re-run

#### Step 2/6: Dependencies from pyproject.toml
- Single source: `pyproject.toml`
- Installs production dependencies: `pip install -e .`
- Installs dev dependencies: `pip install -e ".[dev]"`
- Fail-fast on errors

#### Step 3/6: Pre-commit Hooks
- Installs pre-commit hooks
- Installs commit-msg hooks
- Fail-fast on errors (was optional before)

#### Step 4/6: .env Configuration
- Creates from template if not exists
- Preserves existing .env
- Idempotent behavior

#### Step 5/6: GPU Environment (REQUIRED)
- Checks nvidia-smi availability
- **Exits immediately if no GPU**
- Installs CuPy for CUDA 12.x
- Installs RAPIDS (cuDF, cuGraph, RMM 24.12.0)
- Removes numba-cuda conflicts
- Verifies all GPU packages with imports
- Fail-fast on any error

#### Step 6/6: Final Verification with Assertions
- Verifies gogooku3 package
- **Asserts** CUDA is available (not just warning)
- Verifies Polars
- Verifies GPU stack (CuPy, cuDF, cuGraph, RMM)
- Shows system information (GPU name, CUDA version, Python version)
- Fail-fast on any error

### Files Modified

#### 1. Makefile (lines 37-134)
**Changes**:
- Complete rewrite of `setup` target
- Reduced from 7 steps to 6 steps (removed requirements.txt step)
- Added idempotency checks
- Made all error handling fail-fast
- Made GPU required (not optional)
- Added comprehensive verification with assertions
- Improved "Next steps" guidance

**Key improvements**:
```makefile
# Before (line 48-49):
./venv/bin/pip install -r requirements.txt
@echo "✅ Python dependencies installed"

# After (line 53-57):
@echo "   📝 Installing from pyproject.toml (production + dev)"
@./venv/bin/pip install -e . || { echo "❌ Dependency installation failed"; exit 1; }
@./venv/bin/pip install -e ".[dev]" || { echo "❌ Dev tools installation failed"; exit 1; }
```

#### 2. requirements.txt
**Status**: Replaced with deprecation notice

**Content**:
- Clear deprecation warning
- Migration guide (old → new)
- Explanation of why pyproject.toml is better
- GPU dependencies installation guide
- References to documentation

#### 3. requirements.txt.deprecated
**Status**: Created (backup of original)

**Content**:
- Full original requirements.txt (79 lines, 3.7KB)
- Historical reference
- Available if needed for comparison

## Verification Tests

### Test 1: Fresh Setup
```bash
# Clean environment
rm -rf venv

# Run setup
make setup

# Expected:
# ✅ All 6 steps complete
# ✅ GPU detected and configured
# ✅ All assertions pass
# ✅ No warnings or errors
```

### Test 2: Idempotent Re-run
```bash
# With existing venv
make setup

# Expected:
# ⚠️  venv already exists - skipping creation
# ✅ Dependencies reinstalled (safe)
# ✅ All other steps complete
# ✅ No errors
```

### Test 3: No GPU Failure
```bash
# Simulate no GPU
alias nvidia-smi="false"
make setup

# Expected:
# ❌ GPU NOT detected - nvidia-smi not found
# ❌ This project requires GPU for dataset generation and training
# Exit 1 (setup stops immediately)
```

### Test 4: Broken Dependency
```bash
# Edit pyproject.toml with invalid package
make setup

# Expected:
# ❌ Dependency installation failed
# Exit 1 (setup stops immediately)
```

### Test 5: CUDA Not Available
```bash
# With GPU but CUDA broken
make setup

# Expected:
# (GPU packages install successfully)
# ❌ PyTorch/CUDA verification failed
# Exit 1 (setup stops at final verification)
```

## Benefits Summary

### 1. **Idempotency** ✅
- Safe to run multiple times
- Preserves existing environment
- Clear guidance on full rebuild

### 2. **Fail-Fast** ✅
- Every step has proper error handling
- Exit immediately on failures
- No silent errors or warnings

### 3. **GPU Required** ✅
- No CPU-only fallback
- Clear error messages
- Prevents accidental slow execution

### 4. **Single Source of Truth** ✅
- pyproject.toml only
- No duplicate dependency lists
- Modern Python standard (PEP 621)

### 5. **Strong Verification** ✅
- Assertions instead of warnings
- Comprehensive package checks
- System information display

### 6. **Developer Experience** ✅
- Clear step-by-step progress
- Helpful error messages
- Detailed "Next steps" guidance
- Better documentation

## Migration for Existing Users

### For New Users
```bash
# Just run setup
make setup

# Follow the "Next steps" guidance
```

### For Existing Users (with old venv)
```bash
# Option 1: Keep existing venv (safe)
make setup
# Will skip venv creation, reinstall dependencies

# Option 2: Fresh start (recommended)
rm -rf venv
make setup
```

### If You Used requirements.txt Directly
```bash
# Old way (DEPRECATED):
pip install -r requirements.txt

# New way:
pip install -e .          # Production deps
pip install -e ".[dev]"   # Dev deps

# Or just use:
make setup
```

## References

- **PEP 621**: Python Project Metadata (pyproject.toml standard)
- **Makefile**: Lines 37-134 (setup target)
- **pyproject.toml**: Lines 26-113 (dependencies)
- **requirements.txt**: Deprecation notice
- **requirements.txt.deprecated**: Historical backup

## Approval Trail

**User Request**: "make setupを完璧な状態にしたい"

**Plan Presented**: 2025-10-13
- Remove requirements.txt redundancy
- Ensure idempotency
- Strengthen error handling
- Require GPU (no CPU-only mode)
- Strong final verification

**User Approval**: Approved ✅

**Implementation**: 2025-10-13 ✅

## Status: Complete ✅

All improvements have been implemented and documented. The `make setup` target is now production-ready with:
- Idempotency
- Fail-fast error handling
- GPU requirement
- Single source of truth (pyproject.toml)
- Strong verification
- Excellent developer experience
