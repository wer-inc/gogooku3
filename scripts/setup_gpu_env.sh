#!/usr/bin/env bash
#
# GPU Environment Setup Script for ATFT-GAT-FAN
# Purpose: Setup GPU environment for both dataset generation and training
# Usage: bash scripts/setup_gpu_env.sh [--dry-run]
#
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOG_DIR="_logs"
LOG_FILE="${LOG_DIR}/setup_gpu_env.log"
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
    esac
done

# Setup logging
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# Helper functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "${GREEN}▶${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

run_cmd() {
    local desc="$1"
    shift
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $desc: $*"
        return 0
    fi
    echo "Running: $*" >> "$LOG_FILE"
    "$@"
}

# Start
print_header "GPU Environment Setup for ATFT-GAT-FAN"
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
if [ "$DRY_RUN" = true ]; then
    print_warn "DRY-RUN MODE: No changes will be made"
fi
echo ""

# Step 1: Environment Check
print_header "Step 1/7: Environment Check"

print_step "Checking Python version..."
if python3 --version | grep -q "Python 3\.1[0-9]"; then
    print_success "Python $(python3 --version | cut -d' ' -f2) detected"
else
    print_error "Python 3.10+ required"
    exit 1
fi

print_step "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)
    print_success "GPU detected: $GPU_INFO"
else
    print_error "nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

print_step "Checking CUDA compiler..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    print_success "CUDA version: $CUDA_VERSION"
else
    print_warn "nvcc not found (CUDA toolkit may not be installed)"
fi

print_step "Checking .env file..."
if [ -f .env ]; then
    if grep -q "JQUANTS_AUTH_EMAIL" .env && grep -q "JQUANTS_AUTH_PASSWORD" .env; then
        print_success ".env file with JQuants credentials found"
    else
        print_warn ".env exists but missing JQuants credentials"
    fi
else
    print_warn ".env file not found (needed for dataset generation)"
fi

echo ""

# Step 2: Clean conflicting packages
print_header "Step 2/7: Remove Conflicting RAPIDS Packages"

print_step "Checking for RAPIDS packages..."
RAPIDS_PKGS=$(pip list 2>/dev/null | grep -E "cudf|cugraph|rmm" | awk '{print $1}' || true)

if [ -n "$RAPIDS_PKGS" ]; then
    print_warn "Found conflicting RAPIDS packages:"
    echo "$RAPIDS_PKGS" | sed 's/^/  - /'

    if [ "$DRY_RUN" = false ]; then
        print_step "Uninstalling RAPIDS packages..."
        for pkg in $RAPIDS_PKGS; do
            pip uninstall -y "$pkg" &>> "$LOG_FILE" || true
        done
        print_success "RAPIDS packages removed"
    fi
else
    print_success "No conflicting RAPIDS packages found"
fi

echo ""

# Step 3: Install/Update dependencies
print_header "Step 3/7: Install GPU Dependencies"

print_step "Installing CuPy for CUDA 12.x..."
if python3 -c "import cupy; print(cupy.__version__)" &> /dev/null; then
    CUPY_VERSION=$(python3 -c "import cupy; print(cupy.__version__)")
    print_success "CuPy $CUPY_VERSION already installed"
else
    run_cmd "Install CuPy" pip install cupy-cuda12x --quiet
    print_success "CuPy installed"
fi

print_step "Upgrading typing_extensions..."
run_cmd "Upgrade typing_extensions" pip install --upgrade typing_extensions --quiet
print_success "typing_extensions updated"

print_step "Checking PyTorch CUDA support..."
if python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
    print_success "PyTorch $TORCH_VERSION with CUDA $TORCH_CUDA"
else
    print_error "PyTorch CUDA not available"
    exit 1
fi

echo ""

# Step 4: Fix graph_builder_gpu.py
print_header "Step 4/7: Fix graph_builder_gpu.py Import"

GRAPH_BUILDER_GPU="src/data/utils/graph_builder_gpu.py"

print_step "Checking $GRAPH_BUILDER_GPU..."
if [ -f "$GRAPH_BUILDER_GPU" ]; then
    # Check if already fixed
    if grep -q "# Optional imports (not critical for functionality)" "$GRAPH_BUILDER_GPU"; then
        print_success "graph_builder_gpu.py already fixed"
    else
        print_warn "graph_builder_gpu.py needs fixing"

        if [ "$DRY_RUN" = false ]; then
            # Backup
            cp "$GRAPH_BUILDER_GPU" "${GRAPH_BUILDER_GPU}.backup"

            # Apply fix using sed
            sed -i '/^# GPU libraries/,/^$/c\
# GPU libraries\
try:\
    import cupy as cp\
    GPU_AVAILABLE = True\
except ImportError:\
    GPU_AVAILABLE = False\
    cp = np  # Fallback to NumPy if CuPy not available\
\
# Optional imports (not critical for functionality)\
try:\
    import cudf\
    from cupyx.scipy import stats as cu_stats\
except ImportError:\
    pass  # cudf/cupyx are optional\
' "$GRAPH_BUILDER_GPU"

            print_success "graph_builder_gpu.py fixed"
        fi
    fi
else
    print_error "$GRAPH_BUILDER_GPU not found"
    exit 1
fi

echo ""

# Step 5: Fix Pipeline cuGraph Dependencies
print_header "Step 5/7: Fix Pipeline cuGraph Dependencies"

RUN_FULL_DATASET="scripts/pipelines/run_full_dataset.py"
FULL_DATASET_PY="src/pipeline/full_dataset.py"

print_step "Fixing $RUN_FULL_DATASET preflight check..."
if [ -f "$RUN_FULL_DATASET" ]; then
    # Check if already fixed
    if grep -q "# cuGraph は不要（graph_builder_gpu.py が CuPy のみで動作）" "$RUN_FULL_DATASET"; then
        print_success "run_full_dataset.py preflight check already fixed"
    else
        print_warn "run_full_dataset.py needs preflight check fix"

        if [ "$DRY_RUN" = false ]; then
            # Backup
            cp "$RUN_FULL_DATASET" "${RUN_FULL_DATASET}.backup"

            # Fix preflight check: remove 'import cugraph' line
            sed -i '/def _check_gpu_graph_support/,/return False/{
                s/import cugraph  # type: ignore/# cuGraph は不要（graph_builder_gpu.py が CuPy のみで動作）/
                s/"GPU graph dependencies detected (cuGraph %s, CuPy/"GPU graph dependencies detected (CuPy/
                s/getattr(cugraph, "__version__", "?"),//
                s/"GPU graph dependencies unavailable (cuGraph\/CuPy)/"GPU graph dependencies unavailable (CuPy)/
            }' "$RUN_FULL_DATASET"

            print_success "run_full_dataset.py preflight check fixed"
        fi
    fi
else
    print_error "$RUN_FULL_DATASET not found"
    exit 1
fi

print_step "Fixing $FULL_DATASET_PY graph builder selection..."
if [ -f "$FULL_DATASET_PY" ]; then
    # Check if already fixed
    if grep -q "# CuPy のみで GPU 動作可能" "$FULL_DATASET_PY"; then
        print_success "full_dataset.py graph selection already fixed"
    else
        print_warn "full_dataset.py needs graph selection fix"

        if [ "$DRY_RUN" = false ]; then
            # Backup
            cp "$FULL_DATASET_PY" "${FULL_DATASET_PY}.backup"

            # Fix graph builder selection: remove 'import cugraph' line
            sed -i '/use_gpu_graph = False/,/logger.info("📊 Using CPU graph computation/{
                s/import cugraph/# CuPy のみで GPU 動作可能/
                s/"✅ Using GPU-accelerated graph computation (cuGraph detected)"/"✅ Using GPU-accelerated graph computation (CuPy detected)"/
                s/"📊 Using CPU graph computation (cuGraph not available)"/"📊 Using CPU graph computation (CuPy not available)"/
            }' "$FULL_DATASET_PY"

            print_success "full_dataset.py graph selection fixed"
        fi
    fi
else
    print_error "$FULL_DATASET_PY not found"
    exit 1
fi

echo ""

# Step 6: Run tests
print_header "Step 6/7: Verification Tests"

print_step "Testing PyTorch CUDA..."
if python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.device_count() > 0, 'No GPU devices'
print(f'✓ PyTorch CUDA OK: {torch.cuda.get_device_name(0)}')
" 2>> "$LOG_FILE"; then
    print_success "PyTorch CUDA test passed"
else
    print_error "PyTorch CUDA test failed"
    exit 1
fi

print_step "Testing CuPy GPU..."
if python3 -c "
import cupy as cp
assert cp.cuda.runtime.getDeviceCount() > 0, 'No GPU devices'
print(f'✓ CuPy OK: {cp.cuda.runtime.getDeviceCount()} GPU(s)')
" 2>> "$LOG_FILE"; then
    print_success "CuPy GPU test passed"
else
    print_error "CuPy GPU test failed"
    exit 1
fi

print_step "Testing graph_builder_gpu..."
if python3 -c "
from src.data.utils.graph_builder_gpu import FinancialGraphBuilder
import numpy as np
import pandas as pd

# Quick test
codes = ['A', 'B', 'C']
dates = pd.date_range('2024-01-01', periods=60, freq='D')
data = []
for code in codes:
    for date in dates:
        data.append({'code': code, 'date': date, 'return_1d': np.random.randn() * 0.01})
df = pd.DataFrame(data)

builder = FinancialGraphBuilder(verbose=False)
result = builder.build_graph(df, codes, '2024-02-29')

assert result['n_nodes'] > 0, 'No nodes'
assert 'peer_features' in result, 'No peer_features'
assert len(result['peer_features']) > 0, 'Empty peer_features'

print('✓ graph_builder_gpu OK')
" 2>> "$LOG_FILE"; then
    print_success "graph_builder_gpu test passed"
else
    print_error "graph_builder_gpu test failed"
    exit 1
fi

print_step "Testing peer features calculation..."
if python3 -c "
from src.data.utils.graph_builder_gpu import FinancialGraphBuilder
import numpy as np
import pandas as pd

np.random.seed(42)
codes = ['A', 'B', 'C', 'D', 'E']
dates = pd.date_range('2024-01-01', periods=60, freq='D')
data = []
for code in codes:
    for date in dates:
        data.append({'code': code, 'date': date, 'return_1d': np.random.randn() * 0.01})
df = pd.DataFrame(data)

builder = FinancialGraphBuilder(
    correlation_window=30,
    min_observations=20,
    correlation_threshold=0.3,
    verbose=False
)
result = builder.build_graph(df, codes, '2024-02-29')

peer_feats = result['peer_features']
assert len(peer_feats) > 0, 'No peer features'

# Check feature keys
sample = list(peer_feats.values())[0]
required_keys = {'peer_mean_return', 'peer_var_return', 'peer_count', 'peer_correlation_mean'}
assert required_keys.issubset(sample.keys()), f'Missing keys: {required_keys - sample.keys()}'

print('✓ Peer features calculation OK')
" 2>> "$LOG_FILE"; then
    print_success "Peer features test passed"
else
    print_error "Peer features test failed"
    exit 1
fi

echo ""

# Step 7: Summary
print_header "Step 7/7: Setup Summary"

echo ""
print_success "GPU Environment Setup Complete!"
echo ""
echo "System Configuration:"
echo "  • GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  • Python: $(python3 --version | cut -d' ' -f2)"
echo "  • PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  • CuPy: $(python3 -c 'import cupy; print(cupy.__version__)')"
echo "  • CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo ""
echo "Next Steps:"
echo "  1. Dataset Generation (5 years, GPU-accelerated):"
echo "     make dataset-bg START=2020-10-11 END=2025-10-10"
echo ""
echo "  2. Training (with GPU):"
echo "     make train-optimized"
echo ""
echo "  3. Quick test:"
echo "     make smoke"
echo ""
echo "Log saved to: $LOG_FILE"
echo "Completed at: $(date)"
