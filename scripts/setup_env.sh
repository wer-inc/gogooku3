#!/usr/bin/env bash
#
# Complete Environment Setup Script for ATFT-GAT-FAN
# Purpose: Setup GPU environment, GCS access, and all dependencies
# Usage: bash scripts/setup_env.sh [--dry-run] [--skip-gcs] [--skip-gpu]
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
LOG_FILE="${LOG_DIR}/setup_env.log"
DRY_RUN=false
SKIP_GCS=false
SKIP_GPU=false
GCS_KEY_FILE="secrets/gogooku-b3b34bc07639.json"
GCLOUD_SDK_DIR="google-cloud-sdk"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-gcs)
            SKIP_GCS=true
            shift
            ;;
        --skip-gpu)
            SKIP_GPU=true
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
print_header "Complete Environment Setup for ATFT-GAT-FAN"
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
if [ "$DRY_RUN" = true ]; then
    print_warn "DRY-RUN MODE: No changes will be made"
fi
if [ "$SKIP_GCS" = true ]; then
    print_warn "Skipping GCS setup"
fi
if [ "$SKIP_GPU" = true ]; then
    print_warn "Skipping GPU setup"
fi
echo ""

# Step 1: Environment Check
print_header "Step 1/9: Environment Check"

print_step "Checking Python version..."
if python3 --version | grep -q "Python 3\.1[0-9]"; then
    print_success "Python $(python3 --version | cut -d' ' -f2) detected"
else
    print_error "Python 3.10+ required"
    exit 1
fi

if [ "$SKIP_GPU" = false ]; then
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

# Step 2: GCS Setup
if [ "$SKIP_GCS" = false ]; then
    print_header "Step 2/9: Google Cloud SDK & GCS Setup"

    print_step "Checking for gcloud CLI..."
    if command -v gcloud &> /dev/null; then
        GCLOUD_VERSION=$(gcloud version --format="value(core)" 2>/dev/null || echo "unknown")
        print_success "gcloud CLI already installed (version: $GCLOUD_VERSION)"
    else
        print_warn "gcloud CLI not found, installing..."

        if [ "$DRY_RUN" = false ]; then
            print_step "Downloading Google Cloud SDK..."
            TEMP_DIR=$(mktemp -d)
            cd "$TEMP_DIR"
            curl -sS https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz -o gcloud-sdk.tar.gz

            print_step "Extracting Google Cloud SDK..."
            tar -xzf gcloud-sdk.tar.gz

            print_step "Moving to project directory..."
            cd - > /dev/null
            if [ -d "$GCLOUD_SDK_DIR" ]; then
                print_warn "Removing existing $GCLOUD_SDK_DIR..."
                rm -rf "$GCLOUD_SDK_DIR"
            fi
            mv "$TEMP_DIR/google-cloud-sdk" ./
            rm -rf "$TEMP_DIR"

            print_step "Installing Google Cloud SDK..."
            ./$GCLOUD_SDK_DIR/install.sh --quiet --path-update=false --usage-reporting=false

            print_success "Google Cloud SDK installed"
        fi
    fi

    # Setup PATH for gcloud
    if [ -d "$GCLOUD_SDK_DIR/bin" ]; then
        export PATH="$PATH:$(pwd)/$GCLOUD_SDK_DIR/bin"
        print_success "gcloud added to PATH"
    fi

    print_step "Checking for GCS service account key..."
    if [ -f "$GCS_KEY_FILE" ]; then
        print_success "GCS service account key found: $GCS_KEY_FILE"

        if [ "$DRY_RUN" = false ]; then
            print_step "Authenticating with service account..."
            gcloud auth activate-service-account --key-file="$GCS_KEY_FILE" 2>&1 | grep -v "WARNING" || true

            # Set GOOGLE_APPLICATION_CREDENTIALS
            export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/$GCS_KEY_FILE"

            # Add to .env if not present
            if [ -f .env ]; then
                if ! grep -q "GOOGLE_APPLICATION_CREDENTIALS" .env; then
                    echo "" >> .env
                    echo "# GCS Configuration" >> .env
                    echo "GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/$GCS_KEY_FILE" >> .env
                    print_success "Added GOOGLE_APPLICATION_CREDENTIALS to .env"
                else
                    print_success "GOOGLE_APPLICATION_CREDENTIALS already in .env"
                fi
            fi

            print_success "GCS authentication configured"
        fi
    else
        print_warn "GCS service account key not found at $GCS_KEY_FILE"
        print_warn "GCS features will be disabled"
    fi

    print_step "Verifying gsutil..."
    if command -v gsutil &> /dev/null; then
        print_success "gsutil available"

        if [ "$DRY_RUN" = false ] && [ -f "$GCS_KEY_FILE" ]; then
            print_step "Testing GCS access..."
            if gsutil ls gs://gogooku-ml-data/ &> /dev/null; then
                print_success "GCS bucket access verified"
            else
                print_warn "Cannot access GCS bucket (check permissions)"
            fi
        fi
    else
        print_error "gsutil not available after gcloud installation"
        exit 1
    fi

    echo ""
else
    print_header "Step 2/9: GCS Setup (SKIPPED)"
    echo ""
fi

# Step 3: Clean conflicting packages
if [ "$SKIP_GPU" = false ]; then
    print_header "Step 3/9: Remove Conflicting RAPIDS Packages"

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
else
    print_header "Step 3/9: RAPIDS Packages (SKIPPED)"
    echo ""
fi

# Step 4: Install/Update dependencies
if [ "$SKIP_GPU" = false ]; then
    print_header "Step 4/9: Install GPU Dependencies"

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
else
    print_header "Step 4/9: GPU Dependencies (SKIPPED)"
    echo ""
fi

# Step 5: Install GCS Python dependencies
if [ "$SKIP_GCS" = false ]; then
    print_header "Step 5/9: Install GCS Python Dependencies"

    print_step "Checking google-cloud-storage..."
    if python3 -c "import google.cloud.storage" &> /dev/null; then
        GCS_VERSION=$(python3 -c "from google.cloud import storage; print(storage.__version__)")
        print_success "google-cloud-storage $GCS_VERSION already installed"
    else
        print_step "Installing google-cloud-storage..."
        run_cmd "Install google-cloud-storage" pip install google-cloud-storage --quiet
        print_success "google-cloud-storage installed"
    fi

    echo ""
else
    print_header "Step 5/9: GCS Python Dependencies (SKIPPED)"
    echo ""
fi

# Step 6: Fix graph_builder_gpu.py
if [ "$SKIP_GPU" = false ]; then
    print_header "Step 6/9: Fix graph_builder_gpu.py Import"

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
else
    print_header "Step 6/9: graph_builder_gpu.py (SKIPPED)"
    echo ""
fi

# Step 7: Fix Pipeline cuGraph Dependencies
if [ "$SKIP_GPU" = false ]; then
    print_header "Step 7/9: Fix Pipeline cuGraph Dependencies"

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
else
    print_header "Step 7/9: Pipeline cuGraph Dependencies (SKIPPED)"
    echo ""
fi

# Step 8: Run tests
if [ "$SKIP_GPU" = false ]; then
    print_header "Step 8/9: Verification Tests"

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
else
    print_header "Step 8/9: Verification Tests (SKIPPED)"
    echo ""
fi

# Step 9: Summary
print_header "Step 9/9: Setup Summary"

echo ""
print_success "Environment Setup Complete!"
echo ""
echo "System Configuration:"
if [ "$SKIP_GPU" = false ]; then
    echo "  • GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo "  • Python: $(python3 --version | cut -d' ' -f2)"
if [ "$SKIP_GPU" = false ]; then
    echo "  • PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
    echo "  • CuPy: $(python3 -c 'import cupy; print(cupy.__version__)')"
    echo "  • CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
fi
if [ "$SKIP_GCS" = false ] && command -v gcloud &> /dev/null; then
    echo "  • gcloud: $(gcloud version --format='value(core)' 2>/dev/null || echo 'unknown')"
    if [ -f "$GCS_KEY_FILE" ]; then
        echo "  • GCS Auth: ✓ Configured"
    else
        echo "  • GCS Auth: ✗ Key file not found"
    fi
fi
echo ""

# Add PATH export to .bashrc if gcloud was installed
if [ "$SKIP_GCS" = false ] && [ -d "$GCLOUD_SDK_DIR/bin" ] && [ "$DRY_RUN" = false ]; then
    GCLOUD_PATH_EXPORT="export PATH=\"\$PATH:$(pwd)/$GCLOUD_SDK_DIR/bin\""
    if ! grep -q "$GCLOUD_SDK_DIR/bin" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# Google Cloud SDK" >> ~/.bashrc
        echo "$GCLOUD_PATH_EXPORT" >> ~/.bashrc
        print_success "Added gcloud to ~/.bashrc"
        echo "  • Run 'source ~/.bashrc' or start a new shell to use gcloud"
        echo ""
    fi
fi

echo "Next Steps:"
if [ "$SKIP_GCS" = false ]; then
    echo "  • Download data from GCS:"
    echo "    gsutil -m cp -r gs://gogooku-ml-data/output ."
    echo ""
fi
echo "  • Dataset Generation (5 years, GPU-accelerated):"
echo "    make dataset-bg START=2020-10-11 END=2025-10-10"
echo ""
echo "  • Training (with GPU):"
echo "    make train-optimized"
echo ""
echo "  • Quick test:"
echo "    make smoke"
echo ""
echo "Log saved to: $LOG_FILE"
echo "Completed at: $(date)"
