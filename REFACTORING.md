# Dataset Pipeline Refactoring Guide

## Overview

The `make dataset-full-gpu` pipeline has been refactored to improve:
- **Maintainability**: Smaller, focused modules (~200 lines each)
- **Testability**: Clear interfaces and dependency injection
- **Configuration**: Unified Pydantic-based config management
- **Reliability**: Consistent error handling and logging

## What Changed

### 1. Configuration Management

**Before**:
```python
# Scattered across environment variables and CLI args
MAX_CONCURRENT_FETCH = os.getenv("MAX_CONCURRENT_FETCH", "75")
RMM_POOL_SIZE = os.getenv("RMM_POOL_SIZE", "70GB")
# ... 40+ arguments
```

**After**:
```python
from gogooku3.config import DatasetConfig

# Single source of truth with validation
config = DatasetConfig.from_cli_and_env(cli_args)
print(config.jquants.max_concurrent_fetch)  # 75
print(config.gpu.rmm_pool_size)  # "70GB"
```

### 2. API Client Architecture

**Before**:
```python
# Monolithic 2220-line JQuantsAsyncFetcher
fetcher = JQuantsAsyncFetcher(email, password)
await fetcher.authenticate(session)
df = await fetcher.get_daily_quotes(session, date)
```

**After**:
```python
# Modular, focused components
from gogooku3.api import JQuantsClient, PricesFetcher
from gogooku3.config import JQuantsAPIConfig

config = JQuantsAPIConfig(
    auth_email="your@email.com",
    auth_password="password"
)
client = JQuantsClient(config)
await client.authenticate(session)

# Specialized fetchers
prices = PricesFetcher(client)
df = await prices.fetch_daily_quotes_for_date(session, "2025-01-01")
```

### 3. Pipeline Stages

**Before**:
```python
# 1130-line monolithic script
async def main():
    # ... 500+ lines of mixed concerns
    # Fetching, normalizing, enriching all in one function
```

**After** (planned):
```python
# Clear separation of stages
from gogooku3.pipeline import DatasetPipeline

pipeline = DatasetPipeline(config)
result = await pipeline.run()  # Orchestrates all stages
```

## Migration Path

### For Existing Scripts

The old scripts will continue to work with deprecation warnings:

```python
# scripts/pipelines/run_full_dataset.py (legacy)
# Still works, will show: DeprecationWarning

# scripts/pipelines/run_full_dataset_v5.py (new)
# Recommended for new code
```

### For Custom Code

If you have custom code using `JQuantsAsyncFetcher`:

**Option 1: Minimal Change (Backward Compatible)**
```python
# Keep using the old interface (will wrap new components)
from src.gogooku3.components.jquants_async_fetcher import JQuantsAsyncFetcher

fetcher = JQuantsAsyncFetcher(email, password)
# ... existing code works unchanged
```

**Option 2: Migrate to New API**
```python
# Recommended: Use new modular API
from gogooku3.api import JQuantsClient, PricesFetcher
from gogooku3.config import JQuantsAPIConfig

# 1. Create config
config = JQuantsAPIConfig(
    auth_email=email,
    auth_password=password,
    max_concurrent_fetch=75
)

# 2. Create client
client = JQuantsClient(config)
await client.authenticate(session)

# 3. Use specialized fetchers
prices = PricesFetcher(client)
df = await prices.fetch_daily_quotes_bulk(session, business_days)
```

## Configuration Examples

### From Environment Variables

Create `.env`:
```bash
# J-Quants API
JQUANTS_AUTH_EMAIL=your@email.com
JQUANTS_AUTH_PASSWORD=yourpassword
JQUANTS_MAX_CONCURRENT_FETCH=75

# GPU Settings
USE_GPU_ETL=1
RMM_ALLOCATOR=cuda_async
RMM_POOL_SIZE=70GB

# Features
ENABLE_GRAPH_FEATURES=1
ENABLE_DAILY_MARGIN=1
```

Load in Python:
```python
from gogooku3.config import DatasetConfig

config = DatasetConfig()  # Automatically loads from .env
```

### From CLI Arguments

```python
import argparse
from gogooku3.config import DatasetConfig

parser = argparse.ArgumentParser()
parser.add_argument("--start-date", type=str)
parser.add_argument("--end-date", type=str)
args = parser.parse_args()

config = DatasetConfig.from_cli_and_env(vars(args))
```

### Programmatic Configuration

```python
from gogooku3.config import (
    DatasetConfig,
    JQuantsAPIConfig,
    GPUConfig,
    FeatureFlagsConfig
)

config = DatasetConfig(
    start_date="2020-01-01",
    end_date="2025-01-01",
    jquants=JQuantsAPIConfig(
        auth_email="your@email.com",
        auth_password="password",
        max_concurrent_fetch=50  # Override default
    ),
    gpu=GPUConfig(
        use_gpu_etl=True,
        rmm_pool_size="50GB"
    ),
    features=FeatureFlagsConfig(
        graph_features=True,
        daily_margin=True,
        short_selling=False  # Disable specific feature
    )
)
```

## Testing

### Unit Tests

```python
import pytest
from gogooku3.config import DatasetConfig, JQuantsAPIConfig

def test_config_validation():
    with pytest.raises(ValueError):
        # Invalid date range
        DatasetConfig(start_date="2025-01-01", end_date="2020-01-01")

def test_config_defaults():
    config = DatasetConfig()
    assert config.jquants.max_concurrent_fetch == 75
    assert config.gpu.use_gpu_etl == True
```

### Integration Tests

```python
import pytest
from gogooku3.api import JQuantsClient, PricesFetcher
from gogooku3.config import JQuantsAPIConfig

@pytest.mark.integration
async def test_fetch_prices(aiohttp_session):
    config = JQuantsAPIConfig(
        auth_email=os.getenv("JQUANTS_AUTH_EMAIL"),
        auth_password=os.getenv("JQUANTS_AUTH_PASSWORD")
    )
    client = JQuantsClient(config)
    await client.authenticate(aiohttp_session)

    prices = PricesFetcher(client)
    df = await prices.fetch_daily_quotes_for_date(aiohttp_session, "2025-01-10")

    assert not df.is_empty()
    assert "Code" in df.columns
    assert "Close" in df.columns
```

## Rollback

If you need to rollback:

```bash
# The old scripts are preserved
git checkout HEAD~1 scripts/pipelines/run_full_dataset.py

# Or use legacy interface
python scripts/pipelines/run_full_dataset.py \
  --jquants --start-date 2020-01-01 --end-date 2025-01-01
```

## Next Steps

1. ✅ **Phase 1**: Configuration management (COMPLETED)
2. ✅ **Phase 2**: Base API client (COMPLETED)
3. 🚧 **Phase 3**: Pipeline stages (IN PROGRESS)
4. 📋 **Phase 4**: Main script refactoring (PLANNED)
5. 📋 **Phase 5**: Testing & documentation (PLANNED)

## Support

For issues or questions:
1. Check existing GitHub issues
2. Review CLAUDE.md for architecture details
3. Create new issue with `refactoring` label
