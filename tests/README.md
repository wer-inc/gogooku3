# Tests Directory Structure

This directory contains all test files for the gogooku3-standalone project.

## Directory Organization

```
tests/
├── unit/                   # Unit tests for individual components
│   └── test_market_features.py
│
├── integration/            # Integration tests
│   ├── test_market_integration.py
│   ├── test_migration_smoke.py
│   ├── test_safety_components.py
│   └── test_gogooku3_improvements.py
│
├── exploratory/           # Exploratory and manual testing scripts
│   ├── test_daily_quotes_api.py      # JQuants API daily quotes testing
│   ├── test_enhanced_financial.py     # Financial features testing
│   ├── test_flow_features.py          # Flow features testing
│   ├── test_flow_join_comparison.py   # Flow join strategy comparison
│   ├── test_market_features.py        # Market features testing
│   ├── test_market_integration.py     # Market integration testing
│   └── test_trading_calendar.py       # Trading calendar testing
│
├── test_data_quality.py    # Data quality validation tests
├── test_jquants_integration.py  # JQuants API integration tests
├── test_safe_joins.py      # Safe join implementation tests
├── test_statements_join.py # Financial statements join tests
├── test_flow_join_asof.py  # Flow as-of join tests
└── test_p0_improvements.py # P0 improvements validation tests

```

## Test Categories

### Unit Tests (`unit/`)
- Isolated component testing
- Fast execution
- No external dependencies
- Run frequently during development

### Integration Tests (`integration/`)
- Test interactions between components
- May use sample data
- Validate complete workflows
- Run before commits

### Exploratory Tests (`exploratory/`)
- Manual testing scripts
- API exploration and validation
- Performance benchmarking
- Data exploration utilities
- Not part of CI/CD pipeline

## Running Tests

### Run all unit tests
```bash
pytest tests/unit/ -v
```

### Run integration tests
```bash
pytest tests/integration/ -v
```

### Run specific test file
```bash
pytest tests/test_p0_improvements.py -v
```

### Run exploratory scripts (manual)
```bash
python tests/exploratory/test_daily_quotes_api.py
```

## Test Coverage

### Core Components Covered
- ✅ Market features generation
- ✅ Flow features and joining strategies
- ✅ Financial statements processing
- ✅ Data safety (Walk-Forward, Cross-sectional normalization)
- ✅ P0 improvements (min_periods, FY×Q YoY, etc.)
- ✅ Trading calendar utilities
- ✅ JQuants API integration

## Adding New Tests

1. **Unit tests**: Add to `tests/unit/` for isolated component testing
2. **Integration tests**: Add to `tests/integration/` for workflow testing
3. **Exploratory scripts**: Add to `tests/exploratory/` for manual testing

## CI/CD Integration

The following tests are run automatically in CI/CD:
- All unit tests in `tests/unit/`
- All integration tests in `tests/integration/`
- Specific validation tests (e.g., `test_p0_improvements.py`)

Exploratory tests in `tests/exploratory/` are excluded from CI/CD.