# Exploratory Tests

This directory contains exploratory and manual testing scripts that are not part of the automated test suite.

## Purpose

These scripts are used for:
- Manual API testing and validation
- Data exploration and quality checks
- Performance benchmarking
- Feature development and debugging
- One-off investigations

## Files

### API Testing
- `test_daily_quotes_api.py` - Test JQuants daily quotes API endpoints
- `test_trading_calendar.py` - Test trading calendar and business day calculations

### Feature Testing
- `test_enhanced_financial.py` - Test financial feature engineering
- `test_flow_features.py` - Test flow features from trades_spec data
- `test_flow_join_comparison.py` - Compare different flow joining strategies
- `test_market_features.py` - Test TOPIX market features generation
- `test_market_integration.py` - Test market data integration

## Usage

These are standalone scripts that can be run manually:

```bash
# Test JQuants API daily quotes
python tests/exploratory/test_daily_quotes_api.py

# Test flow features generation
python tests/exploratory/test_flow_features.py

# Compare flow joining strategies
python tests/exploratory/test_flow_join_comparison.py
```

## Environment Setup

Most scripts require JQuants API credentials in `.env`:

```bash
JQUANTS_AUTH_EMAIL=your_email
JQUANTS_AUTH_PASSWORD=your_password
```

## Notes

- These scripts are **NOT** run during CI/CD
- They may have longer execution times
- They may require external API access
- Results may vary based on market data availability
- Use for development and debugging purposes