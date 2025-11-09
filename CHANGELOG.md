# Changelog

All notable changes to gogooku3-standalone will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- 📐 **Canonical OHLCV Governance**: standardized dataset builder to use split-adjusted `Adjustment*` prices exclusively, persist schema governance metadata, and fail-fast when non-canonical columns appear.
- 🧾 **Feature Manifest Enforcement**: emit `feature_index.json` with canonical column order/dtypes/normalization metadata, update dataset metadata with hash summaries, and teach loaders to fail-fast on manifest mismatches (optional strict mode).
- ⚡ **Gap Decomposition**: split `ret_prev_1d` into `gap_ov_prev1`/`gap_id_prev1`, enforce leak-safe prev-day semantics, purge redundant `log_returns_1d`, and add persist-time validation.
- 🕘 **Morning Session Features**: gate `/prices/prices_am` by default with T+1 as-of, emit a minimal six-column feature set (`am_gap_prev_close`, `am_body`, `am_range`, `am_vol_ratio_20`, `am_pos_in_am_range`, `am_to_full_range_prev`, `is_am_valid`), and add CLI/config controls for SAME_DAY_PM scenarios.
- 🚚 **DataLoader Defaults**: switched `ALLOW_UNSAFE_DATALOADER` to `auto` with automatic NUM_WORKERS/PIN_MEMORY/PREFETCH defaults so multi-worker pipelines are enabled out of the box while safe-mode overrides still force single-worker operation.

### Fixed
- 🐛 **DataLoader Hanging Issue**: Fixed missing import causing training script to hang indefinitely
  - Added explicit import of `ProductionDataModuleV2` to prevent silent failures
  - Added regression test with timeout detection to prevent future occurrences
  - Documented fix in `docs/fixes/dataloader_hanging_fix.md`
- 🖥️ **Codex TUI OSC Queries**: Automatically skip OSC 10/11 color queries in SSH/non-truecolor terminals to prevent `10;rgb:...` garbage output ([#4945](https://github.com/openai/codex/issues/4945))

### Added
- 🔒 **Security Hardening**: Environment variable-based credential management
- 🏥 **Health Checks**: Comprehensive health check endpoints (`/healthz`, `/readyz`, `/metrics`)
- 📊 **Monitoring**: Prometheus-compatible metrics exporter
- 🔄 **Log Rotation**: Automated log management with configurable retention
- 🧪 **Testing Suite**: Unit, integration, E2E, and performance tests
- 🔍 **Security Scanning**: Trivy, Gitleaks, Bandit integration in CI/CD
- 📈 **Performance Benchmarks**: Automated performance monitoring and reporting
- 📚 **Documentation**: Comprehensive runbook, security guides, and architecture docs
- 🐳 **Docker Security**: Secure container configurations with environment variable overrides

### Phase 2 Features
- 🧪 **Great Expectations Integration**: Data quality validation with 6 comprehensive checks
- ⚡ **Performance Optimization**: PERF_* flags for Polars streaming, parallel processing, memory optimization
- 📊 **RED/SLA Metrics**: Rate, Error, Duration metrics with SLA compliance tracking
- 🔄 **CI/CD Enhancement**: Benchmark testing, semantic release, backup validation
- 🎯 **Data Quality Framework**: Automated quality gates and validation pipelines

### Phase 3 Features
- 📋 **Enhanced Runbook**: Detailed incident response procedures for all services
- 💾 **Automated Backup Validation**: Daily CI/CD backup integrity verification
- 🏗️ **Architecture Documentation**: 15 detailed Mermaid diagrams and data lineage
- 🚨 **Incident Response**: Comprehensive failure recovery and escalation procedures
- 📈 **Operational Monitoring**: Enhanced observability with custom dashboards

### Changed
- 🔐 **Credential Management**: Moved hardcoded secrets to environment variables
- 📋 **Docker Compose**: Added secure override configuration
- 🔧 **CI/CD Pipeline**: Enhanced with security, testing, and performance automation

### Security
- 🚨 **Critical**: Removed hardcoded credentials from Docker Compose
- 🔑 **Environment Variables**: Implemented secure credential management
- 🛡️ **SAST Integration**: Added automated security scanning
- 📝 **Leak Prevention**: Implemented secrets detection and prevention measures

## [2.0.0] - 2024-01-XX

### Added
- 🎯 **ATFT-GAT-FAN Model**: Advanced graph attention network implementation
- 📊 **632 Stock Coverage**: Optimized stock universe with quality improvements
- ⚡ **Polars Integration**: High-performance data processing engine
- 🏗️ **Modern Architecture**: Modular package structure with proper separation
- 🔄 **Complete Migration**: Full transition from gogooku2 to standalone system

### Changed
- 🏗️ **Architecture**: Complete restructure to modern Python package
- 📈 **Performance**: Significant improvements in data processing speed
- 🔧 **Dependencies**: Updated to latest versions with security patches

### Fixed
- 🐛 **Data Quality**: Enhanced data validation and cleaning pipelines
- 🔧 **Memory Management**: Improved resource utilization
- 📊 **Model Training**: More robust training pipelines

## [1.0.0] - 2023-12-XX

### Added
- 🚀 **Initial Release**: gogooku3-standalone core functionality
- 📊 **ML Pipeline**: Basic machine learning training pipeline
- 🗄️ **Data Processing**: Stock data processing and feature engineering
- 🐳 **Docker Support**: Containerized deployment
- 📋 **Basic Monitoring**: Simple logging and error handling

### Infrastructure
- 🐳 **Docker Compose**: Multi-service container setup
- 📊 **MinIO**: Object storage for data and models
- 🗄️ **ClickHouse**: OLAP database for analytical queries
- 🔄 **Redis**: Caching and session management

---

## Development Guidelines

### Types of Changes
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` for vulnerability fixes

### Version Numbering
We use [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Release Process
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create git tag
4. Deploy to production
5. Update documentation

---

*Changelog automatically generated by CI/CD pipeline*
