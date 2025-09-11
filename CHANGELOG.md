# Changelog

All notable changes to gogooku3-standalone will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-09-11 - Minimal Production Configuration

### 🚀 Major Changes
This release refactors gogooku3 into a minimal production configuration by removing development and debugging files while maintaining all core functionality.

### Added
- 🎯 **Core Training Commands**: `make train`, `make safe-train`, `python scripts/train_atft.py`
- 📊 **Data Pipeline Commands**: `make dataset-full`, direct dataset construction scripts
- 🤖 **New Model Implementations**: ATFTGATFANModel, LightGBMFinancialBaseline, QualityFinancialFeaturesGenerator, FinancialGraphBuilder
- 🛡️ **Enhanced Components**: SafeTrainingPipeline, MLDatasetBuilder, CrossSectionalNormalizerV2, WalkForwardSplitterV2
- 🧪 **Comprehensive Testing**: New smoke test suite with 5-component verification (13/13 imports, 5/5 smoke tests, 3/3 core scripts)

### ⚠️ Breaking Changes
- **Deleted Scripts**: `scripts/run_safe_training.py` → `make safe-train`, `scripts/_archive/` → components moved to proper locations
- **Updated Import Paths**: `from scripts._archive.run_pipeline import JQuantsAsyncFetcher` → `from src.data.jquants.fetcher import JQuantsAsyncFetcher`
- **New Model Imports**: Core models now available from `src.gogooku3.models`, `src.gogooku3.training`, `src.gogooku3.features`

### Migration Guide
- **Training Commands**: Use `make safe-train` instead of `python scripts/run_safe_training.py`
- **Import Updates**: Update any imports from `scripts._archive/` to new proper locations
- **New APIs**: Use new model implementations from `src.gogooku3.*` modules

### Verification Results
- **Import Tests**: 13/13 Passed ✅ (All critical imports successful)
- **Core Script Tests**: 3/3 Passed ✅ (All scripts show help without ImportError)  
- **Smoke Tests**: 5/5 Passed ✅ (Core imports, model instantiation, pipeline initialization, legacy compatibility, core script imports)

### Production Benefits
- **Simplified Architecture**: Removed 9,895+ lines of development/debug code
- **Clear Entry Points**: 3 core scripts + Makefile targets
- **Robust Testing**: Comprehensive smoke test coverage
- **Better Documentation**: Clear migration paths and breaking changes
- **Production Ready**: Focus on essential ML pipeline components

## [Unreleased]

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
