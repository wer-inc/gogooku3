# Migration Guide

## v1 → v2 Migration

This guide will help you migrate from Gogooku3 v1 to v2.

### Key Changes

1. **Package Structure**: Modern `src/gogooku3/` structure
2. **Configuration**: Updated config paths (see `CLAUDE.md`)
3. **Training Pipeline**: New `SafeTrainingPipeline` with 7-step validation
4. **Feature Generation**: Up to 395 features (currently ~307 active)

### Migration Steps

1. **Update Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Update Imports**:
   ```python
   # Old (v1)
   from scripts.run_safe_training import SafeTrainingPipeline

   # New (v2)
   from gogooku3.training import SafeTrainingPipeline
   ```

3. **Update Configs**:
   - Check `configs/atft/config_production_optimized.yaml`
   - Verify `.env` settings (see `CLAUDE.md` for required variables)

4. **Rebuild Dataset**:
   ```bash
   make dataset-bg  # Recommended: background build with latest features
   ```

5. **Test Training**:
   ```bash
   make train-quick  # 3-epoch validation run
   ```

### Backward Compatibility

All legacy `scripts/` commands still work. Migration is **optional** and can be done gradually.

### Support

- See [CLAUDE.md](./CLAUDE.md) for comprehensive documentation
- Check [FAQ](./docs/faq.md) for common issues
- Review [Changelog](./docs/releases/changelog.md) for detailed changes
