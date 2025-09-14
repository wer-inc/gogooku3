# Fill Policy

- Do not forward-fill label columns (`feat_ret_*`) or realized returns (`returns_*`).
- For EOD/T+1 features joined via `effective_date`, leave missing as NULL and add validity masks where needed.
- Avoid global forward-fill across long gaps (holidays/halts). Prefer:
  - Feature-specific defaults (e.g., counts → 0, booleans → 0/False) only where semantically safe.
  - Cross-sectional normalization that handles NULLs without time imputation.
- For training pipelines that use cross-sectional normalizers, prefer `fillna_method='zero'` or mask‑aware transforms for unstable features; avoid `forward_fill` unless explicitly justified for that feature.

This policy reduces stale-value carryover and subtle look‑ahead risks.
