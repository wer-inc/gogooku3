# Dagster Home (.dagster/)

This directory is intended to be used as `DAGSTER_HOME` for gogooku5.

- `dagster.yaml` is read from `<DAGSTER_HOME>/dagster.yaml`.
- Run storage and event log storage are configured to write under
  `<DAGSTER_HOME>/.dagster/storage` (see `dagster.yaml`).

Notes:

- All files under `.dagster/` are **runtime artifacts or configuration**
  and should not be committed to Git, except for `dagster.yaml` if you
  choose to keep it here.
- It is safe to delete the storage subdirectory; Dagster will recreate
  the necessary files the next time you execute jobs.
