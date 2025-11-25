# Instructions for Claude / Codex Agents

- Treat this directory as an independent Python package. Activate its virtual environment when running commands locally.
- Prefer GPU-enabled pipelines (RAPIDS/cuDF) and only fall back to CPU when explicitly requested.
- Run `make lint` and `make test` before handing off work or requesting a review.
- Document non-obvious design decisions in `docs/development/memories.md` and mention the relevant migration phase.
