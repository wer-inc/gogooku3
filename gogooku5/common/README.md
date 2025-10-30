# gogooku5 Common Utilities (Optional)

Only add shared components here when at least two packages require them.

## Structure
- `src/common/data/`: dataset and loader abstractions.
- `src/common/metrics/`: evaluation metrics reused across models.
- `src/common/utils/`: logging, monitoring, or storage helpers.

Document each addition with the consuming packages and update `docs/development/memories.md` explaining why commonization was necessary.
