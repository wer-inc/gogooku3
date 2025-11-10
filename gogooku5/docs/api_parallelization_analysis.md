# J-Quants API Parallelization Analysis (gogooku5)

## Long-running Fetch Workflows

| Phase | Module/Function | Description | Current Runtime | Notes |
|-------|-----------------|-------------|-----------------|-------|
| Index Options (1st pass) | `builder.api.jquants_async_fetcher.fetch_index_option` | Bulk range fetch with pagination | ~25-30 min for 6 months | API often returns large payloads; fallback path triggered frequently |
| Index Options (2nd pass) | same as above | Daily refetch to backfill missing days | ~90 min for 588 days | Sequential day loop, single request at a time |
| Futures fallback | `fetch_futures` (date-by-date) | Daily fallback for derivatives | ~12-15 min for ~200 days | Currently sequential |
| Financial statements | `advanced_fetcher.fetch_fs_details` | Chunked by API default | 10-12 min for 6 months | Each request still sequential per date range |
| Dividends / earnings / short positions | `fetch_dividends`, `fetch_earnings`, `fetch_short_positions` | Date-range fetch, repeated per chunk | 5-8 min combined | API throughput OK but could parallelize |

## Concurrency Targets & Rate-Limit Rules

| API | Proposed Task Granularity | Target Concurrency | Rate-Limit Constraints | Retry Strategy |
|-----|---------------------------|--------------------|------------------------|----------------|
| Index Options | Day × Option Category | 4 tasks | J-Quants recommends <= 5 concurrent connections | Retry up to 3 times with exponential backoff (base 2s) |
| Futures | Day × Contract (fallback) | 8 tasks | Data volume moderate; ensure 200ms spacing per task | Same as above |
| Financial Statements | 7-day chunks | 3 tasks | API throttles >10 req/min; chunk size ensures compliance | Retry 2 times |
| Dividends/Earnings | 14-day chunks | 4 tasks | Lightweight responses; concurrency mainly I/O bound | Retry 2 times |
| Short Positions | Weekly chunks | 4 tasks | Similar to dividends; limited by server response time | Retry 3 times |

## Planned Settings

- `api_max_concurrency`: default 4 (override per endpoint)
- `api_batch_days`: default 7 (index options override to 1)
- `api_retry_limit`: default 3
- `api_retry_backoff`: exponential, base 2.0 seconds (jitter ±0.5s)
- `api_timeout_seconds`: default 60
- `api_enable_parallel_fetch`: feature flag for gradual rollout

## Next Steps

1. Implement async utility with semaphore + retry.
2. Refactor index option / futures fetchers to use parallel helpers.
3. Extend to financial APIs and hook into DatasetBuilder status reporting.
