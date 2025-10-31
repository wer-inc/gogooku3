# Monitoring Stack

This folder provides ready-to-use configuration for the production observability stack.

- `prometheus.yml` – scrape definition for the APEX-Ranker FastAPI service. Configure
  retention (e.g. `--storage.tsdb.retention.time=90d`) when launching Prometheus.
- `grafana_dashboard.json` – starter dashboard covering prediction latency, optimisation
  latency, and request throughput. Import into Grafana and wire the Prometheus datasource
  UID to the `prometheus` placeholder used in the dashboard.

Suggested deployment steps:
1. Launch Prometheus with the scrape config and desired retention / storage flags.
2. Import the Grafana dashboard and extend it with transaction-cost and portfolio metrics
   once those time-series are available.
3. Configure alert rules on latency, error rate, and staleness thresholds consistent with
   the deployment checklist.
