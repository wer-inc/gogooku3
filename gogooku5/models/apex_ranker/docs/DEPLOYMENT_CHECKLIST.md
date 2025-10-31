# APEX-Ranker Production Deployment Checklist

**Version**: 0.2.0-production
**Target Launch**: Mid-November 2025
**Last Updated**: 2025-10-29

---

## Pre-Deployment Phase

### 1. Model Validation ✅
- [x] Phase 3 backtest complete (2023-2025, 688 trading days)
- [x] Enhanced model performance validated:
  - Total return: 56.43%
  - Sharpe ratio: 0.933
  - Max drawdown: 20.01%
- [x] Model comparison complete (enhanced > pruned)
- [x] Phase 4.1 complete: Transaction costs < 20% (cost-aware optimisation)
- [ ] Phase 4.2 complete: Walk-forward validation passed
- [ ] Final model checkpoint saved: `models/apex_ranker_v0.2.0_production.pt`

### 2. Configuration Optimization
- [x] **Rebalancing frequency optimized**:
  - [x] Weekly vs monthly backtest comparison complete
  - [x] Decision made: monthly rebalancing
  - [x] Expected cost reduction: ~86% (156% → ~23%)

- [x] **Portfolio configuration optimized**:
  - [x] Top-K selected (35 holdings)
  - [x] Minimum position size threshold set (2%)
  - [x] Daily turnover limit configured (35%)

- [x] **Production config file created**: `configs/v0_production.yaml`
  ```yaml
  rebalance_freq: monthly
  top_k: 35  # Optimized from Phase 4.1
  min_position_size: 0.015
  max_daily_turnover: 0.20
  horizon: 20
  ```

- [x] **Monthly walk-forward scheduler ready**:
  - [x] Script: `scripts/run_rolling_retrain.py`
  - [x] Packages model+config per month to track degradation
  - [x] Summary JSON emitted for 2024-2025 analysis

### 3. Infrastructure Setup
- [ ] **Production server provisioned**:
  - [ ] GPU instance configured (A100 or equivalent)
  - [ ] Python 3.12+ installed
  - [ ] PyTorch 2.8+ with CUDA support
  - [ ] All dependencies installed (`pip install -e .`)

- [ ] **Staging environment configured**:
  - [ ] Mirrors production setup
  - [ ] Test data available
  - [ ] Used for final validation before launch

- [ ] **Database setup**:
  - [ ] PostgreSQL or equivalent installed
  - [ ] Prediction logging tables created
  - [ ] User/API key tables created
  - [ ] Backup schedule configured

- [ ] **Storage**:
  - [ ] Dataset storage: 500GB+ available
  - [ ] Model checkpoint storage
  - [ ] Log file storage with rotation
  - [ ] Backup storage configured

### 4. Monitoring & Alerting
- [x] **Prometheus deployed**:
  - [x] Metrics collection configured (`ops/monitoring/prometheus.yml`)
  - [ ] Retention policy set (90 days)
  - [ ] Backup schedule configured

- [ ] **Grafana deployed**:
  - [ ] Dashboards created:
    - [ ] Real-time prediction distribution
    - [ ] Portfolio composition
    - [ ] Transaction costs accumulation
    - [ ] Model performance (P@K, returns, Sharpe)
    - [x] API latency/throughput (`ops/monitoring/grafana_dashboard.json`)
    - [ ] System metrics (CPU, GPU, memory)
  - [ ] User accounts created
  - [ ] Alerts configured in Grafana

- [ ] **Alert channels configured**:
  - [ ] Email notifications
  - [ ] Slack/Discord webhook (optional)
  - [ ] PagerDuty integration (optional)

- [ ] **Critical alerts defined**:
  - [ ] Prediction distribution anomalies
  - [ ] Inference failures (>5% error rate)
  - [ ] Latency spikes (>10s p99)
  - [ ] Model staleness (>35 days since update)
  - [ ] GPU out of memory
  - [ ] Dataset unavailable

### 5. API Server
- [x] **FastAPI server implemented (`apex_ranker/api/server.py`)**:
  - [x] `/predict` endpoint functional
  - [x] `/rebalance` endpoint returns cost-aware weights
  - [x] `/optimize` endpoint for custom predictions
  - [x] `/healthz` and `/metrics` endpoints available
  - [x] Request/response logging
  - [x] Error handling, authentication, rate limiting

- [x] **Authentication configured**:
  - [x] API key support via `X-API-Key`
  - [ ] User/team access control
  - [x] Rate limiting per identity (configurable)

- [ ] **Rate limiting implemented**:
  - [ ] Per-user limits (e.g., 100 req/hour)
  - [ ] Global limits (e.g., 1000 req/hour)
  - [ ] 429 responses tested

- [ ] **Load testing**:
  - [ ] Stress test: 100 concurrent requests
  - [ ] Latency under load: p95 < 5s, p99 < 10s
  - [ ] Memory stability over 24 hours

### 6. Data Pipeline
- [ ] **Latest dataset available**:
  - [ ] Data freshness: <24 hours old
  - [ ] All features present (89 features for enhanced model)
  - [ ] No missing data for active stocks
  - [ ] Cross-sectional normalization verified

- [ ] **Automated data refresh**:
  - [ ] Daily cron job configured (6am JST)
  - [ ] Dataset builder runs successfully
  - [ ] Failure alerts configured
  - [ ] Manual fallback documented

- [ ] **Data quality checks**:
  - [ ] Feature distribution monitoring
  - [ ] Missing data detection
  - [ ] Outlier detection
  - [ ] Automated alerts on anomalies

---

## Deployment Phase

### 7. Code Deployment
- [ ] **Git release**:
  - [ ] Code frozen on release branch: `release/v0.2.0`
  - [ ] Git tag created: `v0.2.0-production`
  - [ ] Release notes written
  - [ ] Changelog updated

- [ ] **Staging deployment**:
  - [ ] Code deployed to staging
  - [ ] Model checkpoint deployed
  - [ ] Config files deployed
  - [ ] Environment variables set (`.env`)
  - [ ] Services started successfully

- [ ] **Staging validation**:
  - [ ] Health check passes
  - [ ] Sample predictions tested
  - [ ] Monitoring metrics flowing
  - [ ] Alerts tested (trigger test alerts)
  - [ ] API endpoints tested
  - [ ] Rollback tested successfully

### 8. Production Deployment
- [ ] **Pre-deployment backup**:
  - [ ] Database backed up
  - [ ] Previous model checkpoint saved
  - [ ] Config files backed up
  - [ ] Rollback plan reviewed

- [ ] **Production deployment**:
  - [ ] Code deployed to production
  - [ ] Model checkpoint deployed: `models/apex_ranker_v0.2.0_production.pt`
  - [ ] Config deployed: `configs/v0_production.yaml`
  - [ ] Environment variables set
  - [ ] Services started

- [ ] **Initial validation**:
  - [ ] Health check passes: `curl http://api:8000/health`
  - [ ] Test predictions run successfully
  - [ ] Monitoring dashboards showing data
  - [ ] No critical alerts firing

- [ ] **Shadow mode** (optional, recommended):
  - [ ] Run predictions alongside existing system
  - [ ] Compare outputs (no live trading yet)
  - [ ] Monitor for 3-7 days
  - [ ] Validate consistency and stability

---

## Post-Deployment Phase

### 9. Monitoring & Validation (First 24 Hours)
- [ ] **Continuous monitoring**:
  - [ ] Check Grafana dashboards every 2 hours
  - [ ] Review prediction logs for anomalies
  - [ ] Monitor system resource usage
  - [ ] Watch for alerts

- [ ] **Functional validation**:
  - [ ] Daily predictions generated successfully
  - [ ] Prediction distribution looks normal (scores 0-1, reasonable spread)
  - [ ] API response times acceptable (p95 < 5s)
  - [ ] No errors in logs

- [ ] **Performance validation**:
  - [ ] GPU utilization reasonable (not pegged at 100%)
  - [ ] Memory usage stable (no leaks)
  - [ ] Database queries fast (<100ms)
  - [ ] No unexpected costs (API, compute)

### 10. Monitoring & Validation (First Week)
- [ ] **Daily checks**:
  - [ ] Review daily summary report
  - [ ] Check prediction quality metrics
  - [ ] Verify portfolio construction
  - [ ] Monitor transaction costs

- [ ] **Performance comparison**:
  - [ ] Compare predictions vs previous model (if applicable)
  - [ ] Track P@K metric daily
  - [ ] Monitor feature importance changes
  - [ ] Check for drift or anomalies

- [ ] **Operational checks**:
  - [ ] Data refresh working daily
  - [ ] All cron jobs running
  - [ ] Logs rotating properly
  - [ ] Backups completing successfully

### 11. Monitoring & Validation (First Month)
- [ ] **Weekly performance review**:
  - [ ] Calculate weekly P@K
  - [ ] Review portfolio returns
  - [ ] Analyze transaction costs
  - [ ] Compare vs Phase 3 backtest expectations

- [ ] **Model health check**:
  - [ ] Feature importance drift analysis
  - [ ] Prediction distribution stability
  - [ ] No signs of model decay
  - [ ] Retraining schedule on track

- [ ] **Operational review**:
  - [ ] Review all incidents (if any)
  - [ ] Update runbook based on learnings
  - [ ] Optimize alerts (reduce false positives)
  - [ ] Team feedback on operations

---

## Documentation & Training

### 12. Documentation Complete
- [x] **User documentation**:
  - [x] Inference guide: `INFERENCE_GUIDE.md`
  - [x] Model comparison: `BACKTEST_COMPARISON_2023_2025.md`
  - [ ] API documentation (OpenAPI/Swagger)

- [ ] **Operational documentation**:
  - [ ] Production runbook: `PRODUCTION_RUNBOOK.md`
  - [ ] Incident response plan: `INCIDENT_RESPONSE.md`
  - [ ] Rollback procedures: `ROLLBACK.md`
  - [ ] Deployment checklist: `DEPLOYMENT_CHECKLIST.md` (this file)

- [ ] **Technical documentation**:
  - [ ] Architecture diagram
  - [ ] Data flow diagram
  - [ ] Feature engineering documentation
  - [ ] Model training documentation

### 13. Team Training
- [ ] **Training sessions completed**:
  - [ ] Model overview and performance
  - [ ] Production runbook walkthrough
  - [ ] Monitoring dashboard tour
  - [ ] Incident response procedures
  - [ ] Rollback procedures

- [ ] **Access and permissions**:
  - [ ] Production server access (SSH keys)
  - [ ] Grafana/Prometheus access
  - [ ] Database access (read-only for most)
  - [ ] API keys for testing
  - [ ] Emergency contact list distributed

- [ ] **On-call rotation**:
  - [ ] On-call schedule defined
  - [ ] Primary and secondary contacts
  - [ ] Escalation procedures
  - [ ] Emergency contacts reachable

---

## Final Go/No-Go Decision

### Go Criteria (All Must Pass)
- [ ] All critical items above completed
- [ ] Staging validation passed
- [ ] Team trained and confident
- [ ] Monitoring operational
- [ ] Rollback tested successfully
- [ ] Phase 4.1 and 4.2 complete
- [ ] Transaction costs < 30% target achieved
- [ ] Sharpe ratio > 0.75 in optimized backtest
- [ ] No blockers or critical issues

### No-Go Criteria (Any One Fails)
- [ ] Staging validation failures
- [ ] Critical monitoring alerts
- [ ] Team not confident in operations
- [ ] Rollback procedure untested
- [ ] Transaction costs still > 50%
- [ ] Performance degradation vs Phase 3
- [ ] Critical dependencies unavailable

---

## Emergency Contacts

**Primary Contacts**:
- ML Engineer: [Name] ([Email]) ([Phone])
- Backend Engineer: [Name] ([Email]) ([Phone])
- DevOps: [Name] ([Email]) ([Phone])

**Escalation**:
- Engineering Manager: [Name] ([Email]) ([Phone])
- CTO: [Name] ([Email]) ([Phone])

**External Support**:
- Cloud Provider Support: [URL] ([Phone])
- Database Support: [URL] ([Phone])

---

## Post-Launch Review (After 1 Month)

### Scheduled Review Meeting
- [ ] **Date**: [TBD, ~1 month after launch]
- [ ] **Attendees**: Full team + stakeholders
- [ ] **Agenda**:
  - Review first month performance
  - Discuss incidents and learnings
  - Optimize based on operational experience
  - Plan Phase 5 improvements

### Review Topics
- [ ] Model performance vs expectations
- [ ] Transaction costs achieved vs target
- [ ] Operational challenges faced
- [ ] Monitoring effectiveness
- [ ] Team feedback on runbook
- [ ] Ideas for Phase 5

---

## Sign-Off

### Deployment Approval
- [ ] **ML Engineer**: _________________ Date: _________
  - Model validated and ready

- [ ] **Backend Engineer**: _________________ Date: _________
  - API and infrastructure ready

- [ ] **DevOps**: _________________ Date: _________
  - Deployment and monitoring ready

- [ ] **Engineering Manager**: _________________ Date: _________
  - Overall deployment approved

---

## Notes

- This checklist should be reviewed and updated based on team needs
- Some items may not apply depending on deployment environment
- Add organization-specific requirements as needed
- Keep this document updated with actual deployment dates and outcomes
- Archive this checklist after successful deployment for future reference

---

**Generated**: 2025-10-29
**Author**: Claude Code (Autonomous Development Agent)
**Status**: Template (to be filled during Phase 4)
- [ ] **Release artifacts packaged**:
  - [x] Model bundle script `scripts/package_production_bundle.py`
  - [ ] Bundle archived to `production/apex_ranker_bundle.tar.gz`
  - [ ] Metadata stored alongside release
