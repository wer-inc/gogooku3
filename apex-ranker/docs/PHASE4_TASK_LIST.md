# APEX-Ranker Phase 4: Cost Optimization & Production Deployment

**Created**: 2025-10-29
**Status**: Planning
**Target Launch**: Mid-November 2025

---

## Overview

**Objective**: Reduce transaction costs from 156% to <30% of capital and deploy enhanced model to production with monitoring infrastructure.

**Phase 3 Results**:
- ✅ Enhanced model: 56.43% return, Sharpe 0.933
- ⚠️ Transaction costs: ¥15.6M (156% of capital)
- 📊 Backtest period: 2023-2025 (688 trading days)

**Critical Issue**: Weekly rebalancing with Top-50 portfolio generates excessive transaction costs that significantly reduce net returns.

---

## Phase 4.1: Transaction Cost Reduction (Week 1-2)

### Objective
Reduce transaction costs from 156% to <30% of capital through optimized rebalancing and portfolio configuration.

### Task 1.1: Monthly Rebalancing Implementation
**Priority**: 🔴 Critical
**Owner**: TBD
**Effort**: 2-3 days

**Description**:
Modify backtest script to support monthly rebalancing frequency instead of weekly.

**Deliverables**:
- [ ] Add `--rebalance-freq` parameter to `backtest_smoke_test.py` (values: daily, weekly, monthly)
- [ ] Update portfolio rebalancing logic in `apex_ranker/backtest/portfolio.py`
- [ ] Run comparative backtest: weekly vs monthly (2023-2025)
- [ ] Document performance trade-offs (return vs cost reduction)

**Expected Impact**:
- Transaction costs: 156% → ~40% (~75% reduction)
- Total return: 56.43% → ~48-52% (expected 5-15% degradation)
- Net impact: Positive (cost savings > return loss)

**Success Criteria**:
- Monthly backtest completes successfully
- Transaction costs < 50% of capital
- Sharpe ratio remains > 0.8

---

### Task 1.2: Portfolio Configuration Optimization
**Priority**: 🟡 High
**Owner**: TBD
**Effort**: 2-3 days

**Description**:
Optimize portfolio size and position constraints to reduce unnecessary trades.

**Sub-tasks**:
- [ ] **Top-K reduction experiment**: Test 30, 40, 50 stocks
  - Hypothesis: Smaller portfolio = lower turnover, similar performance
  - Backtest each configuration (2023-2025)
  - Compare P@K, transaction costs, Sharpe ratio

- [ ] **Minimum position size threshold**: Implement trade filtering
  - Add `--min-position-size` parameter (e.g., 1% of portfolio)
  - Skip trades below threshold (avoid small inefficient trades)
  - Measure impact on costs and turnover

- [ ] **Turnover constraints**: Add daily/weekly turnover limits
  - Cap maximum daily turnover (e.g., 20% of portfolio)
  - Implement gradual rebalancing if limit exceeded
  - Test impact on tracking error and costs

**Expected Impact**:
- Top-K 30: ~25% cost reduction vs Top-K 50
- Min position size: ~10-15% cost reduction
- Turnover constraints: ~15-20% cost reduction
- **Combined**: Target <30% total transaction costs

**Success Criteria**:
- Find optimal Top-K balancing performance and costs
- Reduce transaction costs to <30% of capital
- Maintain Sharpe ratio > 0.75

---

### Task 1.3: Cost-Aware Portfolio Optimization
**Priority**: 🟢 Medium
**Owner**: TBD
**Effort**: 3-4 days

**Description**:
Integrate transaction costs directly into portfolio construction objective function.

**Sub-tasks**:
- [ ] **Objective function design**:
  ```python
  # Current: max(expected_return)
  # New: max(expected_return - transaction_cost - turnover_penalty)
  objective = (
      alpha * expected_return
      - beta * transaction_cost
      - gamma * turnover_penalty
  )
  ```

- [ ] **Markowitz + cost constraints**: Implement mean-variance optimization with cost integration
  - Add transaction cost estimation to optimization
  - Test different cost sensitivity parameters (beta, gamma)
  - Compare with simple Top-K ranking

- [ ] **Turnover penalties**: Penalize excessive portfolio changes
  - Add quadratic turnover penalty to objective
  - Test different penalty strengths
  - Measure impact on stability and costs

**Expected Impact**:
- Smoother portfolio changes (lower turnover)
- 10-20% additional cost reduction
- Potentially higher risk-adjusted returns

**Success Criteria**:
- Cost-aware optimization reduces costs vs baseline
- Portfolio turnover decreases by >30%
- Sharpe ratio remains > 0.75

---

## Phase 4.2: Walk-Forward Validation Framework (Week 2-3)

### Objective
Validate model robustness over time and establish optimal retraining schedule to prevent model decay.

### Task 2.1: Rolling Window Implementation
**Priority**: 🟡 High
**Owner**: TBD
**Effort**: 3-4 days

**Description**:
Implement rolling walk-forward validation with periodic model retraining.

**Sub-tasks**:
- [ ] **WalkForwardValidator class**: Create validation framework
  ```python
  class WalkForwardValidator:
      def __init__(self, train_window=252, retrain_freq=21):
          self.train_window = train_window  # 1 year
          self.retrain_freq = retrain_freq  # Monthly (21 trading days)

      def split_data(self, data, start_date, end_date):
          """Generate train/test splits with rolling window"""
          ...

      def validate(self, model, data, splits):
          """Run validation across all splits"""
          ...
  ```

- [ ] **Training window size experiments**: Test different lookback periods
  - 126 days (6 months)
  - 252 days (1 year) ← baseline
  - 504 days (2 years)
  - Measure out-of-sample P@K for each

- [ ] **Retraining frequency experiments**: Test different schedules
  - Weekly (5 trading days)
  - Bi-weekly (10 trading days)
  - Monthly (21 trading days) ← baseline
  - Quarterly (63 trading days)
  - Measure P@K vs training cost trade-off

- [ ] **Out-of-sample backtest**: Run full walk-forward backtest (2023-2025)
  - Retrain monthly with 1-year rolling window
  - Compare vs static model (current)
  - Document performance over time

**Expected Impact**:
- Adaptive model maintains P@K > 0.55 over time
- Static model may degrade to P@K ~0.50 after 1 year
- ~5-10% improvement in long-term returns

**Success Criteria**:
- Walk-forward framework implemented and tested
- Optimal retraining schedule identified
- Out-of-sample backtest shows stable/improving performance

---

### Task 2.2: Model Decay Analysis
**Priority**: 🟢 Medium
**Owner**: TBD
**Effort**: 2-3 days

**Description**:
Analyze model performance degradation over time and identify retraining triggers.

**Sub-tasks**:
- [ ] **Decay curve visualization**: Plot P@K over time since training
  - Track daily P@K for static model
  - Identify decay rate (e.g., -0.01 P@K per month)
  - Determine when retraining is critical

- [ ] **Feature importance drift**: Track feature contribution changes
  - Monitor top-10 feature importance monthly
  - Detect regime changes (e.g., volatility spikes)
  - Flag when feature distribution shifts

- [ ] **Adaptive retraining triggers**: Implement performance-based retraining
  - Trigger 1: P@K drops below threshold (e.g., 0.50)
  - Trigger 2: Feature drift exceeds threshold
  - Trigger 3: Time-based (monthly fallback)
  - Compare adaptive vs fixed schedule

**Expected Impact**:
- Early detection of model degradation
- Optimized retraining schedule (retrain only when needed)
- Reduced training costs while maintaining performance

**Success Criteria**:
- Decay patterns documented and visualized
- Adaptive retraining reduces training cost by >50% vs monthly
- Performance maintained (P@K > 0.55)

---

## Phase 4.3: Production Deployment Preparation (Week 3-4)

### Objective
Package enhanced model for production deployment with monitoring, API, and operational procedures.

### Task 3.1: Model/Config Packaging
**Priority**: 🔴 Critical
**Owner**: TBD
**Effort**: 1-2 days

**Description**:
Prepare production-ready model package with optimized configuration.

**Deliverables**:
- [ ] **Production model checkpoint**: `models/apex_ranker_v0.2.0_production.pt`
  - Copy enhanced model with best Phase 4.1 configuration
  - Version tag: v0.2.0-production
  - Document training date, dataset, hyperparameters

- [ ] **Production config**: `apex-ranker/configs/v0_production.yaml`
  ```yaml
  # Optimized production settings
  rebalance_freq: monthly  # From Phase 4.1
  top_k: 35                # From Phase 4.1 optimization
  min_position_size: 0.015 # 1.5% of portfolio
  max_daily_turnover: 0.20 # 20% limit
  horizon: 20              # 20-day prediction
  ```

- [ ] **Environment setup**: Production `.env` template
  - API keys, database connections
  - Monitoring endpoints
  - Logging configuration

**Success Criteria**:
- Model and config tested in staging environment
- All parameters optimized from Phase 4.1 results
- Documentation complete (README, runbook)

---

### Task 3.2: Monitoring Infrastructure
**Priority**: 🔴 Critical
**Owner**: TBD
**Effort**: 3-4 days

**Description**:
Implement comprehensive monitoring for production model.

**Sub-tasks**:
- [ ] **Prometheus metrics export**:
  ```python
  # apex-ranker/monitoring/metrics.py
  from prometheus_client import Counter, Histogram, Gauge

  PREDICTIONS_TOTAL = Counter('apex_predictions_total', 'Total predictions')
  PREDICTION_SCORE_DIST = Histogram('apex_prediction_score', 'Prediction score distribution')
  INFERENCE_LATENCY = Histogram('apex_inference_latency_seconds', 'Inference latency')
  MODEL_VERSION = Gauge('apex_model_version', 'Current model version')
  ```

- [ ] **Alerting rules**: Define critical alerts
  - Prediction distribution anomalies (e.g., all scores > 0.9)
  - Inference failures (> 5% error rate)
  - Latency spikes (> 10 seconds p99)
  - Model staleness (last update > 35 days)

- [ ] **Grafana dashboards**: Create visualization dashboards
  - Real-time prediction distribution
  - Portfolio composition over time
  - Transaction costs accumulation
  - Model performance (P@K, returns, Sharpe)

**Expected Impact**:
- Early detection of production issues
- Visibility into model behavior
- Data-driven operational decisions

**Success Criteria**:
- All metrics exported to Prometheus
- Alerts configured and tested (test scenarios trigger alerts)
- Dashboards accessible and informative

---

### Task 3.3: API Server (FastAPI)
**Priority**: 🟡 High
**Owner**: TBD
**Effort**: 3-4 days

**Description**:
Create REST API wrapper for inference service.

**Sub-tasks**:
- [ ] **FastAPI application**: Implement API server
  ```python
  # apex-ranker/api/server.py
  from fastapi import FastAPI, HTTPException
  from pydantic import BaseModel

  app = FastAPI(title="APEX-Ranker API", version="0.2.0")

  class PredictionRequest(BaseModel):
      date: str  # YYYY-MM-DD
      top_k: int = 50
      horizon: int = 20

  class PredictionResponse(BaseModel):
      date: str
      predictions: list[dict]  # [{code, score, rank}, ...]
      model_version: str
      latency_ms: float

  @app.post("/predict", response_model=PredictionResponse)
  async def predict(request: PredictionRequest):
      """Generate top-K predictions for given date"""
      ...

  @app.get("/health")
  async def health():
      """Health check endpoint"""
      return {"status": "healthy", "model_version": "0.2.0"}
  ```

- [ ] **Request logging**: Log all API requests
  - Request params (date, top_k, horizon)
  - Response data (predictions, latency)
  - Error details (if failed)
  - Store in database for auditing

- [ ] **Rate limiting**: Implement request throttling
  - Per-user limits (e.g., 100 req/hour)
  - Global limits (e.g., 1000 req/hour)
  - 429 Too Many Requests response

- [ ] **Authentication**: Add API key validation
  - API key generation and management
  - User/team-based access control
  - Usage tracking per key

**Expected Impact**:
- Easy integration with trading systems
- Controlled access and usage monitoring
- Production-ready inference service

**Success Criteria**:
- API endpoints functional and tested
- Response latency < 5 seconds p95
- Health checks pass consistently
- Authentication working correctly

---

### Task 3.4: Release Checklist & Runbook
**Priority**: 🟡 High
**Owner**: TBD
**Effort**: 1-2 days

**Description**:
Create operational documentation for production deployment and incident response.

**Deliverables**:
- [ ] **Pre-deployment checklist**: `apex-ranker/docs/DEPLOYMENT_CHECKLIST.md`
  - [ ] Model trained and validated (Phase 4.1 complete)
  - [ ] Walk-forward validation passed (Phase 4.2 complete)
  - [ ] Monitoring infrastructure deployed (Phase 4.3.2 complete)
  - [ ] API server tested in staging (Phase 4.3.3 complete)
  - [ ] Rollback procedures documented
  - [ ] Team training completed
  - [ ] Production credentials configured
  - [ ] Data pipeline verified (latest dataset available)
  - [ ] Backup systems tested

- [ ] **Production runbook**: `apex-ranker/docs/PRODUCTION_RUNBOOK.md`
  - **Daily operations**:
    - Data refresh schedule (6am JST)
    - Prediction generation (8am JST)
    - Portfolio rebalancing (monthly, 1st trading day)
  - **Monitoring**:
    - Dashboard URLs (Grafana, Prometheus)
    - Key metrics to watch
    - Normal vs abnormal ranges
  - **Common issues and resolutions**:
    - Prediction failures → Check dataset availability
    - High latency → Check GPU utilization
    - Distribution anomalies → Trigger manual review
  - **Emergency contacts**: Team members, escalation paths

- [ ] **Incident response plan**: `apex-ranker/docs/INCIDENT_RESPONSE.md`
  - **Severity levels**:
    - P0 (Critical): Model producing invalid predictions
    - P1 (High): API unavailable, prediction failures >50%
    - P2 (Medium): High latency (>10s p95), monitoring alerts
    - P3 (Low): Non-critical warnings, feature drift
  - **Response procedures**:
    - P0: Immediate rollback to previous model
    - P1: Investigate within 1 hour, fix within 4 hours
    - P2: Investigate within 4 hours, fix within 24 hours
    - P3: Add to backlog, fix within 1 week
  - **Communication plan**: Who to notify, when to escalate

- [ ] **Rollback procedures**: `apex-ranker/docs/ROLLBACK.md`
  - **Automated rollback**: Script to revert to previous model
    ```bash
    # apex-ranker/scripts/rollback.sh
    #!/bin/bash
    # Rollback to previous production model
    PREV_MODEL="models/apex_ranker_v0.1.0_production.pt"
    PREV_CONFIG="configs/v0.1.0_production.yaml"

    # Swap models
    mv models/apex_ranker_production.pt models/apex_ranker_production_backup.pt
    cp $PREV_MODEL models/apex_ranker_production.pt
    cp $PREV_CONFIG configs/production.yaml

    # Restart services
    systemctl restart apex-ranker-api

    # Verify health
    curl http://localhost:8000/health
    ```
  - **Manual rollback steps**: Step-by-step guide
  - **Validation**: How to verify rollback success

**Success Criteria**:
- All documentation complete and reviewed
- Team trained on runbook procedures
- Rollback tested successfully in staging

---

## Timeline & Milestones

### Week 1: Transaction Cost Reduction
- **Days 1-3**: Task 1.1 (Monthly rebalancing)
- **Days 4-5**: Task 1.2 (Portfolio optimization)
- **Days 6-7**: Task 1.3 (Cost-aware optimization)
- **Milestone**: Transaction costs < 30% of capital

### Week 2: Walk-Forward Validation
- **Days 1-4**: Task 2.1 (Rolling window implementation)
- **Days 5-7**: Task 2.2 (Model decay analysis)
- **Milestone**: Walk-forward framework validated

### Week 3: Production Infrastructure
- **Days 1-2**: Task 3.1 (Model/config packaging)
- **Days 3-5**: Task 3.2 (Monitoring infrastructure)
- **Days 6-7**: Task 3.3 (API server - initial)

### Week 4: Final Deployment Prep
- **Days 1-2**: Task 3.3 (API server - complete)
- **Days 3-4**: Task 3.4 (Release checklist & runbook)
- **Days 5-7**: Final testing, staging validation, team training
- **Milestone**: Production launch

---

## Success Metrics

### Phase 4.1 Success (Cost Reduction)
- ✅ Transaction costs reduced to <30% of capital (from 156%)
- ✅ Sharpe ratio maintained > 0.75 (vs 0.933 baseline)
- ✅ Total return > 40% over 2023-2025 period

### Phase 4.2 Success (Walk-Forward Validation)
- ✅ Walk-forward validation framework implemented
- ✅ Model decay patterns documented
- ✅ Optimal retraining schedule identified
- ✅ Out-of-sample P@K > 0.55 maintained

### Phase 4.3 Success (Production Deployment)
- ✅ API server deployed and tested
- ✅ Monitoring dashboards operational
- ✅ All documentation complete
- ✅ Team trained on runbook
- ✅ Rollback procedures tested

### Overall Phase 4 Success
- ✅ Enhanced model deployed to production
- ✅ Transaction costs < 30% of capital
- ✅ Sharpe ratio > 0.75
- ✅ Monitoring and alerting operational
- ✅ Team confident in production operations

---

## Risk Mitigation

### Risk 1: Cost reduction degrades performance too much
**Probability**: Medium
**Impact**: High
**Mitigation**:
- Run extensive backtests before deployment
- Set minimum acceptable Sharpe ratio (0.75)
- Keep weekly rebalancing as fallback option
- Monitor live performance closely in first month

### Risk 2: Walk-forward validation reveals model instability
**Probability**: Medium
**Impact**: High
**Mitigation**:
- Implement adaptive retraining triggers
- Keep multiple model versions ready for quick rollback
- Test ensemble methods (combine multiple models)
- Consider shorter retraining windows if decay is fast

### Risk 3: Production deployment issues (API, monitoring)
**Probability**: Low
**Impact**: Medium
**Mitigation**:
- Thorough staging environment testing
- Gradual rollout (shadow mode before live)
- Comprehensive monitoring and alerting
- Documented rollback procedures

### Risk 4: Team not ready for production operations
**Probability**: Low
**Impact**: High
**Mitigation**:
- Comprehensive training sessions
- Clear documentation (runbook, incident response)
- On-call rotation established
- Initial period with external support

---

## Dependencies

### Internal Dependencies
- Phase 3 complete (backtest validation) ✅
- Enhanced model trained and available ✅
- Dataset pipeline operational ✅
- GPU infrastructure available ✅

### External Dependencies
- Production server environment (staging + production)
- Monitoring infrastructure (Prometheus, Grafana)
- API hosting (Docker, Kubernetes, or similar)
- Database for logging (PostgreSQL recommended)

### Team Dependencies
- ML Engineer: Model optimization, walk-forward validation
- Backend Engineer: API server, monitoring infrastructure
- DevOps: Deployment, CI/CD, production environment
- Quant Researcher: Portfolio optimization, performance analysis

---

## Notes

- This task list is comprehensive but can be parallelized
- Some tasks can be done concurrently (e.g., 1.2 + 1.3, 3.2 + 3.3)
- Adjust priorities based on team availability and production urgency
- Mid-November target is aggressive - allow buffer for unexpected issues
- Consider soft launch (internal only) before full production deployment

---

**Last Updated**: 2025-10-29
**Author**: Claude Code (Autonomous Development Agent)
