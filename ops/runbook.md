# 🚨 gogooku3-standalone Operations Runbook

## Overview

This runbook provides operational procedures for maintaining, troubleshooting, and scaling the gogooku3-standalone application. It covers common issues, preventive maintenance, and emergency procedures.

## 📋 Table of Contents

1. [Quick Reference](#quick-reference)
2. [Health Checks](#health-checks)
3. [Common Issues & Solutions](#common-issues--solutions)
4. [Maintenance Procedures](#maintenance-procedures)
5. [Emergency Procedures](#emergency-procedures)
6. [Scaling Procedures](#scaling-procedures)
7. [Backup & Recovery](#backup--recovery)
8. [Monitoring & Alerts](#monitoring--alerts)

## 🔍 Quick Reference

### Service Status Check
```bash
# Overall health check
python ops/health_check.py health

# Readiness check
python ops/health_check.py ready

# Liveness check
python ops/health_check.py live

# Metrics endpoint
python ops/metrics_exporter.py --once
```

### Log Locations
- Main logs: `logs/main.log`
- ML training logs: `logs/ml_training.log`
- Safe training logs: `logs/safe_training.log`
- Docker logs: `docker logs gogooku3-minio`

### Configuration Files
- Main config: `pyproject.toml`
- Docker: `docker-compose.yml` / `docker-compose.override.yml`
- Environment: `.env` (create from `.env.example`)

## 🏥 Health Checks

### Automated Health Checks

1. **System Health Check**
   ```bash
   python ops/health_check.py health --format json
   ```

2. **Readiness Check** (for load balancers)
   ```bash
   python ops/health_check.py ready
   ```

3. **Liveness Check** (for orchestrators)
   ```bash
   python ops/health_check.py live
   ```

### Manual Health Verification

1. **Database Connectivity**
   ```bash
   # Check ClickHouse
   docker exec gogooku3-clickhouse clickhouse-client --query "SELECT 1"

   # Check Redis
   docker exec gogooku3-redis redis-cli ping
   ```

2. **Storage Connectivity**
   ```bash
   # Check MinIO
   docker exec gogooku3-minio mc ls gogooku
   ```

3. **Dependencies**
   ```bash
   python -c "import polars, torch, pandas; print('Dependencies OK')"
   ```

## 🔧 Common Issues & Solutions

### Issue 1: Memory Exhaustion

**Symptoms:**
- Application becomes unresponsive
- High memory usage alerts
- Out of memory errors in logs

**Solutions:**

1. **Immediate Actions:**
   ```bash
   # Check memory usage
   python ops/health_check.py health | grep memory

   # Restart application
   docker compose restart
   ```

2. **Root Cause Analysis:**
   ```bash
   # Check process memory
   ps aux --sort=-%mem | head -10

   # Analyze logs for memory leaks
   tail -f logs/main.log | grep -i memory
   ```

3. **Preventive Measures:**
   - Increase memory limits in docker-compose.yml
   - Enable memory profiling: `PERF_MEMORY_PROFILE=1`
   - Implement memory limits: `docker run --memory=16g`

### Issue 2: Training Pipeline Failures

**Symptoms:**
- Training jobs fail with errors
- No model outputs generated
- Error messages in training logs

**Solutions:**

1. **Check Data Availability:**
   ```bash
   # Verify data files exist
   ls -la data/processed/

   # Check data quality
   python -c "import polars as pl; df = pl.read_parquet('data/processed/dataset.parquet'); print(df.shape)"
   ```

2. **Validate Configuration:**
   ```bash
   # Check environment variables
   env | grep -E "(MINIO|CLICKHOUSE|REDIS)"

   # Validate training config
   python main.py safe-training --mode quick --dry-run
   ```

3. **Restart Training:**
   ```bash
   # Clean previous state
   rm -rf output/checkpoints/*
   rm -rf logs/ml_training.log

   # Restart with verbose logging
   LOG_LEVEL=DEBUG python main.py safe-training --mode full
   ```

### Issue 3: Database Connection Issues

**Symptoms:**
- ClickHouse connection errors
- Data loading failures
- Timeout errors

**Solutions:**

1. **Check Database Status:**
   ```bash
   # Check ClickHouse health
   docker ps | grep clickhouse

   # Test connection
   docker exec gogooku3-clickhouse clickhouse-client --query "SELECT version()"
   ```

2. **Restart Database:**
   ```bash
   # Restart ClickHouse
   docker compose restart clickhouse

   # Check logs
   docker logs gogooku3-clickhouse
   ```

3. **Connection Pool Issues:**
   ```bash
   # Check connection limits
   docker exec gogooku3-clickhouse clickhouse-client --query "SELECT * FROM system.settings WHERE name LIKE '%max_%'"
   ```

### Issue 4: Storage Issues

**Symptoms:**
- MinIO connection failures
- File upload/download errors
- Storage quota exceeded

**Solutions:**

1. **Check Storage Status:**
   ```bash
   # Check MinIO status
   docker ps | grep minio

   # Test storage access
   docker exec gogooku3-minio mc ls gogooku
   ```

2. **Storage Cleanup:**
   ```bash
   # Check storage usage
   docker exec gogooku3-minio mc du gogooku/

   # Clean old files (if applicable)
   docker exec gogooku3-minio mc rm --older-than 30d gogooku/backups/
   ```

3. **Increase Storage:**
   ```yaml
   # docker-compose.yml
   minio:
     volumes:
       - minio_data:/data
   # Add more storage space to host system
   ```

## 🛠️ Maintenance Procedures

### Weekly Maintenance

1. **Log Rotation Check:**
   ```bash
   # Check log sizes
   du -sh logs/*.log

   # Force log rotation if needed
   sudo logrotate -f /etc/logrotate.d/gogooku3
   ```

2. **Disk Space Monitoring:**
   ```bash
   # Check disk usage
   df -h /

   # Clean old logs
   find logs/ -name "*.log" -mtime +7 -delete
   ```

3. **Backup Verification:**
   ```bash
   # List recent backups
   ls -la backups/

   # Verify backup integrity
   tar -tzf backups/latest.tar.gz | head -10
   ```

### Monthly Maintenance

1. **Security Updates:**
   ```bash
   # Update Python packages
   pip list --outdated
   pip install --upgrade -r requirements.txt

   # Update Docker images
   docker compose pull
   ```

2. **Performance Tuning:**
   ```bash
   # Analyze performance metrics
   python ops/metrics_exporter.py --once

   # Check system resources
   python ops/health_check.py health
   ```

3. **Data Quality Assessment:**
   ```bash
   # Run data quality checks
   python -m pytest tests/ -k "quality"

   # Validate data integrity
   python scripts/data/validate_dataset.py
   ```

## 🚨 Emergency Procedures

### Application Down (Critical)

1. **Immediate Assessment:**
   ```bash
   # Check application status
   python ops/health_check.py health

   # Check system resources
   htop

   # Review recent logs
   tail -f logs/main.log
   ```

2. **Recovery Steps:**
   ```bash
   # Attempt graceful restart
   docker compose restart

   # If restart fails, force recreate
   docker compose up --force-recreate

   # Check logs after restart
   docker logs --tail 50 gogooku3-app
   ```

3. **Escalation:**
   - If issue persists > 30 minutes: Notify on-call engineer
   - If data loss suspected: Initiate backup recovery procedure
   - If security incident suspected: Follow security incident response

### Data Loss (High Priority)

1. **Assess Impact:**
   ```bash
   # Check what data is missing
   ls -la data/
   ls -la output/

   # Check backup status
   ls -la backups/
   ```

2. **Recovery Process:**
   ```bash
   # Stop application to prevent corruption
   docker compose stop

   # Restore from backup
   tar -xzf backups/latest.tar.gz -C /

   # Verify data integrity
   python scripts/data/validate_dataset.py

   # Restart application
   docker compose start
   ```

3. **Post-Incident Review:**
   - Document root cause
   - Update backup procedures
   - Implement preventive measures

## 📈 Scaling Procedures

### Horizontal Scaling

1. **Add More Workers:**
   ```yaml
   # docker-compose.yml
   services:
     worker:
       image: gogooku3:latest
       deploy:
         replicas: 3
       environment:
         - WORKER_TYPE=training
   ```

2. **Load Balancer Configuration:**
   ```yaml
   # Add to docker-compose.yml
   lb:
     image: nginx:alpine
     ports:
       - "80:80"
     volumes:
       - ./nginx.conf:/etc/nginx/nginx.conf
   ```

### Vertical Scaling

1. **Increase Resources:**
   ```yaml
   # docker-compose.yml
   services:
     app:
       deploy:
         resources:
           limits:
             cpus: '4.0'
             memory: 16G
           reservations:
             cpus: '2.0'
             memory: 8G
   ```

2. **Optimize Configuration:**
   ```bash
   # Environment variables
   export PERF_POLARS_STREAM=1
   export PERF_MEMORY_OPTIMIZATION=1
   ```

## 💾 Backup & Recovery

### Automated Backup

1. **Database Backup:**
   ```bash
   # ClickHouse backup
   docker exec gogooku3-clickhouse clickhouse-client --query "BACKUP DATABASE gogooku3 TO Disk('backups', 'backup_$(date +%Y%m%d_%H%M%S)')"

   # Redis backup
   docker exec gogooku3-redis redis-cli SAVE
   ```

2. **File System Backup:**
   ```bash
   # Create backup archive
   tar -czf backups/data_$(date +%Y%m%d).tar.gz data/ output/ configs/

   # Clean old backups (keep last 7 days)
   find backups/ -name "data_*.tar.gz" -mtime +7 -delete
   ```

### Manual Recovery

1. **Complete Recovery:**
   ```bash
   # Stop services
   docker compose down

   # Restore data
   tar -xzf backups/latest.tar.gz -C /

   # Restore databases
   docker exec gogooku3-clickhouse clickhouse-client --query "RESTORE DATABASE gogooku3 FROM Disk('backups', 'latest_backup')"

   # Start services
   docker compose up -d
   ```

2. **Partial Recovery:**
   ```bash
   # Restore specific files
   tar -xzf backups/data_20231201.tar.gz data/specific_file.parquet

   # Verify integrity
   python scripts/data/validate_dataset.py
   ```

## 📊 Monitoring & Alerts

### Key Metrics to Monitor

1. **Application Metrics:**
   - Response times
   - Error rates
   - Throughput

2. **System Metrics:**
   - CPU usage
   - Memory usage
   - Disk I/O

3. **Business Metrics:**
   - Training completion rate
   - Model accuracy
   - Data processing volume

### Alert Response

1. **Warning Alerts:**
   - Review logs
   - Check system resources
   - Plan capacity upgrades

2. **Critical Alerts:**
   - Immediate investigation
   - Stakeholder notification
   - Emergency response team activation

3. **Info Alerts:**
   - Log for trend analysis
   - Plan preventive maintenance

## 🚨 障害対応手順

### 緊急停止・再起動手順

```bash
# 即時停止（緊急時）
docker compose down --timeout 30

# 安全停止（通常時）
docker compose down --timeout 300

# 強制再作成（設定変更時）
docker compose up -d --force-recreate

# ローリングアップデート（段階的）
docker compose up -d --scale app=2 --no-deps app

# ログ確認
tail -f logs/main.log
```

### サービス別障害対応

#### MinIO (Object Storage) 障害時

**症状確認:**
```bash
# MinIOステータス確認
docker ps | grep minio

# MinIOログ確認
docker logs gogooku3-minio | tail -50

# MinIOヘルスチェック
curl -f http://localhost:9000/minio/health/live
```

**回復手順:**
```bash
# MinIO再起動
docker compose restart minio

# MinIO設定確認
docker exec gogooku3-minio mc admin info local

# データ整合性チェック
docker exec gogooku3-minio mc ls gogooku/
```

#### ClickHouse (Database) 障害時

**症状確認:**
```bash
# ClickHouseステータス確認
docker ps | grep clickhouse

# ClickHouseログ確認
docker logs gogooku3-clickhouse | tail -50

# ClickHouse接続テスト
docker exec gogooku3-clickhouse clickhouse-client --query "SELECT 1"
```

**回復手順:**
```bash
# ClickHouse再起動
docker compose restart clickhouse

# データベース状態確認
docker exec gogooku3-clickhouse clickhouse-client --query "SHOW DATABASES"

# テーブル整合性チェック
docker exec gogooku3-clickhouse clickhouse-client --query "SHOW TABLES FROM gogooku3"
```

#### Redis (Cache) 障害時

**症状確認:**
```bash
# Redisステータス確認
docker ps | grep redis

# Redisログ確認
docker logs gogooku3-redis | tail -50

# Redis接続テスト
docker exec gogooku3-redis redis-cli ping
```

**回復手順:**
```bash
# Redis再起動
docker compose restart redis

# Redisデータ確認
docker exec gogooku3-redis redis-cli dbsize

# Redisパフォーマンス確認
docker exec gogooku3-redis redis-cli info stats
```

#### アプリケーション障害時

**症状確認:**
```bash
# プロセス確認
ps aux | grep python

# アプリケーションステータス
python ops/health_check.py health

# リソース使用状況
htop -p $(pgrep -f main.py)
```

**回復手順:**
```bash
# アプリケーション再起動
docker compose restart app

# メモリ解放（OOM時）
echo 1 > /proc/sys/vm/drop_caches

# ログ分析
grep -i error logs/main.log | tail -20
```

### データ破損・損失時の対応

#### データファイル破損時

```bash
# バックアップ確認
ls -la backups/

# データ整合性チェック
python -c "
import pandas as pd
try:
    df = pd.read_parquet('data/processed/dataset.parquet')
    print(f'Data integrity OK: {len(df)} rows')
except Exception as e:
    print(f'Data corruption detected: {e}')
"

# バックアップからのリストア
tar -xzf backups/data_$(date +%Y%m%d).tar.gz -C /
```

#### データベース破損時

```bash
# ClickHouseバックアップ確認
docker exec gogooku3-clickhouse clickhouse-client --query "SHOW BACKUPS"

# バックアップからのリストア
docker exec gogooku3-clickhouse clickhouse-client --query "
RESTORE DATABASE gogooku3 FROM Disk('backups', 'latest_backup')
"

# データ整合性検証
docker exec gogooku3-clickhouse clickhouse-client --query "
SELECT COUNT(*) FROM gogooku3.prices
"
```

### 容量逼迫時の対応

#### ディスク容量逼迫

**監視:**
```bash
# ディスク使用状況
df -h /

# 大きなファイル特定
find / -type f -size +100M -exec ls -lh {} \; | head -10

# ログファイルサイズ確認
du -sh logs/*.log
```

**対応:**
```bash
# ログローテーション実行
sudo logrotate -f /etc/logrotate.d/gogooku3

# 一時ファイル削除
find /tmp -name "*.tmp" -mtime +1 -delete

# Dockerイメージ整理
docker system prune -f

# 古いバックアップ削除（7日以上前）
find backups/ -name "*.tar.gz" -mtime +7 -delete
```

#### メモリ容量逼迫

**監視:**
```bash
# メモリ使用状況
free -h

# プロセス別メモリ使用
ps aux --sort=-%mem | head -10

# メモリ使用トレンド
python ops/health_check.py health | grep memory
```

**対応:**
```bash
# メモリ解放
echo 1 > /proc/sys/vm/drop_caches
echo 2 > /proc/sys/vm/drop_caches
echo 3 > /proc/sys/vm/drop_caches

# 大量メモリ使用プロセス特定
ps aux --sort=-%mem | head -5

# Dockerコンテナメモリ制限調整
docker compose up -d --scale app=1
```

#### CPU容量逼迫

**監視:**
```bash
# CPU使用状況
top -b -n 1 | head -10

# CPU使用トレンド
python ops/health_check.py health | grep cpu
```

**対応:**
```bash
# CPU使用プロセス特定
ps aux --sort=-%cpu | head -5

# 負荷軽減のための並列処理無効化
unset PERF_PARALLEL_PROCESSING

# Docker CPU制限調整
docker update --cpus 2 gogooku3-app
```

### ネットワーク障害時の対応

#### 外部API接続障害

```bash
# J-Quants API接続テスト
curl -I https://api.jquants.com/v1/

# DNS解決確認
nslookup api.jquants.com

# ネットワーク疎通確認
ping -c 3 api.jquants.com
```

**対応:**
```bash
# リトライ設定変更
export API_RETRY_ATTEMPTS=5
export API_RETRY_DELAY=60

# 代替エンドポイント設定（利用可能な場合）
export JQUANTS_BASE_URL=https://backup-api.jquants.com/v1/
```

### インシデント対応フロー

#### 1. 検知・評価（Detection & Assessment）

```bash
# インシデント検知
python ops/health_check.py health

# 影響範囲評価
docker compose ps
docker stats

# ログ分析
grep -i error logs/*.log | tail -20
```

#### 2. 対応・回復（Response & Recovery）

```bash
# 一次対応
docker compose restart

# 詳細調査
tail -f logs/main.log

# バックアップ確認
ls -la backups/
```

#### 3. 復旧確認（Verification）

```bash
# サービス正常性確認
python ops/health_check.py health

# 機能テスト
python main.py safe-training --mode quick

# 監視ダッシュボード確認
curl http://localhost:8000/metrics
```

#### 4. 原因分析・対策（Analysis & Prevention）

```bash
# 根本原因分析
# 1. ログ分析
# 2. メトリクス分析
# 3. 設定変更履歴確認
# 4. 依存関係更新確認

# 対策実施
# 1. 設定最適化
# 2. 監視強化
# 3. バックアップ改善
# 4. ドキュメント更新
```

## 📞 Contact Information

### On-Call Schedule
- Primary: [Engineer Name] - [Phone/Slack]
- Secondary: [Engineer Name] - [Phone/Slack]
- Escalation: [Manager Name] - [Phone/Slack]

### External Resources
- Documentation: [Internal Wiki]
- Monitoring Dashboard: [Grafana URL]
- Incident Management: [Tool URL]

---

**Last Updated:** $(date +%Y-%m-%d)
**Version:** 1.0
**Review Cycle:** Monthly
