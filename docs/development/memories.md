# Gogooku3 統一ログ設定
# Hydra統合・JSON Lines・JST時刻対応

defaults:
  - _self_

# Hydra run/job設定
hydra:
  job:
    # 実行ディレクトリの変更無効化
    chdir: false
  run:
    # ログディレクトリ: _logs/dev/app/YYYY/MM/DD/
    dir: _logs/dev/app/${now:%Y/%m/%d}
  
  # ログ設定
  job_logging:
    # Hydraジョブログ設定
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
        datefmt: '%Y-%m-%d %H:%M:%S'
      json:
        format: '{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s","mod":"%(name)s"}'
        datefmt: '%Y-%m-%dT%H:%M:%S+09:00'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
      file:
        class: logging.handlers.RotatingFileHandler
        formatter: json
        filename: ${hydra:runtime.output_dir}/hydra.jsonl
        maxBytes: 52428800  # 50MB
        backupCount: 5
        encoding: utf-8
    root:
      level: INFO
      handlers: [console, file]
    disable_existing_loggers: false

  hydra_logging:
    # Hydra自体のログ設定
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][HYDRA] %(message)s'
        datefmt: '%Y-%m-%d %H:%M:%S'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
      file:
        class: logging.FileHandler
        formatter: simple
        filename: ${hydra:runtime.output_dir}/hydra.log
        encoding: utf-8
    root:
      level: INFO
      handlers: [console, file]
    disable_existing_loggers: false

# Gogooku3アプリケーションログ設定
gogooku_logging:
  # 基本設定
  level: INFO
  service: app
  timezone: "Asia/Tokyo"
  
  # ファイル出力
  file:
    enabled: true
    format: jsonl
    max_bytes: 52428800    # 50MB
    backup_count: 5
    encoding: utf-8
    
  # コンソール出力
  console:
    enabled: true
    format: simple
    
  # 環境別設定
  environments:
    dev:
      level: DEBUG
      console:
        enabled: true
      file:
        retention_days: 14
    prd:
      level: INFO
      console:
        enabled: false  # 本番はファイルのみ
      file:
        retention_days: 90
    
  # サービス別設定
  services:
    training:
      extra_fields:
        component: "ml"
        category: "training"
    dagster:
      extra_fields:
        component: "orchestration" 
        category: "workflow"
    feast:
      extra_fields:
        component: "feature_store"
        category: "serving"
        
  # 特殊設定
  filters:
    # ノイズの多いログの除外
    ignore_patterns:
      - ".*connectionpool.*"
      - ".*urllib3.*"
      - ".*requests.*Starting new HTTP.*"
    
    # 機密情報のマスク
    mask_fields:
      - "password"
      - "token" 
      - "api_key"
      - "secret"