# A.4 安全版実装完了レポート

**Date**: 2025-11-03
**Status**: ✅ **実装完了 (Implementation Complete)**
**対応issue**: A.3/A.4過剰中立化によるSharpe 23.6%低下

---

## 📋 実装サマリー

過去のA.3/A.4実装（全中立化 γ=1.0, Ridge α=0.1）がSharpe比率を23.6%低下させる問題に対し、**部分中立化 + Ridge強化 + 再スケール + 安全ガード**を備えたA.4安全版を実装しました。

**主な変更**:
1. **部分中立化**: γ=0.3 で 30% のみファクター除去（100% → 30%）
2. **Ridge正則化強化**: α=0.1 → α=10.0 で過学習防止
3. **分布保持**: 再中心化 + 再スケールで元のスコア分布維持
4. **安全ガード**: R²チェック、t統計量チェック、補正クリップ (0.25σ)
5. **デフォルト変更**: Sector + Volatility のみ（Beta/Size は除外）

---

## 🎯 実装内容

### 1. `risk_neutralize()` 関数の全面刷新

**ファイル**: `apex-ranker/apex_ranker/backtest/enhanced_inference.py` (lines 214-361)

**新しい関数シグネチャ**:
```python
def risk_neutralize(
    scores: NDArray[np.float32],
    df_features: pd.DataFrame,
    factors: list[str] | None = None,
    alpha: float = 10.0,    # 旧: 0.1
    gamma: float = 0.3,     # NEW: 部分中立化パラメータ
) -> NDArray[np.float32]:
```

**6ステップ安全アルゴリズム**:
```python
# Step 1: Z-score正規化 (X と y 両方)
y_normalized = (y - mean(y)) / std(y)
X_normalized = (X - mean(X)) / std(X)

# Step 2: Ridge回帰 (α=10で強正則化)
β = argmin ||y - Xβ||² + α||β||²

# Step 3: 部分中立化 (30%のみ除去)
y_resid = y_normalized - γ·(Xβ)  # γ=0.3

# Step 4: 再中心化
y_resid ← y_resid - mean(y_resid)

# Step 5: 再スケール (元のstd維持)
y_resid ← y_resid · (std(y) / std(y_resid))

# Step 6: 安全ガード
- R² < 0.05 → スキップ (モデル無効)
- max(|t(β)|) < 2 → スキップ (係数有意でない)
- ||y - y_resid||∞ > 0.25·std(y) → クリップ (過剰補正防止)
```

**デフォルトファクター変更**:
```python
# 旧: ["beta_60d", "log_mktcap", "Sector33Code"]
# 新: ["Sector33Code", "volatility_60d"]
```

### 2. CLI引数の追加

**ファイル**: `apex-ranker/scripts/backtest_smoke_test.py`

**新規追加された引数**:
```bash
--ei-neutralize-gamma FLOAT       # 部分中立化係数 (デフォルト: 0.3, 範囲: [0.2, 0.5])
--ei-ridge-alpha FLOAT             # Ridge正則化係数 (デフォルト: 10.0)
--ei-risk-factors STR              # 中立化ファクター (デフォルト: Sector33Code,volatility_60d)
```

**既存引数の変更**:
```bash
--ei-risk-factors のデフォルト値変更:
  旧: "beta_60d,log_mktcap,Sector33Code"
  新: "Sector33Code,volatility_60d"
```

### 3. 関数シグネチャの更新

**ファイル**: `apex-ranker/scripts/backtest_smoke_test.py`

**run_backtest_smoke_test() 関数** (lines 148-149):
```python
def run_backtest_smoke_test(
    # ... 既存パラメータ ...
    ei_risk_factors: list[str] | None = None,
    ei_neutralize_gamma: float = 0.3,   # NEW
    ei_ridge_alpha: float = 10.0,       # NEW
    # ... 以下略 ...
) -> dict:
```

**main() 関数の呼び出し** (lines 1319-1320):
```python
run_backtest_smoke_test(
    # ... 既存引数 ...
    ei_risk_factors=(
        args.ei_risk_factors.split(",") if args.ei_risk_factors else None
    ),
    ei_neutralize_gamma=args.ei_neutralize_gamma,  # NEW
    ei_ridge_alpha=args.ei_ridge_alpha,            # NEW
    # ... 以下略 ...
)
```

### 4. risk_neutralize() 呼び出しの更新

**ファイル**: `apex-ranker/scripts/backtest_smoke_test.py` (lines 435, 459-460)

```python
# デフォルトファクター変更
default_factors = ["Sector33Code", "volatility_60d"]  # 旧: ["beta_60d", "log_mktcap", "Sector33Code"]

# 関数呼び出し更新
scores_neutralized = risk_neutralize(
    scores_tensor.numpy(),
    df_risk_pd,
    factors=available_factors,
    alpha=ei_ridge_alpha,           # 旧: alpha=0.1 (ハードコード)
    gamma=ei_neutralize_gamma,      # NEW
)
```

---

## 🧪 検証方法

### ステップ1: モック予測で構文チェック (即座に実行可能)

```bash
# 5日間のモック予測で安全版A.4をテスト
python apex-ranker/scripts/backtest_smoke_test.py \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2025-09-01 \
  --end-date 2025-09-05 \
  --top-k 10 \
  --use-mock-predictions \
  --use-enhanced-inference \
  --ei-neutralize-risk \
  --ei-neutralize-gamma 0.3 \
  --ei-ridge-alpha 10.0 \
  --output /tmp/a4_safe_test.json

# 期待される出力:
# - エラー無く完了
# - risk_neutralize() が呼ばれる
# - gamma=0.3, alpha=10.0 で動作確認
```

### ステップ2: 実験マトリクス実行 (ユーザー指定の4実験)

#### 実験1: BASE (A.3/A.4 OFF) - **完了済み** ✅
```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2024-01-01 --end-date 2025-10-31 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --features-mode fill-zero \
  --output results/bt_enhanced_monthly_h20_BASE.json

# 結果 (既存):
# Sharpe: 1.439, Return: 44.85%, MaxDD: 16.40%
```

#### 実験2: A.3のみ (Hysteresis Selection)
```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2024-01-01 --end-date 2025-10-31 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 \
  --ei-hysteresis-exit-k 60 \
  --features-mode fill-zero \
  --output results/bt_enhanced_monthly_h20_A3_ONLY.json
```

#### 実験3: A.4のみ (Risk Neutralization) - γ ∈ {0.2, 0.3, 0.5}
```bash
# γ=0.2 (弱中立化)
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2024-01-01 --end-date 2025-10-31 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference \
  --ei-neutralize-risk \
  --ei-neutralize-gamma 0.2 \
  --ei-ridge-alpha 10.0 \
  --features-mode fill-zero \
  --output results/bt_enhanced_monthly_h20_A4_gamma02.json

# γ=0.3 (標準中立化) - **推奨**
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2024-01-01 --end-date 2025-10-31 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference \
  --ei-neutralize-risk \
  --ei-neutralize-gamma 0.3 \
  --ei-ridge-alpha 10.0 \
  --features-mode fill-zero \
  --output results/bt_enhanced_monthly_h20_A4_gamma03.json

# γ=0.5 (強中立化)
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2024-01-01 --end-date 2025-10-31 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference \
  --ei-neutralize-risk \
  --ei-neutralize-gamma 0.5 \
  --ei-ridge-alpha 10.0 \
  --features-mode fill-zero \
  --output results/bt_enhanced_monthly_h20_A4_gamma05.json
```

#### 実験4: A.3 + A.4 (Combined) - 最良γを使用
```bash
# 実験3で最良のγを選択後実行
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2024-01-01 --end-date 2025-10-31 \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 \
  --ei-hysteresis-exit-k 60 \
  --ei-neutralize-risk \
  --ei-neutralize-gamma 0.3 \
  --ei-ridge-alpha 10.0 \
  --features-mode fill-zero \
  --output results/bt_enhanced_monthly_h20_A3A4_SAFE.json
```

---

## 📊 期待される結果

### GO条件 (ユーザー指定)
- ✅ **Sharpe比率**: +5〜10% (1.439 → 1.51〜1.58)
- ✅ **Turnover**: ▲5〜20% 削減
- ✅ **MaxDD**: +5pp以内 (16.40% → 21.40%以内)

### 比較対象
| 実験 | Sharpe | Return | MaxDD | 判定 |
|------|--------|--------|-------|------|
| **BASE** | 1.439 | 44.85% | 16.40% | ✅ ベースライン |
| **A.3/A.4 (旧)** | 1.100 | 31.74% | 17.13% | ❌ -23.6% Sharpe低下 |
| **A.4 (安全版)** | TBD | TBD | TBD | 🔄 検証待ち |

---

## 🔧 技術的詳細

### 旧A.4の問題点
```python
# 問題1: 全中立化 (暗黙的 γ=1.0)
residual_scores = scores - reg.predict(X)  # 100%除去

# 問題2: Ridge弱すぎ (α=0.1)
reg = Ridge(alpha=0.1)  # 過学習リスク

# 問題3: 分布非保持
return residual_scores  # std圧縮, mean乖離

# 問題4: 安全ガード無し
# R², t統計量チェック無し
# 過剰補正クリップ無し
```

### 新A.4の解決策
```python
# 解決1: 部分中立化 (γ=0.3)
correction = gamma * y_pred  # 30%のみ除去
y_resid = y_normalized - correction

# 解決2: Ridge強化 (α=10)
reg = Ridge(alpha=10, fit_intercept=False)

# 解決3: 分布保持
y_resid = (y_resid - mean(y_resid))  # 再中心化
y_resid = y_resid * (std(y) / std(y_resid))  # 再スケール
y_resid = y_resid + y_mean  # 元mean復元

# 解決4: 3段安全ガード
if r2 < 0.05: return scores         # R²チェック
if max(t_stats) < 2.0: return scores  # t統計量チェック
if max(|correction|) > 0.25*std(y):   # 補正クリップ
    clip correction to ±0.25*std(y)
```

---

## 📁 変更ファイル一覧

| ファイル | 変更内容 | 行数 |
|---------|---------|------|
| **apex-ranker/apex_ranker/backtest/enhanced_inference.py** | risk_neutralize() 全面刷新 | 214-361 |
| **apex-ranker/scripts/backtest_smoke_test.py** | CLI引数追加 (--ei-neutralize-gamma, --ei-ridge-alpha) | 1178-1189 |
| **apex-ranker/scripts/backtest_smoke_test.py** | デフォルトファクター変更 | 435 |
| **apex-ranker/scripts/backtest_smoke_test.py** | risk_neutralize() 呼び出し更新 | 459-460 |
| **apex-ranker/scripts/backtest_smoke_test.py** | run_backtest_smoke_test() シグネチャ更新 | 148-149 |
| **apex-ranker/scripts/backtest_smoke_test.py** | main() 呼び出し更新 | 1319-1320 |

---

## ✅ 完了した作業

- [x] risk_neutralize() 関数の6ステップアルゴリズム実装
- [x] 部分中立化パラメータ gamma 追加 (デフォルト: 0.3)
- [x] Ridge正則化強化 alpha 変更 (0.1 → 10.0)
- [x] 再中心化 + 再スケール実装
- [x] 3段安全ガード実装 (R², t統計量, クリップ)
- [x] デフォルトファクター変更 (beta/size → sector/vol)
- [x] CLI引数追加 (--ei-neutralize-gamma, --ei-ridge-alpha)
- [x] 関数シグネチャ更新 (backtest script)
- [x] 構文チェック完了 (Python compile check passed)

---

## 📋 次のステップ

### 即座に実行可能
```bash
# ステップ1: モック予測でA.4安全版テスト (5分)
python apex-ranker/scripts/backtest_smoke_test.py \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --start-date 2025-09-01 --end-date 2025-09-05 \
  --top-k 10 --use-mock-predictions \
  --use-enhanced-inference --ei-neutralize-risk \
  --ei-neutralize-gamma 0.3 --ei-ridge-alpha 10.0 \
  --output /tmp/a4_safe_test.json
```

### 本格検証 (ユーザー判断)
```bash
# ステップ2: 4実験マトリクス実行 (各1-2時間)
# 1. BASE (完了済み)
# 2. A.3のみ
# 3. A.4 (γ=0.2/0.3/0.5)
# 4. A.3+A.4 (最良γ)

# ステップ3: 結果比較
python scripts/compare_backtest_results.py \
  results/bt_enhanced_monthly_h20_BASE.json \
  results/bt_enhanced_monthly_h20_A4_gamma*.json \
  --output results/a4_safe_comparison.md
```

---

## 🎓 実装の要点

1. **部分中立化 (γ=0.3)**: 過剰補正を防ぐため、30%のみファクター除去
2. **Ridge強化 (α=10)**: 過学習を防ぐため正則化を100倍に強化
3. **分布保持**: 再中心化 + 再スケールで元のスコア分布を維持
4. **3段安全ガード**: R²/t統計量チェック + 補正クリップで過剰補正防止
5. **デフォルト変更**: Sector + Volatility のみ (Beta/Size は高リスクのため除外)

---

## 🔗 関連ドキュメント

- ユーザー指定仕様: [前回の会話サマリー](セッション要約参照)
- 旧A.4の問題分析: [P0-1実験結果](前回の分析レポート)
- APEX-Ranker概要: `apex-ranker/README.md`
- 実験ステータス: `apex-ranker/EXPERIMENT_STATUS.md`

---

**実装者**: Claude (AI Assistant)
**レビュー推奨**: A.4安全版の数学的妥当性、ハイパーパラメータ選択の根拠
