# APEX-Ranker 現在の状況レポート
**作成日時**: 2025-11-03 08:59 UTC
**対象期間**: P0-1A/1B実行結果と現在の実装状況

---

## 📊 エグゼクティブサマリー

### 現在の状態
- ✅ **P0-1A（BASE）**: 実行完了（Sharpe 1.439）
- ✅ **P0-1B（A3+A4）**: 実行完了（Sharpe 1.100、**▲23.6%悪化**）
- ⚠️ **A.3/A.4実装**: 基本版のみ（改善版は未実装）
- 🔄 **ATFTトレーニング**: 2プロセス実行中（1.5-1.6時間経過）

### 重要な発見
**ユーザーの分析は正確です** - A.3/A.4は現状では**逆効果**。主な原因：
1. ✅ **完全中立化（gamma=1.0）** - 分布を過度に圧縮
2. ✅ **再スケールなし** - スコア分布が負側にシフト
3. ✅ **安全ガードなし** - 極端な補正を防げない

---

## 📈 P0-1A/1B バックテスト結果（詳細）

### 比較表

| 指標 | BASE（A3/A4 OFF） | A3A4（ON） | 変化 | 評価 |
|------|------------------|-----------|------|------|
| **Sharpe Ratio** | **1.439** | 1.100 | **▲23.6%** | ❌ 大幅悪化 |
| **Total Return** | +44.85% | +31.74% | ▲13.11pp | ❌ 悪化 |
| **Max Drawdown** | 16.40% | 17.13% | +0.73pp | ❌ 悪化 |
| **Sortino** | 1.810 | 1.382 | ▲23.6% | ❌ 悪化 |
| **Calmar** | 1.434 | 0.993 | ▲30.8% | ❌ 大幅悪化 |
| **Avg Turnover** | 1.39% | 1.44% | +0.05pp | ⚠️ 横ばい |
| **Total Trades** | 3,314 | 2,975 | ▲339 | ⚠️ 減少 |
| **Costs** | ¥0.37M | ¥0.34M | ▲¥0.03M | ✅ 微改善 |

### 共通設定
- **期間**: 2024-01-01 〜 2025-10-31（442取引日、22リバランス）
- **リバランス**: 月次
- **予測ホライズン**: 20日
- **Top-K**: 35銘柄

### 結論
**Go判定は不可** - 本番は**BASE（A.3/A.4 OFF）**を維持すべき

---

## 🔍 悪化の根本原因分析

### 1. 現在の実装の問題点

#### A.4（リスク中立化）の実装（`enhanced_inference.py:214-289`）

**現在のコード**:
```python
def risk_neutralize(
    scores: NDArray[np.float32],
    df_features: pd.DataFrame,
    factors: list[str] | None = None,
    alpha: float = 0.1,  # 固定値
) -> NDArray[np.float32]:
    # ...
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X_normalized, scores)

    # ❌ 完全中立化（gamma=1.0固定）
    residual_scores = scores - reg.predict(X_normalized)
    return residual_scores.astype(np.float32)
```

**問題点**:
1. ❌ **gamma=1.0固定** - 完全中立化（取りすぎ）
2. ❌ **再中心化なし** - ゼロ平均に戻していない
3. ❌ **再スケールなし** - 分布圧縮を元に戻していない
4. ❌ **安全ガードなし** - 極端な補正をクリップしていない

**結果**:
- スコア分布が**負側にシフト**（threshold ≈ -0.27 〜 -0.35）
- 分布が**過度に圧縮**（std減少）
- 選定ゲートが常時マイナス域に張り付く

### 2. 供給ガードのバイパス

**ログから観察**:
- `candidate_kept=35` - 候補が目標銘柄数に縮退
- `fallback=0` - フォールバックなし
- 本来 **k_min ≥ 53** を守る設計が機能していない

**影響**:
- 分散低下 → ドローダウン耐性悪化
- Sharpe悪化

### 3. ヒステリシスの効果減少

**原因**:
- 分布圧縮＋負側シフトにより、Entry/Exit閾値の**改善余地が消失**
- ランク序列は維持されても、スコアレンジが狭すぎて意図した「無駄な入替抑制」が効かない

---

## 🛠️ 現在の実装状況

### 実装済み（基本版）

| 機能 | ファイル | 状態 | 備考 |
|------|---------|------|------|
| **A.1 Rank Ensemble** | `enhanced_inference.py:26-80` | ✅ 実装済み | fold平均化、正規化ランク |
| **A.2 Uncertainty Filter** | `enhanced_inference.py:83-141` | ✅ 実装済み | rank分散でフィルタ |
| **A.3 Hysteresis** | `enhanced_inference.py:143-211` | ✅ 実装済み | Entry/Exit閾値 |
| **A.4 Risk Neutralize** | `enhanced_inference.py:214-289` | ⚠️ **基本版のみ** | 改善版は未実装 |

### 未実装（改善版）

| 機能 | 必要性 | 優先度 |
|------|--------|--------|
| **部分中立化（gamma）** | ✅ 必須 | P0 |
| **再中心化** | ✅ 必須 | P0 |
| **再スケール** | ✅ 必須 | P0 |
| **安全ガード（クリップ）** | ✅ 必須 | P0 |
| **gamma/alphaパラメータ化** | ✅ 必須 | P0 |
| **Beta/Size列の前処理** | ⚠️ 推奨 | P1 |

---

## 🚀 推奨される改善策（優先順位順）

### P0: 即座に実装（今日中、2-3時間）

#### 1. A.4の改善版実装

**必要な変更**:

```python
def risk_neutralize(
    scores: NDArray[np.float32],
    df_features: pd.DataFrame,
    factors: list[str] | None = None,
    alpha: float = 10.0,      # NEW: Ridge正則化（デフォルト10）
    gamma: float = 0.3,       # NEW: 部分中立化係数（0.2-0.5）
    rescale: bool = True,     # NEW: 再スケール有効化
) -> NDArray[np.float32]:
    """A.4: Risk Neutralization - Improved version with partial neutralization."""

    # 1. 標準化
    X_normalized = standardize_features(X)
    y_standardized = (scores - scores.mean()) / (scores.std() + 1e-9)

    # 2. Ridge回帰
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X_normalized, y_standardized)

    # 3. 部分中立化（gamma < 1.0）
    y_resid = y_standardized - gamma * reg.predict(X_normalized)

    # 4. 再中心化（ゼロ平均）
    y_resid = y_resid - y_resid.mean()

    # 5. 再スケール（元のstdに戻す）
    if rescale:
        current_std = y_resid.std() + 1e-9
        target_std = scores.std()
        y_resid = y_resid * (target_std / current_std)

    # 6. 安全ガード（最大補正をクリップ）
    max_correction = 0.25 * scores.std()
    correction = scores - y_resid
    correction = np.clip(correction, -max_correction, max_correction)
    y_resid = scores - correction

    # 7. 元のスケールに戻す
    y_resid = y_resid * scores.std() + scores.mean()

    return y_resid.astype(np.float32)
```

**期待効果**:
- 負側シフトの防止 ✅
- 分布圧縮の回避 ✅
- 極端な補正の抑制 ✅

#### 2. コマンドライン引数の追加

**必要な追加**（`backtest_smoke_test.py`）:
```python
parser.add_argument("--ei-neutralize-gamma", type=float, default=0.3,
                    help="Partial neutralization coefficient (0.2-0.5)")
parser.add_argument("--ei-ridge-alpha", type=float, default=10.0,
                    help="Ridge regression regularization")
parser.add_argument("--ei-rescale", action="store_true", default=True,
                    help="Re-scale residuals to original std")
```

#### 3. 供給ガードの強化

**実装箇所**: 選定後の再チェック
```python
# A.4適用後、供給ガードの強制チェック
if candidate_kept < max(ceil(k_ratio * N), k_min):
    fallback = True
    candidate_kept = k_min
    logger.warning(f"Supply guard triggered: {candidate_kept} < {k_min}")
```

### P0.5: 速攻AB実験（半日）

**実行コマンド**:

```bash
# 1. BASE（確認）
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --output results/p0_final/BASE.json

# 2. A.3のみ
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 --ei-hysteresis-exit-k 60 \
  --output results/p0_final/A3_only.json

# 3. A.4のみ（改善版、gamma=0.3）
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference --ei-neutralize-risk \
  --ei-neutralize-exposures sector,vol \
  --ei-neutralize-gamma 0.3 --ei-ridge-alpha 10 \
  --output results/p0_final/A4_g03.json

# 4. A.3 + A.4（改善版）
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 --ei-hysteresis-exit-k 60 \
  --ei-neutralize-risk --ei-neutralize-exposures sector,vol \
  --ei-neutralize-gamma 0.3 --ei-ridge-alpha 10 \
  --output results/p0_final/A3A4_g03.json

# 5. gamma=0.2（保守的）
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 --ei-hysteresis-exit-k 60 \
  --ei-neutralize-risk --ei-neutralize-exposures sector,vol \
  --ei-neutralize-gamma 0.2 --ei-ridge-alpha 10 \
  --output results/p0_final/A3A4_g02.json

# 6. gamma=0.5（積極的）
python apex-ranker/scripts/backtest_smoke_test.py \
  --model models/apex_ranker_v0_enhanced.pt \
  --config apex-ranker/configs/v0_base_89_cleanADV.yaml \
  --data output/ml_dataset_latest_clean_with_adv.parquet \
  --rebalance-freq monthly --horizon 20 --top-k 35 \
  --use-enhanced-inference \
  --ei-hysteresis-entry-k 35 --ei-hysteresis-exit-k 60 \
  --ei-neutralize-risk --ei-neutralize-exposures sector,vol \
  --ei-neutralize-gamma 0.5 --ei-ridge-alpha 10 \
  --output results/p0_final/A3A4_g05.json
```

**合否判定基準**:
- ✅ Sharpe: +5〜10%以上（BASE比）
- ✅ Turnover: ▲5〜20%減少
- ✅ MaxDD: +5pp以内の悪化
- ✅ 供給: `selected_count ≥ 53`、`fallback ≤ 20%`

### P1: 今週中の恒久策（2-3日）

1. **Beta/Size列の前処理実装**
   - `beta_60d`: 60日回帰係数
   - `ln_size`: log(price × shares_outstanding)

2. **Feature-ABI恒久対策**
   - チェックポイントにメタデータ保存

3. **供給メタの常時記録**
   - `selected_count / eligible_universe / fallback_used / k_over_n / threshold`

4. **日次ガードレール**
   - A.4適用後のTop-K overlap < 50% → その日はスキップ
   - std縮小率 > 30% → スキップ

---

## 🔄 現在実行中のプロセス

### ATFTトレーニング（メインプロジェクト）

| PID | CPU | メモリ | 経過時間 | 状態 |
|-----|-----|--------|---------|------|
| 1707961 | 49.9% | 0.1% | 01:37:43 | ✅ 実行中 |
| 1712539 | 33.1% | 0.3% | 01:12:53 | ✅ 実行中 |

**コマンド**: `python scripts/train_atft.py`
- データ: `output/atft_data`
- バッチサイズ: 1024
- エポック: 3
- 学習率: 0.0002
- ワーカー: 8

**ステータス**: 正常実行中（多スレッドDataLoaderで安定動作）

---

## 📋 次のステップ（タイムライン）

### 今日中（2-3時間）
1. ✅ A.4改善版の実装（部分中立化+再スケール+ガード）
2. ✅ コマンドライン引数の追加
3. ✅ 供給ガードの強化

### 今日中〜明日（半日）
4. ✅ AB実験6本実行（BASE, A.3, A.4×3, A.3+A.4×3）
5. ✅ 結果比較とgamma最適値の決定

### 今週中（2-3日）
6. ✅ Beta/Size列の実装
7. ✅ Feature-ABI恒久対策
8. ✅ 供給メタの常時記録
9. ✅ 日次ガードレールの実装

---

## 📊 期待される改善効果

### 現実的レンジ（改善版A.4導入後）

| Phase | 施策 | 期待Sharpe改善 | 累積Sharpe |
|-------|------|--------------|-----------|
| Baseline | BASE（現状維持） | - | 1.439 |
| P0 | A.3+A.4（改善版、gamma=0.3） | +5〜10% | 1.51〜1.58 |
| P0.5 | gamma最適化（0.2〜0.5） | +追加3〜5% | 1.55〜1.66 |
| P1 | Beta/Size追加 | +3〜5% | 1.60〜1.74 |
| P1+ | A.1 Ensemble + A.2 Filter | +5〜10% | 1.68〜1.91 |

**目標**: Sharpe 1.7+（現状比 +18%）

---

## ⚠️ 重要な注意事項

### 本番運用

**現時点では**:
- ✅ **本番はBASE（A.3/A.4 OFF）を維持**
- ❌ 改善版実装・検証が完了するまでA.3/A.4は使用しない

### データ依存性

現在使用可能な中立化軸:
- ✅ **Sector**: `sec_ret_1d_eq`（セクター等加重リターン）
- ✅ **Volatility**: `yz_vol_20`（20日ボラティリティ）
- ⚠️ **Beta**: 列なし（計算必要）
- ⚠️ **Size**: 列なし（近似計算必要）

**暫定策**: Sector+Volのみで開始、Beta/Sizeは後段追加

---

**レポート作成日時**: 2025-11-03 08:59 UTC
**作成者**: Claude Code (Autonomous Mode)
**ステータス**: P0改善版実装準備完了
