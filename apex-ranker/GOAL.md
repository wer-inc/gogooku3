# Apex Ranker Goal (VVMD)

**結論**：はい、Apex Ranker の正式ゴールとして採択することを強く推奨します。  
理由はシンプルで、(1) **短期で上がる銘柄を当てる**という事業目的、(2) **可微分な学習目的**、(3) **実運用での採用判定**が一気通貫で整合するためです。

---

## 1. GOAL（North Star と代理目的）

### North Star
次の **1–5 営業日**における、**コスト込み Top‑K ポートフォリオの期待リターン最大化**（VVMD＝需給/出来高/モメンタム/需要の手掛かりを活用）。

### 学習時の代理目的（可微分）
- 主：**Soft NDCG@K（Top‑K 重視のリストワイズ）**
- 補：**Top‑K 重み付き Pairwise（RankNet/LambdaRank 系）**（境界の入れ替えに勾配集中）
- 補：**キャリブレーション/MSE**（確率的整合性の維持）
- 正則化：**ターンオーバー**（`λ_to · mean(|rank_t − rank_{t−1}|)`）＋任意で **コスト/容量近似**
- 重み付け：**時間減衰**（τ ≈ 90–180、最近重視）

> CompositeLoss 例  
> `L = λ_ndcg · L_ndcg@K + λ_pair · L_pair(top-weighted) + λ_mse · L_calib + λ_to · L_turnover (+ L_port)`

---

## 2. 評価（採用判定に使う KPI と手順）

### オフライン（ランキングの質：1d/5d 中心）
- **ΔP@K_pos**：上位 K に「正のリターン」が入る率 − ランダム期待値
- **ΔNDCG@K**：`gain = softplus(α · r)`、ランダム差分で解釈
- **Top‑K Overlap**、**Top‑Bottom Spread**（% または σ）
- **RankIC（符号監視）**：負なら符号反転モードを検討

### プロトコル
- 分割：**Purged K‑Fold + Embargo**（実装済み）
- 有意性：**Diebold–Mariano（DM）**、**moving‑block bootstrap 95% CI**
- **採用条件（例）**
  - `h5_ΔP@K_pos` の **DM > +1.96** かつ **95% CI(平均差) > 0**
  - `h5_spread` も同方向・有意

### ポートフォリオ（実運用近似）
- **コスト込み Sharpe/Sortino/Calmar/MaxDD**（HAC/ブートストラップで CI）
- **Turnover / 平均保有日数 / fallback 率**
- **曝露**：β・セクター・サイズの事後回帰を日次監視
- **採用条件（例）**：週次リバランスで Sharpe が **旧版比 −10% 以内**か **上回る**

### 運用ガード
- 全リバランス日で **candidate_count > 0**、**k_over_n ≈ 設定 k_ratio**、**ゼロ選定率 0%**
- **fallback 率 < 20%** を目安
- **曝露バンド**を逸脱したら失格

---

## 3. OKR（採用時に見る “数字の着地点” 例）

**Objective**：短期 Top‑K の質と実運用 Sharpe をともに引き上げる。

**Key Results（例）**
- KR1：`h5_ΔP@K_pos` **+0.015 以上**（DM > 1.96 & CI > 0）
- KR2：`h5_spread` **+0.08σ 以上**（CI > 0）
- KR3：週次 **Sharpe ≥ 旧版 −10%**、**Turnover 旧版 ±15% 内**
- KR4：**ゼロ選定 0%**、**fallback 率 < 20%**、曝露は上限内

---

## 4. 推奨ハイパラ初期値（YAML）

```yaml
eval:
  k_ratio: 0.07              # 短期は 0.05–0.07 を推奨
  ndcg_beta: 1.0
  es_weights: "0.6,0.3,0.1"  # 5d,10d,20d（可能なら ΔP@K_pos ベース）
train:
  time_decay_tau_days: 120
  use_ema: true
  ema_beta: 0.999
loss:
  lambda_ndcg: 1.0
  lambda_pair: 0.5
  lambda_mse: 0.1
  lambda_turnover: 1e-3
gate:
  mode: percentile
  k_min: 5
  sign:
    h5: +1
    h10: +1   # 負相関なら -1 に切替
    h20: +1
```

---

## 5. リポジトリへの反映案（そのまま PR できる構成）

1. **`GOAL.md` を追加**（このファイル）
2. **`configs/v0_base.yaml`** を上記キーで更新（`v0_base_baseline.yaml` は固定線表）
3. **`README.md`** に Goal → Metrics → Promotion の導線を追記
4. **CI/テスト**
   - `tests/.../test_selection.py`（ゼロ選定ガード）
   - `tests/.../test_metrics_random_baselines.py`（Δの基礎性質）
   - `tests/.../test_purged_kfold_splitter.py`（リーケージ不在）

---

## 6. 採用/却下のフロー（運用を止めないために）

- **Production**：旧版（LKG）を常設。研究モデルは必ず **A/B + DM/CI 合格後に昇格**。
- **失敗時の即応**：`tau (90–180) / k_ratio (0.05–0.07) / ema_beta (0.995–0.9995)` の 3 点をまず調整。
- **長期ヘッド** が負相関なら、**符号反転**モードを正式サポート（`gate.sign.h10/h20 = -1`）。

---

## まとめ

- 今回定義した **“短期 Top‑K × Δ 指標 × 運用ガード”** は、**目的 ⇄ 学習 ⇄ 評価 ⇄ 運用** のズレを最小にする。
- そのまま `GOAL.md` と `v0_base.yaml` を更新して **Apex Ranker の公式ゴール**とする。
- 既に用意済みの Purged K-Fold、Δ 指標、百分位ゲート、EMA スナップショット、A/B & DM スクリプトと完全互換。

この方針で進めれば、短期での実運用 Sharpe をブーストしつつ、検証の信頼性と再現性も保てます。
