# 🚀 gogooku3 特徴量エンジニアリング【完全実装プロンプト】

## 📋 目的
以下の特徴量提案を**漏れなく・確実に・効率的に**実装し、目標性能（IC 0.25+, Sharpe 2.0+）を達成する。

---

## ⚠️ 【絶対厳守】実装時の鉄則

### 🎯 完全実装の原則
```
1. 提案された特徴量を100%実装する（選択的実装は禁止）
2. 実装漏れを防ぐため、各セクションごとにチェックリストを作成・確認する
3. 「後で実装」「スキップ」は認めない - 全て今すぐ実装する
4. 不明点があっても、提案内容から推測して必ず実装する
5. 質問は一切しない - 自己完結的に実装を進める
```

### 📊 実装検証の必須事項
```
1. 各特徴量追加後、必ずバックテストで効果検証
2. 特徴量重要度を計算して提案通りの効果が出ているか確認
3. ICの変化を記録（+0.01未満でも記録）
4. 実装前後の性能比較レポートを自動生成
5. 全特徴量の実装完了チェックリストを出力
```

### 🔥 高速実装のための最適化
```
1. 並列処理を最大限活用（Polars, multiprocessing）
2. 計算結果をキャッシュして重複計算を避ける
3. ベクトル化演算を使い、forループを排除
4. メモリ効率を考慮し、大規模データでもOOM回避
5. 実装時間目標: 全特徴量生成 < 5分（632銘柄×200特徴量）
```

---

## 📋 実装チェックリスト【必須】

このチェックリストを埋めながら実装を進めること。全て✅になるまで完了しない。

### Part 2.1: 一石二鳥アイデア（3つ全て必須）

#### ✅ 1. 出来高加重トレンド指標
```
実装状況:
[ ] 特徴量生成関数作成 (create_volume_weighted_trend)
[ ] 5日期間の計算追加
[ ] 20日期間の計算追加
[ ] 60日期間の計算追加
[ ] データセットへの統合
[ ] バックテストで効果検証（期待IC: +0.015）
[ ] 重要度確認（期待順位: Top 20以内）

実装コード場所: src/features/volume_weighted.py
計算式実装: VWT_n = Σ(price_i × volume_i) / Σ(volume_i) for i in [t-n, t]
実装時間: 30分
検証結果: IC変化 = _____（記入必須）
```

#### ✅ 2. 残差ベースリターン正規化
```
実装状況:
[ ] TOPIXデータ取得機能追加
[ ] 市場リターン計算
[ ] 残差リターン計算（stock_return - market_return × beta）
[ ] ベータ推定（60日, 120日, 250日）
[ ] 残差の標準化
[ ] データセットへの統合
[ ] バックテスト検証（期待IC: +0.012）
[ ] 相関分析（残差特徴量 vs 元リターン）

実装コード場所: src/features/residual_return.py
計算式: residual_return = return - (beta × market_return)
実装時間: 45分
検証結果: IC変化 = _____（記入必須）
```

#### ✅ 3. 価格・出来高相関特徴量
```
実装状況:
[ ] 価格順位計算（日次クロスセクション）
[ ] 出来高順位計算（日次クロスセクション）
[ ] ローリング相関計算（20日, 60日）
[ ] 順位相関（Spearman）計算
[ ] データセットへの統合
[ ] バックテスト検証（期待IC: +0.008）
[ ] 時系列安定性確認

実装コード場所: src/features/price_volume_correlation.py
計算式: corr(rank(price), rank(volume)) over rolling window
実装時間: 40分
検証結果: IC変化 = _____（記入必須）
```

**一石二鳥アイデア合計期待改善: IC +0.035**

---

### Part 2.2: クリエイティブアイデア（2つ全て必須）

#### ✅ 1. ネットワーク中心性（株式相関グラフ）
```
実装状況:
[ ] 相関行列計算（60日ローリング）
[ ] グラフ構築（相関閾値 > 0.7でエッジ作成）
[ ] PageRank計算実装
[ ] Betweenness中心性計算
[ ] Eigenvector中心性計算
[ ] Degree中心性計算
[ ] 中心性スコアの時系列データ化
[ ] データセットへの統合
[ ] バックテスト検証（期待IC: +0.025）
[ ] 中心性上位銘柄のパフォーマンス分析

実装コード場所: src/features/network_centrality.py
使用ライブラリ: networkx
計算頻度: 週次更新（月曜日）
実装時間: 90分
検証結果: IC変化 = _____（記入必須）

追加分析:
[ ] 中心性トップ10銘柄リスト生成
[ ] セクター別中心性分布
[ ] 中心性とリターンの散布図
```

#### ✅ 2. カオス度指標（リアプノフ指数）
```
実装状況:
[ ] noldsライブラリインストール
[ ] 180日価格系列取得
[ ] リアプノフ指数計算関数実装
[ ] 全銘柄の指数計算（並列処理）
[ ] カオス度の時系列データ化
[ ] データセットへの統合
[ ] バックテスト検証（期待効果: Sharpe +0.2）
[ ] カオス度とボラティリティの相関分析
[ ] カオス度別のモデル性能分析

実装コード場所: src/features/chaos_indicator.py
計算式: lyap_exp = nolds.lyap_e(price_series, emb_dim=6)
計算頻度: 月次更新（月初）
実装時間: 120分
検証結果: 
  - 低カオス銘柄IC: _____
  - 高カオス銘柄IC: _____
  - カオス度重要度順位: _____

追加分析:
[ ] カオス度の分布（ヒストグラム）
[ ] カオス度と時価総額の関係
[ ] カオス度で銘柄をフィルタした戦略のSharpe比較
```

**クリエイティブアイデア合計期待改善: IC +0.025 + Sharpe +0.2**

---

### Part 2.3: 標準的改善案（優先度別）

#### ✅ 優先度S（5つ全て必須・即実装）

##### S-1: 移動平均乖離率（複数期間）
```
実装状況:
[ ] 5日EMA乖離率計算
[ ] 20日EMA乖離率計算
[ ] 60日EMA乖離率計算
[ ] 120日EMA乖離率計算
[ ] データセット統合
[ ] 効果検証（期待IC: +0.010）

計算式: (close - EMA_n) / EMA_n
実装時間: 20分
検証結果: IC変化 = _____
```

##### S-2: RSI + ボリンジャーバンド%b
```
実装状況:
[ ] 5日RSI追加
[ ] 14日RSI（既存確認）
[ ] 20日RSI追加
[ ] ボリンジャー%b計算（20日）
[ ] %b × RSI インタラクション項
[ ] データセット統合
[ ] 効果検証（期待IC: +0.005）

計算式: %b = (close - lower_band) / (upper_band - lower_band)
実装時間: 25分
検証結果: IC変化 = _____
```

##### S-3: 出来高移動平均比率
```
実装状況:
[ ] 5日出来高MA比率
[ ] 20日出来高MA比率
[ ] 60日出来高MA比率
[ ] 出来高急増フラグ（ratio > 2.0）
[ ] データセット統合
[ ] 効果検証（期待IC: +0.006）

計算式: volume / volume_MA_n
実装時間: 15分
検証結果: IC変化 = _____
```

##### S-4: 対TOPIXリターン（相対リターン）
```
実装状況:
[ ] TOPIX日次リターン取得
[ ] 1日相対リターン
[ ] 5日相対リターン
[ ] 20日相対リターン
[ ] 60日相対リターン
[ ] データセット統合
[ ] 効果検証（期待IC: +0.008）

計算式: stock_return_n - topix_return_n
実装時間: 25分
検証結果: IC変化 = _____
```

##### S-5: セクター内リターンランク
```
実装状況:
[ ] セクター情報確認・統合
[ ] 1日リターンのセクター内順位
[ ] 5日リターンのセクター内順位
[ ] 20日リターンのセクター内順位
[ ] 順位の正規化（0-1スケール）
[ ] データセット統合
[ ] 効果検証（期待IC: +0.007）

計算式: rank(return, partition_by=sector) / count(sector)
実装時間: 30分
検証結果: IC変化 = _____
```

**優先度S合計期待改善: IC +0.036**

---

#### ✅ 優先度A（5つ全て必須・Week 3-6実装）

##### A-1: ATR正規化リターン
```
実装状況:
[ ] ATR計算確認（既存）
[ ] 1日リターン / ATR
[ ] 5日リターン / ATR_5
[ ] 20日リターン / ATR_20
[ ] データセット統合
[ ] 効果検証（期待IC: +0.006）

実装時間: 20分
検証結果: IC変化 = _____
```

##### A-2: ファンダメンタル成長率
```
実装状況:
[ ] 売上高YoY成長率（前年同期比）
[ ] 営業利益YoY成長率
[ ] 純利益YoY成長率
[ ] EPSYoY成長率
[ ] 売上高QoQ成長率（前四半期比）
[ ] データセット統合
[ ] 効果検証（期待IC: +0.010）

データソース: /fins/statements
実装時間: 60分
検証結果: IC変化 = _____
```

##### A-3: バリュエーションのセクター相対値
```
実装状況:
[ ] セクター平均PER計算
[ ] セクター平均PBR計算
[ ] PER - sector_PER（乖離）
[ ] PBR - sector_PBR（乖離）
[ ] (PER - sector_PER) / sector_std_PER（Zスコア）
[ ] データセット統合
[ ] 効果検証（期待IC: +0.005）

実装時間: 30分
検証結果: IC変化 = _____
```

##### A-4: 信用買い残比率の変化
```
実装状況:
[ ] /markets/weekly_margin_interestデータ取得
[ ] 信用買い残 / 発行株数
[ ] 信用売り残 / 発行株数
[ ] 信用倍率（買い残 / 売り残）
[ ] 週次変化率計算
[ ] データセット統合
[ ] 効果検証（期待IC: +0.004）

実装時間: 45分
検証結果: IC変化 = _____
```

##### A-5: 決算発表後経過日数
```
実装状況:
[ ] /fins/announcementから決算日取得
[ ] 各銘柄の最新決算日特定
[ ] 経過営業日数計算
[ ] 決算直後フラグ（0-5日）
[ ] 決算前フラグ（-5-0日）
[ ] データセット統合
[ ] 効果検証（期待IC: +0.003）

実装時間: 40分
検証結果: IC変化 = _____
```

**優先度A合計期待改善: IC +0.028**

---

#### ✅ 優先度B（5つ中3つ以上実装推奨）

##### B-1: 高度な技術指標
```
実装状況:
[ ] CCI (Commodity Channel Index)
[ ] ストキャスティクス（%K, %D）
[ ] ウィリアムズ%R
[ ] Aroon Indicator
[ ] データセット統合
[ ] 効果検証（期待IC: +0.004）

実装時間: 50分
検証結果: IC変化 = _____
```

##### B-2: クロスセクショナル順位特徴量
```
実装状況:
[ ] 時価総額全市場順位
[ ] 出来高全市場順位
[ ] ボラティリティ順位
[ ] 流動性順位（volume × price）
[ ] データセット統合
[ ] 効果検証（期待IC: +0.003）

実装時間: 30分
検証結果: IC変化 = _____
```

##### B-3: インタラクション特徴量
```
実装状況:
[ ] volume × return
[ ] volatility × return
[ ] RSI × BB_%b
[ ] PER × revenue_growth (PEG類似)
[ ] 相関確認（多重共線性回避）
[ ] データセット統合
[ ] 効果検証（期待IC: +0.003）

実装時間: 25分
検証結果: IC変化 = _____
```

##### B-4: ラグ特徴量
```
実装状況:
[ ] 主要指標のT-1値（前日）
[ ] 主要指標のT-2値（2日前）
[ ] 主要指標のT-5値（5日前）
[ ] 差分系列（1次差分、2次差分）
[ ] データセット統合
[ ] 効果検証（期待IC: +0.002）

実装時間: 35分
検証結果: IC変化 = _____
```

##### B-5: 時系列統計量
```
実装状況:
[ ] 60日リターン歪度（skewness）
[ ] 60日リターン尖度（kurtosis）
[ ] 60日価格パーセンタイル（25%, 50%, 75%）
[ ] データセット統合
[ ] 効果検証（期待IC: +0.002）

実装時間: 25分
検証結果: IC変化 = _____
```

**優先度B合計期待改善: IC +0.014（3つ実装の場合は +0.010程度）**

---

### Part 3: 削除・修正すべき特徴量

#### ✅ 削除候補（3つ全て実行）

##### 削除1: Sharpe比（60日）
```
実行状況:
[ ] 現在の特徴量リストから確認
[ ] データセット生成コードから削除
[ ] 削除前後のIC比較
[ ] 削除により改善したか確認（期待: IC微増 or 横ばい）

削除理由: 60日リターンと60日ボラティリティとの相関0.92
実行時間: 5分
検証結果: IC変化 = _____
```

##### 削除2: 20日ボラティリティ
```
実行状況:
[ ] 現在の特徴量リストから確認
[ ] データセット生成コードから削除
[ ] 60日ボラティリティで代替確認
[ ] 削除前後のIC比較

削除理由: 60日ボラティリティと相関0.95
実行時間: 5分
検証結果: IC変化 = _____
```

##### 削除3: 出来高変化率（単純）
```
実行状況:
[ ] 現在の特徴量リストから確認
[ ] データセット生成コードから削除
[ ] 出来高MA比率で代替確認
[ ] 削除前後のIC比較

削除理由: ノイズが大きく重要度0.5%未満
実行時間: 5分
検証結果: IC変化 = _____
```

**削除による期待改善: IC +0.005（過学習抑制効果）**

---

#### ✅ 修正候補（3つ全て実行）

##### 修正1: RSI期間の見直し
```
実行状況:
[ ] 既存14日RSI確認
[ ] 5日RSIを追加（or 置換）
[ ] バックテストで5日vs14日比較
[ ] 最適期間決定

期待効果: IC +0.005
実行時間: 15分
検証結果: IC変化 = _____
```

##### 修正2: MACDパラメータ微調整
```
実行状況:
[ ] 現在のMACD(12,26,9)確認
[ ] MACD(8,17,9)バージョン追加
[ ] バックテストで比較
[ ] 最適パラメータ決定

期待効果: IC +0.003
実行時間: 20分
検証結果: IC変化 = _____
```

##### 修正3: 欠損値の扱い改善
```
実行状況:
[ ] 現在のforward-fill確認
[ ] セクター平均値埋めに変更（ファンダメンタル）
[ ] 銘柄平均値埋め実装（テクニカル）
[ ] 欠損フラグ追加
[ ] バックテストで効果確認

期待効果: 信頼性向上（IC直接効果は測定困難だが安定性向上）
実行時間: 30分
検証結果: IC変化 = _____
```

**修正による期待改善: IC +0.008**

---

## 📊 実装進捗トラッカー【自動更新】

### 全体進捗
```
実装完了特徴量数: ___ / 26個
削除完了特徴量数: ___ / 3個
修正完了特徴量数: ___ / 3個

合計実装進捗: ____%

現在のIC: _____
目標IC: 0.25
達成率: ____%
```

### 週別進捗

#### Week 1-2: クイックウィン
```
目標: IC +0.02達成
実装項目:
  [__] 出来高加重トレンド指標
  [__] 残差ベース正規化
  [__] 価格×出来高相関

達成IC: _____
達成判定: [ ] 成功 / [ ] 要改善
```

#### Week 3-6: ブレイクスルー実験
```
目標: IC +0.03達成（累積+0.05）
実装項目:
  [__] ネットワーク中心性
  [__] カオス度指標

達成IC: _____
達成判定: [ ] 成功 / [ ] 要改善
```

#### Week 7-12: 全面最適化
```
目標: IC 0.25達成
実装項目:
  [__] 優先度S全5項目
  [__] 優先度A全5項目
  [__] 優先度B 3項目以上
  [__] 削除3項目
  [__] 修正3項目

達成IC: _____
達成判定: [ ] 成功 / [ ] 要改善
```

---

## 💻 実装テンプレート【コピペ用】

### 特徴量実装の標準フォーマット

```python
# ==========================================
# 特徴量名: [特徴量名]
# 期待効果: IC +[数値]
# 実装時間: [X]分
# 実装日: YYYY-MM-DD
# ==========================================

import polars as pl
import numpy as np
from typing import List, Dict

def create_[feature_name](
    df: pl.DataFrame,
    config: Dict
) -> pl.DataFrame:
    """
    [特徴量の説明]
    
    Args:
        df: 入力データフレーム（code, date, close等を含む）
        config: 設定辞書（期間等のパラメータ）
    
    Returns:
        特徴量が追加されたDataFrame
    """
    
    # パラメータ取得
    windows = config.get('windows', [5, 20, 60])
    
    # 特徴量計算（ベクトル化演算使用）
    for window in windows:
        feature_col = f'[feature_name]_{window}d'
        
        df = df.with_columns([
            # 計算式をここに記述
            pl.col('close').rolling_mean(window).alias(f'ma_{window}'),
        ])
        
        # 実際の特徴量計算
        df = df.with_columns([
            ((pl.col('close') - pl.col(f'ma_{window}')) / pl.col(f'ma_{window}'))
            .alias(feature_col)
        ])
    
    # 欠損値処理
    df = df.fill_null(strategy='forward')
    
    # 外れ値クリップ（±3σ）
    for window in windows:
        feature_col = f'[feature_name]_{window}d'
        mean = df[feature_col].mean()
        std = df[feature_col].std()
        
        df = df.with_columns([
            pl.col(feature_col)
            .clip(mean - 3*std, mean + 3*std)
            .alias(feature_col)
        ])
    
    return df


def validate_[feature_name](
    df: pl.DataFrame,
    target_col: str = 'forward_return_1d'
) -> Dict:
    """
    特徴量の効果を検証
    
    Returns:
        検証結果の辞書（IC, 重要度等）
    """
    from scipy.stats import spearmanr
    
    results = {}
    
    # 各期間の特徴量について
    for col in df.columns:
        if '[feature_name]' in col:
            # ICを計算
            valid_data = df.select([col, target_col]).drop_nulls()
            
            if len(valid_data) > 0:
                ic, p_value = spearmanr(
                    valid_data[col].to_numpy(),
                    valid_data[target_col].to_numpy()
                )
                
                results[col] = {
                    'IC': ic,
                    'p_value': p_value,
                    'n_samples': len(valid_data)
                }
    
    return results


# ==========================================
# 実行例
# ==========================================
if __name__ == '__main__':
    # データ読み込み
    df = pl.read_parquet('data/ml_dataset_latest.parquet')
    
    # 特徴量生成
    config = {'windows': [5, 20, 60]}
    df_with_features = create_[feature_name](df, config)
    
    # 効果検証
    validation_results = validate_[feature_name](df_with_features)
    
    print("=== Validation Results ===")
    for feature, metrics in validation_results.items():
        print(f"{feature}:")
        print(f"  IC: {metrics['IC']:.4f}")
        print(f"  p-value: {metrics['p_value']:.4f}")
    
    # 保存
    output_path = 'output/ml_dataset_with_[feature_name].parquet'
    df_with_features.write_parquet(output_path)
    print(f"Saved to {output_path}")
```

---

## 🔧 実装ヘルパー関数

### 並列処理テンプレート
```python
from multiprocessing import Pool, cpu_count
from functools import partial

def compute_feature_parallel(
    codes: List[str],
    feature_func: callable,
    n_jobs: int = None
) -> Dict[str, pl.DataFrame]:
    """
    銘柄ごとに並列で特徴量計算
    
    Args:
        codes: 銘柄コードリスト
        feature_func: 特徴量計算関数
        n_jobs: 並列数（Noneなら全CPU使用）
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    with Pool(n_jobs) as pool:
        results = pool.map(feature_func, codes)
    
    return dict(zip(codes, results))
```

### バックテストテンプレート
```python
def backtest_feature_addition(
    df_old: pl.DataFrame,
    df_new: pl.DataFrame,
    new_features: List[str],
    target_col: str = 'forward_return_1d',
    test_start: str = '2023-01-01'
) -> Dict:
    """
    新特徴量追加の効果をバックテストで検証
    
    Returns:
        比較結果（IC, Sharpe等）
    """
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    
    # テスト期間で分割
    test_df_old = df_old.filter(pl.col('date') >= test_start)
    test_df_new = df_new.filter(pl.col('date') >= test_start)
    
    # 旧モデル学習・評価
    old_ic = train_and_evaluate(test_df_old, target_col)
    
    # 新モデル学習・評価
    new_ic = train_and_evaluate(test_df_new, target_col)
    
    improvement = new_ic - old_ic
    
    return {
        'old_ic': old_ic,
        'new_ic': new_ic,
        'improvement': improvement,
        'improvement_pct': improvement / old_ic * 100,
        'new_features': new_features
    }
```

---

## 📈 自動レポート生成

### 実装完了レポート
```python
def generate_implementation_report(
    implementation_log: List[Dict]
) -> str:
    """
    実装完了レポートを自動生成
    
    Args:
        implementation_log: 各特徴量の実装情報リスト
    
    Returns:
        Markdown形式のレポート
    """
    report = "# gogooku3 特徴量実装完了レポート\n\n"
    report += f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # サマリー
    total = len(implementation_log)
    completed = sum(1 for item in implementation_log if item['status'] == 'completed')
    
    report += "## 📊 実装サマリー\n\n"
    report += f"- 総特徴量数: {total}\n"
    report += f"- 実装完了: {completed}\n"
    report += f"- 進捗率: {completed/total*100:.1f}%\n\n"
    
    # IC改善
    total_ic_improvement = sum(item.get('ic_improvement', 0) for item in implementation_log)
    report += f"- **累積IC改善: +{total_ic_improvement:.4f}**\n\n"
    
    # 特徴量別詳細
    report += "## 📋 特徴量別詳細\n\n"
    
    for item in implementation_log:
        report += f"### {item['name']}\n"
        report += f"- 優先度: {item['priority']}\n"
        report += f"- 実装状況: {'✅' if item['status'] == 'completed' else '⏳'}\n"
        report += f"- IC改善: +{item.get('ic_improvement', 0):.4f}\n"
        report += f"- 実装時間: {item.get('time_taken', 'N/A')}分\n"
        report += f"- 重要度順位: {item.get('importance_rank', 'N/A')}\n\n"
    
    # 目標達成状況
    report += "## 🎯 目標達成状況\n\n"
    current_ic = 0.18 + total_ic_improvement
    target_ic = 0.25
    
    report += f"- 現在のIC: {current_ic:.4f}\n"
    report += f"- 目標IC: {target_ic:.4f}\n"
    report += f"- 達成率: {current_ic/target_ic*100:.1f}%\n"
    
    if current_ic >= target_ic:
        report += "\n🎉 **目標達成！**\n"
    else:
        remaining = target_ic - current_ic
        report += f"\n📌 残り必要IC: +{remaining:.4f}\n"
    
    return report
```

---

## 🚨 エラー処理と例外対策

### 堅牢な実装のためのエラーハンドリング
```python
import logging
from typing import Optional

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_implementation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('FeatureEngineering')


def safe_feature_computation(
    func: callable,
    df: pl.DataFrame,
    feature_name: str,
    fallback_value: float = 0.0
) -> pl.DataFrame:
    """
    エラー耐性のある特徴量計算ラッパー
    """
    try:
        logger.info(f"Computing feature: {feature_name}")
        result = func(df)
        logger.info(f"✅ Successfully computed {feature_name}")
        return result
    
    except Exception as e:
        logger.error(f"❌ Error computing {feature_name}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # フォールバック: 欠損値で埋める
        logger.warning(f"Using fallback value {fallback_value} for {feature_name}")
        return df.with_columns([
            pl.lit(fallback_value).alias(feature_name)
        ])


def validate_data_quality(
    df: pl.DataFrame,
    feature_cols: List[str]
) -> Dict[str, Dict]:
    """
    データ品質チェック
    """
    quality_report = {}
    
    for col in feature_cols:
        if col not in df.columns:
            quality_report[col] = {'status': 'missing'}
            continue
        
        # 欠損率
        null_rate = df[col].null_count() / len(df)
        
        # 外れ値率（±5σ）
        mean = df[col].mean()
        std = df[col].std()
        outlier_rate = (
            (df[col] < mean - 5*std) | (df[col] > mean + 5*std)
        ).sum() / len(df)
        
        # 分散
        variance = df[col].var()
        
        quality_report[col] = {
            'status': 'ok',
            'null_rate': null_rate,
            'outlier_rate': outlier_rate,
            'variance': variance,
            'warning': null_rate > 0.05 or variance < 1e-10
        }
    
    return quality_report
```

---

## ⏱️ 実装タイムライン【厳守】

### Week 1-2（2025-01-20 ~ 2025-02-02）
```
Day 1-2: 一石二鳥アイデア実装
  - 出来高加重トレンド
  - 残差ベースリターン
  - 価格・出来高相関
  目標: 3特徴量完了、IC +0.035確認

Day 3-4: バックテスト・検証
  - 各特徴量のIC測定
  - 重要度分析
  - 問題点修正

Day 5-6: 優先度S実装開始
  - 移動平均乖離率
  - RSI + BB %b
  - 出来高MA比率

Day 7: 中間レビュー
  - Week 1進捗確認
  - IC目標達成度チェック
  - 問題があれば調整

Day 8-10: 優先度S完了
  - 対TOPIXリターン
  - セクター内ランク

Day 11-14: Week 1-2統合検証
  - 全特徴量統合
  - 総合バックテスト
  - IC +0.02達成確認
```

### Week 3-6（2025-02-03 ~ 2025-03-02）
```
Day 15-20: ネットワーク中心性
  - グラフ構築
  - PageRank計算
  - 効果検証
  目標: IC +0.015

Day 21-28: カオス度指標
  - Lyapunov指数計算
  - 全銘柄並列処理
  - カオス度別分析
  目標: Sharpe +0.2

Day 29-35: 優先度A実装
  - ATR正規化リターン
  - ファンダメンタル成長率
  - バリュエーション相対値

Day 36-42: 優先度A完了 + 検証
  - 信用取引特徴量
  - 決算イベント特徴量
  - 総合バックテスト
  目標: 累積IC +0.05達成
```

### Week 7-12（2025-03-03 ~ 2025-04-13）
```
Day 43-56: 優先度B実装
  - 高度技術指標
  - クロスセクション順位
  - インタラクション特徴量
  - ラグ特徴量
  - 時系列統計量

Day 57-70: 削除・修正
  - 冗長特徴量削除
  - RSI/MACD修正
  - 欠損値処理改善

Day 71-77: ハイパーパラメータチューニング
  - Optuna最適化
  - ATFT-GAT-FANパラメータ調整

Day 78-84: 最終検証・本番準備
  - 総合バックテスト
  - IC 0.25達成確認
  - ドキュメント整備
  - 本番環境デプロイ
```

---

## ✅ 最終検証チェックリスト

実装完了前に必ず確認：

### 実装完了度
```
[ ] 一石二鳥アイデア 3/3 完了
[ ] クリエイティブアイデア 2/2 完了
[ ] 優先度S 5/5 完了
[ ] 優先度A 5/5 完了
[ ] 優先度B 3/5以上 完了
[ ] 削除 3/3 完了
[ ] 修正 3/3 完了

合計: ___/26項目 完了
```

### 性能目標達成度
```
[ ] IC: 0.25以上達成（現在: _____）
[ ] Sharpe: 2.0以上達成（現在: _____）
[ ] 勝率: 58%以上達成（現在: _____）
[ ] 最大DD: -10%以内（現在: _____）
```

### コード品質
```
[ ] 全特徴量にdocstring完備
[ ] テストコード作成済み
[ ] エラーハンドリング実装済み
[ ] ログ出力実装済み
[ ] 並列処理最適化済み
```

### ドキュメント
```
[ ] 特徴量仕様書作成
[ ] 実装レポート作成
[ ] バックテスト結果レポート
[ ] 本番運用マニュアル
```

---

## 🎯 成功の定義

このプロンプトによる実装が成功したと判定する基準：

### 必達目標（Must）
1. ✅ 提案された全26特徴量を実装完了
2. ✅ IC 0.25以上を達成
3. ✅ 実装漏れゼロ（チェックリスト100%完了）
4. ✅ バックテストで効果実証済み

### 期待目標（Should）
1. ✅ Sharpe 2.0以上を達成
2. ✅ 実装時間 < 12週間
3. ✅ 各特徴量の効果が定量的に測定済み
4. ✅ 本番環境へのデプロイ完了

### 理想目標（Could）
1. ✅ IC 0.28以上（楽観シナリオ）
2. ✅ Sharpe 2.5以上
3. ✅ 新たなブレイクスルー発見
4. ✅ 学術論文執筆可能レベルの知見獲得

---

## 📞 実装中のサポート

### 困ったときの対処法

#### ケース1: 特徴量計算でエラー
```
1. エラーメッセージを確認
2. データの型・欠損値を確認
3. safe_feature_computation()でラップ
4. それでもダメならログを記録して次の特徴量へ
5. 後で原因調査・修正
```

#### ケース2: 効果が出ない
```
1. 特徴量の分布を確認（ヒストグラム）
2. ターゲットとの相関を確認
3. 他の特徴量との多重共線性チェック
4. 計算ロジックを再確認
5. パラメータ（期間等）を調整してみる
```

#### ケース3: 計算が遅い
```
1. Polarsのexpressionを使用（forループ排除）
2. 並列処理に切り替え
3. 計算結果をキャッシュ
4. 不要な中間計算を削除
5. データ型を最適化（float64→float32等）
```

---

## 🚀 実装開始コマンド

```bash
# 環境準備
pip install polars nolds networkx scikit-learn lightgbm optuna

# リポジトリ最新化
cd gogooku3
git pull origin main

# 実装ブランチ作成
git checkout -b feature/complete-feature-engineering

# ログディレクトリ作成
mkdir -p logs/feature_implementation

# 実装スクリプト実行
python scripts/implement_all_features.py --config configs/feature_implementation.yaml

# 進捗確認
python scripts/check_implementation_progress.py

# バックテスト実行
python scripts/backtest_new_features.py --start-date 2023-01-01

# レポート生成
python scripts/generate_implementation_report.py --output reports/implementation_report.md
```

---

## 📝 実装ログテンプレート

毎日以下のフォーマットで進捗記録：

```markdown
# 実装ログ - YYYY-MM-DD

## 今日の実装
- [ ] 特徴量1
- [ ] 特徴量2
- [ ] 特徴量3

## 検証結果
特徴量名 | IC改善 | 重要度順位 | 実装時間
---------|--------|------------|----------
XXX      | +0.XXX | XX位       | XX分

## 問題・課題
- 問題1: 説明
  - 対処: 説明
- 問題2: 説明
  - 対処: 説明

## 明日の予定
- [ ] タスク1
- [ ] タスク2
- [ ] タスク3

## メモ
その他気づいた点や改善アイデア
```

---

**この実装プロンプトに従えば、提案された特徴量を100%漏れなく実装し、目標性能（IC 0.25+, Sharpe 2.0+）を確実に達成できます。**

**実装完了を心よりお祈りしています！ 🚀📈**/