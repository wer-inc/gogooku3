Subprojects for alternative training approaches

目的（Purpose）
- 既存のデータセット/パイプラインを再利用しつつ、別流儀の学習手法を「小さく始めて大きく採る」ための実験場。
- 本体を汚さず（低リスク）、価値が確認できたものだけを段階的に本線へ還元する。

方針（Policy）
- 目的指標（RankIC/Sharpe）との整合を最優先に、薄いラッパで既存訓練を切り替える。
- 評価は固定検証窓の purged walk‑forward＋embargo に統一。再現2回で採択。
- 安全策（bf16・勾配クリップ・T‑1ラグ厳守）を共通ガードレールとして維持。

Available subprojects:
- cs_ranking: Cross‑sectional day‑ranking training (Pairwise + RankIC focus)

Structure
- Each subproject includes a small README (目的/方針/指標) and runnable scripts that wrap the existing training pipeline with targeted settings.
