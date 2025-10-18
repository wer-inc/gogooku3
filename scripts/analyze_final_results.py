#!/usr/bin/env python3
"""
最終トレーニング結果の詳細分析
"""

import logging
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_metrics(log_path: str = "logs/ml_training.log"):
    """トレーニングログからメトリクスを抽出"""
    metrics = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_sharpe": [],
        "val_sharpe": [],
        "train_ic": [],
        "val_ic": [],
        "train_rankic": [],
        "val_rankic": [],
        "lr": [],
    }

    with open(log_path) as f:
        lines = f.readlines()

    for line in lines:
        if "Epoch" in line and "Train Loss" in line:
            try:
                # Extract epoch info
                parts = line.split()
                epoch_info = [p for p in parts if "/" in p and p[0].isdigit()][0]
                epoch = int(epoch_info.split("/")[0])

                # Extract losses
                train_loss = float(line.split("Train Loss=")[1].split(",")[0])
                val_loss = float(line.split("Val Loss=")[1].split(",")[0])
                lr = float(line.split("LR=")[1].strip())

                metrics["epochs"].append(epoch)
                metrics["train_loss"].append(train_loss)
                metrics["val_loss"].append(val_loss)
                metrics["lr"].append(lr)
            except:
                continue

        elif "Train Metrics" in line:
            try:
                sharpe = float(line.split("Sharpe:")[1].split(",")[0])
                ic = float(line.split("IC:")[1].split(",")[0])
                rankic = float(line.split("RankIC:")[1].strip())

                metrics["train_sharpe"].append(sharpe)
                metrics["train_ic"].append(ic)
                metrics["train_rankic"].append(rankic)
            except:
                continue

        elif "Val Metrics" in line:
            try:
                sharpe = float(line.split("Sharpe:")[1].split(",")[0])
                ic = float(line.split("IC:")[1].split(",")[0])
                rankic = float(line.split("RankIC:")[1].strip())

                metrics["val_sharpe"].append(sharpe)
                metrics["val_ic"].append(ic)
                metrics["val_rankic"].append(rankic)
            except:
                continue

    # データフレームに変換
    df = pd.DataFrame(metrics)
    return df


def analyze_improvement_trajectory(df: pd.DataFrame):
    """改善の軌跡を分析"""
    logger.info("=== 改善の軌跡分析 ===")

    # 最初と最後の比較
    if len(df) > 0:
        first_val_sharpe = df["val_sharpe"].iloc[0]
        last_val_sharpe = df["val_sharpe"].iloc[-1]
        best_val_sharpe = df["val_sharpe"].max()

        logger.info(f"初回Val Sharpe: {first_val_sharpe:.4f}")
        logger.info(f"最終Val Sharpe: {last_val_sharpe:.4f}")
        logger.info(f"最高Val Sharpe: {best_val_sharpe:.4f}")
        logger.info(
            f"改善率: {(last_val_sharpe - first_val_sharpe) / abs(first_val_sharpe) * 100:.1f}%"
        )

        # ICの改善
        first_val_ic = df["val_ic"].iloc[0]
        last_val_ic = df["val_ic"].iloc[-1]
        logger.info(f"\nVal IC改善: {first_val_ic:.4f} → {last_val_ic:.4f}")

        # 過学習の確認
        train_sharpe_trend = df["train_sharpe"].iloc[-10:].mean()
        val_sharpe_trend = df["val_sharpe"].iloc[-10:].mean()
        logger.info("\n過学習チェック:")
        logger.info(
            f"最近10エポックの平均 - Train: {train_sharpe_trend:.4f}, Val: {val_sharpe_trend:.4f}"
        )

        if train_sharpe_trend > val_sharpe_trend * 2:
            logger.warning("警告: 過学習の兆候があります")


def plot_training_curves(
    df: pd.DataFrame, save_path: str = "output/training_curves.png"
):
    """トレーニング曲線をプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(df["epochs"], df["train_loss"], label="Train Loss")
    axes[0, 0].plot(df["epochs"], df["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Sharpe curves
    axes[0, 1].plot(df["epochs"], df["val_sharpe"], label="Val Sharpe", color="green")
    axes[0, 1].axhline(y=0.849, color="red", linestyle="--", label="Target (0.849)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Sharpe Ratio")
    axes[0, 1].set_title("Validation Sharpe Ratio")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # IC curves
    axes[1, 0].plot(df["epochs"], df["train_ic"], label="Train IC")
    axes[1, 0].plot(df["epochs"], df["val_ic"], label="Val IC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IC")
    axes[1, 0].set_title("Information Coefficient")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate
    axes[1, 1].plot(df["epochs"], df["lr"], label="Learning Rate", color="orange")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Training curves saved to {save_path}")


def generate_recommendations(df: pd.DataFrame):
    """次のステップの推奨事項を生成"""
    logger.info("\n=== 推奨事項 ===")

    recommendations = []

    # Sharpe比の状況に基づく推奨
    if len(df) > 0:
        last_val_sharpe = df["val_sharpe"].iloc[-1]

        if last_val_sharpe > 0.5:
            recommendations.append(
                "✅ Sharpe比が良好です。アンサンブル学習を検討してください。"
            )
        elif last_val_sharpe > 0.2:
            recommendations.append(
                "📈 Sharpe比が改善しています。ハイパーパラメータの微調整を推奨。"
            )
        elif last_val_sharpe > 0:
            recommendations.append(
                "🔄 Sharpe比がプラスです。ポートフォリオ最適化を適用してください。"
            )
        else:
            recommendations.append(
                "⚠️ Sharpe比がまだマイナスです。データ品質の再確認が必要。"
            )

        # 過学習チェック
        if len(df) > 20:
            train_loss_trend = df["train_loss"].iloc[-10:].diff().mean()
            val_loss_trend = df["val_loss"].iloc[-10:].diff().mean()

            if train_loss_trend < 0 and val_loss_trend > 0:
                recommendations.append("⚠️ 過学習の兆候。正則化を強化してください。")

        # IC/RankICチェック
        last_val_ic = df["val_ic"].iloc[-1]
        if abs(last_val_ic) < 0.01:
            recommendations.append(
                "📊 ICが低すぎます。特徴量エンジニアリングの見直しを。"
            )

    # 目標達成までの道筋
    target_sharpe = 0.849
    if last_val_sharpe > 0:
        gap = target_sharpe - last_val_sharpe
        improvement_needed = gap / last_val_sharpe * 100
        recommendations.append(
            f"🎯 目標達成まで: あと{improvement_needed:.0f}%の改善が必要"
        )

    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec}")


def main():
    """メイン実行"""
    logger.info("=== 最終トレーニング結果分析 ===")
    logger.info(f"分析開始: {datetime.now()}")

    # メトリクスを読み込み
    try:
        df = load_training_metrics()
        logger.info(f"読み込んだエポック数: {len(df)}")

        if len(df) > 0:
            # 改善の軌跡を分析
            analyze_improvement_trajectory(df)

            # グラフを作成
            plot_training_curves(df)

            # 推奨事項を生成
            generate_recommendations(df)

            # 結果をCSVに保存
            output_path = f"output/training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"\nメトリクスを保存: {output_path}")

        else:
            logger.warning("トレーニングメトリクスが見つかりません")

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
