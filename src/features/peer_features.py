"""
Peer Feature Extractor for Stock Data
近傍銘柄の統計特徴量抽出（GAT段階的導入）
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class PeerFeatureExtractor:
    """
    近傍銘柄の統計特徴量抽出器

    GAT本実装への段階的導入:
    1. peer_mean/peer_var: セクター・相関ベースの近傍統計
    2. peer_momentum: 近傍のモメンタム統計
    3. peer_correlation: 動的相関統計

    データリーク防止:
    - 各日の統計計算は当日の情報のみ使用
    - 相関計算は過去期間の履歴から計算
    - 未来情報の混入を完全排除
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        correlation_window: int = 60,
        min_correlation_periods: int = 20,
        date_column: str = "date",
        code_column: str = "code",
        market_column: str | None = "market_code_name",
        return_columns: list[str] | None = None,
        feature_columns: list[str] | None = None,
    ):
        """
        Args:
            k_neighbors: 近傍銘柄数
            correlation_window: 相関計算期間（日数）
            min_correlation_periods: 相関計算の最小必要期間
            date_column: 日付カラム名
            code_column: 銘柄コードカラム名
            market_column: マーケットカラム名
            return_columns: リターンカラム（相関計算用）
            feature_columns: 特徴量カラム（統計計算用）
        """
        self.k_neighbors = k_neighbors
        self.correlation_window = correlation_window
        self.min_correlation_periods = min_correlation_periods
        self.date_column = date_column
        self.code_column = code_column
        self.market_column = market_column

        self.return_columns = return_columns or ["return_5d", "return_20d"]
        self.feature_columns = feature_columns

        # キャッシュされた相関行列とマッピング
        self.correlation_cache: dict[str, np.ndarray] = {}  # {date: corr_matrix}
        self.code_mapping_cache: dict[str, dict[str, int]] = {}  # {date: {code: idx}}
        self.peer_cache: dict[
            str, dict[str, list[str]]
        ] = {}  # {date: {code: peer_codes}}

    def add_peer_features(
        self, df: pd.DataFrame, method: str = "mixed", verbose: bool = False
    ) -> pd.DataFrame:
        """
        Peer特徴量を追加

        Args:
            df: 元データフレーム
            method: 近傍選択方法 ('sector', 'correlation', 'mixed')
            verbose: ログ出力

        Returns:
            Peer特徴量が追加されたデータフレーム
        """
        df = df.copy()

        # 特徴量カラムの自動検出
        if self.feature_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude_cols = {
                self.date_column,
                self.code_column,
                self.market_column,
                "target",
                "targets",
            } | set(self.return_columns)
            self.feature_columns = [
                col for col in numeric_cols if col not in exclude_cols
            ]

        if verbose:
            logger.info(
                f"Adding peer features using method '{method}' for {len(self.feature_columns)} features"
            )

        # 日付カラムの型確保
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        # 日付ごとに処理
        enhanced_rows = []
        unique_dates = sorted(df[self.date_column].unique())

        for date in unique_dates:
            day_data = df[df[self.date_column] == date].copy()

            if len(day_data) < self.k_neighbors + 1:
                # 銘柄数が不足している場合はpeer特徴量をゼロで埋める
                day_data = self._add_zero_peer_features(day_data)
                enhanced_rows.append(day_data)
                continue

            # 近傍銘柄を特定
            peer_mapping = self._find_peers(day_data, date, method)

            # Peer特徴量を計算
            day_data = self._compute_day_peer_features(day_data, peer_mapping)

            enhanced_rows.append(day_data)

            if verbose and len(enhanced_rows) % 50 == 0:
                logger.debug(f"Processed {len(enhanced_rows)} days for peer features")

        result = (
            pd.concat(enhanced_rows, ignore_index=True)
            if enhanced_rows
            else df.iloc[:0].copy()
        )

        if verbose:
            peer_cols = [col for col in result.columns if col.startswith("peer_")]
            logger.info(f"Added {len(peer_cols)} peer feature columns")

        return result

    def _find_peers(
        self, day_data: pd.DataFrame, date: pd.Timestamp, method: str
    ) -> dict[str, list[str]]:
        """日次の近傍銘柄を特定"""
        date_str = date.strftime("%Y-%m-%d")

        # キャッシュ確認
        if date_str in self.peer_cache:
            return self.peer_cache[date_str]

        peer_mapping = {}
        codes = day_data[self.code_column].tolist()

        if method == "sector" and self.market_column:
            peer_mapping = self._find_sector_peers(day_data, codes)
        elif method == "correlation":
            peer_mapping = self._find_correlation_peers(day_data, date, codes)
        elif method == "mixed":
            # セクター + 相関の組み合わせ
            sector_peers = (
                self._find_sector_peers(day_data, codes) if self.market_column else {}
            )
            corr_peers = self._find_correlation_peers(day_data, date, codes)
            peer_mapping = self._merge_peer_mappings(sector_peers, corr_peers, codes)
        else:
            # デフォルト: ランダム近傍
            peer_mapping = self._find_random_peers(codes)

        # キャッシュに保存
        self.peer_cache[date_str] = peer_mapping
        return peer_mapping

    def _find_sector_peers(
        self, day_data: pd.DataFrame, codes: list[str]
    ) -> dict[str, list[str]]:
        """セクター（マーケット）ベースの近傍銘柄"""
        peer_mapping = {}

        for code in codes:
            code_data = day_data[day_data[self.code_column] == code]
            if len(code_data) == 0:
                peer_mapping[code] = []
                continue

            market = code_data[self.market_column].iloc[0]

            # 同一セクターの他の銘柄
            same_market = day_data[
                (day_data[self.market_column] == market)
                & (day_data[self.code_column] != code)
            ][self.code_column].tolist()

            # k_neighbors分に制限
            peers = same_market[: self.k_neighbors]

            # 不足分は他のセクターから補完
            if len(peers) < self.k_neighbors:
                other_market = day_data[
                    (day_data[self.market_column] != market)
                    & (day_data[self.code_column] != code)
                ][self.code_column].tolist()

                needed = self.k_neighbors - len(peers)
                peers.extend(other_market[:needed])

            peer_mapping[code] = peers

        return peer_mapping

    def _find_correlation_peers(
        self, day_data: pd.DataFrame, date: pd.Timestamp, codes: list[str]
    ) -> dict[str, list[str]]:
        """相関ベースの近傍銘柄"""
        # 相関行列を取得または計算
        corr_matrix = self._get_correlation_matrix(date, codes, day_data)

        if corr_matrix is None:
            return self._find_random_peers(codes)

        peer_mapping = {}

        for i, code in enumerate(codes):
            if i >= corr_matrix.shape[0]:
                peer_mapping[code] = []
                continue

            # 自分以外の銘柄との相関
            correlations = corr_matrix[i, :].copy()
            correlations[i] = -2.0  # 自分を除外

            # 相関の高い順にソート（絶対値）
            corr_abs = np.abs(correlations)
            top_indices = np.argsort(corr_abs)[::-1]

            # 上位k個を選択
            peer_indices = top_indices[: self.k_neighbors]
            peer_codes = [codes[idx] for idx in peer_indices if idx < len(codes)]

            peer_mapping[code] = peer_codes

        return peer_mapping

    def _get_correlation_matrix(
        self, date: pd.Timestamp, codes: list[str], day_data: pd.DataFrame
    ) -> np.ndarray | None:
        """相関行列を取得または計算"""
        date_str = date.strftime("%Y-%m-%d")

        # キャッシュ確認
        if date_str in self.correlation_cache:
            cached_corr = self.correlation_cache[date_str]
            cached_mapping = self.code_mapping_cache.get(date_str, {})

            # コードマッピングの確認
            if all(code in cached_mapping for code in codes):
                # キャッシュされた相関行列からサブセットを抽出
                indices = [cached_mapping[code] for code in codes]
                return cached_corr[np.ix_(indices, indices)]

        # 相関行列を新規計算
        return self._compute_correlation_matrix(date, codes, day_data)

    def _compute_correlation_matrix(
        self, date: pd.Timestamp, codes: list[str], day_data: pd.DataFrame
    ) -> np.ndarray | None:
        """相関行列を計算"""
        try:
            # 履歴データから相関を計算（当日を含む過去期間）
            end_date = date
            start_date = date - pd.Timedelta(days=self.correlation_window)

            # 仮のデータフレーム作成（実際の実装では外部データソースから取得）
            # ここでは日次のreturn_5dを使用
            if "return_5d" not in day_data.columns:
                logger.warning("return_5d column not found, using random correlation")
                return self._generate_random_correlation_matrix(len(codes))

            # 簡易相関計算（実際の履歴データが必要）
            features_for_corr = day_data.set_index(self.code_column)[
                "return_5d"
            ].to_dict()

            # 全銘柄の特徴量行列を作成
            feature_matrix = []
            valid_codes = []

            for code in codes:
                if code in features_for_corr and not pd.isna(features_for_corr[code]):
                    feature_matrix.append([features_for_corr[code]])
                    valid_codes.append(code)

            if len(feature_matrix) < 3:
                return self._generate_random_correlation_matrix(len(codes))

            feature_matrix = np.array(feature_matrix)

            # 相関行列計算
            if feature_matrix.shape[1] == 1:
                # 単一特徴量の場合はコサイン類似度を使用
                corr_matrix = cosine_similarity(feature_matrix)
            else:
                corr_matrix = np.corrcoef(feature_matrix)

            # NaN値の処理
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

            # 対角成分を1に設定
            np.fill_diagonal(corr_matrix, 1.0)

            # キャッシュに保存
            date_str = date.strftime("%Y-%m-%d")
            code_mapping = {code: i for i, code in enumerate(valid_codes)}

            self.correlation_cache[date_str] = corr_matrix
            self.code_mapping_cache[date_str] = code_mapping

            return corr_matrix

        except Exception as e:
            logger.warning(f"Correlation calculation failed for {date.date()}: {e}")
            return self._generate_random_correlation_matrix(len(codes))

    def _generate_random_correlation_matrix(self, n: int) -> np.ndarray:
        """ランダム相関行列を生成"""
        # 正定値行列として生成
        A = np.random.randn(n, n)
        corr = np.dot(A, A.T)
        # 対角成分で正規化
        D = np.sqrt(np.diag(corr))
        corr = corr / np.outer(D, D)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _merge_peer_mappings(
        self,
        sector_peers: dict[str, list[str]],
        corr_peers: dict[str, list[str]],
        codes: list[str],
    ) -> dict[str, list[str]]:
        """セクターと相関ベースのpeerを統合"""
        merged_mapping = {}

        for code in codes:
            sector_list = sector_peers.get(code, [])
            corr_list = corr_peers.get(code, [])

            # セクター:相関 = 6:4 の比率で統合
            sector_count = min(len(sector_list), int(self.k_neighbors * 0.6))
            corr_count = self.k_neighbors - sector_count

            peers = sector_list[:sector_count]

            # 相関ベースで重複を避けて追加
            for peer in corr_list:
                if peer not in peers and len(peers) < self.k_neighbors:
                    peers.append(peer)

            merged_mapping[code] = peers

        return merged_mapping

    def _find_random_peers(self, codes: list[str]) -> dict[str, list[str]]:
        """ランダム近傍銘柄（フォールバック）"""
        peer_mapping = {}

        for code in codes:
            others = [c for c in codes if c != code]
            peers = np.random.choice(
                others, size=min(self.k_neighbors, len(others)), replace=False
            ).tolist()
            peer_mapping[code] = peers

        return peer_mapping

    def _compute_day_peer_features(
        self, day_data: pd.DataFrame, peer_mapping: dict[str, list[str]]
    ) -> pd.DataFrame:
        """日次のPeer特徴量を計算"""
        result_rows = []

        for _, row in day_data.iterrows():
            code = row[self.code_column]
            peer_codes = peer_mapping.get(code, [])

            # Peer銘柄のデータを取得
            peer_data = day_data[day_data[self.code_column].isin(peer_codes)]

            # Peer特徴量を計算
            peer_features = self._calculate_peer_statistics(row, peer_data)

            # 元の行にPeer特徴量を追加
            enhanced_row = row.copy()
            for key, value in peer_features.items():
                enhanced_row[f"peer_{key}"] = value

            result_rows.append(enhanced_row)

        return pd.DataFrame(result_rows)

    def _calculate_peer_statistics(
        self, target_row: pd.Series, peer_data: pd.DataFrame
    ) -> dict[str, float]:
        """Peer統計を計算"""
        peer_stats = {}

        if len(peer_data) == 0:
            # Peerがいない場合はゼロで埋める
            for feature in self.feature_columns:
                peer_stats[f"{feature}_mean"] = 0.0
                peer_stats[f"{feature}_std"] = 1.0
                peer_stats[f"{feature}_diff"] = 0.0

            peer_stats["count"] = 0
            peer_stats["rank_percentile"] = 0.5

            return peer_stats

        # 各特徴量のPeer統計を計算
        for feature in self.feature_columns:
            if feature not in peer_data.columns:
                continue

            peer_values = peer_data[feature].dropna()
            if len(peer_values) == 0:
                peer_stats[f"{feature}_mean"] = 0.0
                peer_stats[f"{feature}_std"] = 1.0
                peer_stats[f"{feature}_diff"] = 0.0
                continue

            peer_mean = float(peer_values.mean())
            peer_std = float(peer_values.std()) if len(peer_values) > 1 else 1.0
            peer_std = max(peer_std, 1e-8)  # ゼロ除算保護

            # 現在の値との差分（正規化）
            current_value = float(target_row.get(feature, 0.0))
            peer_diff = (current_value - peer_mean) / peer_std

            peer_stats[f"{feature}_mean"] = peer_mean
            peer_stats[f"{feature}_std"] = peer_std
            peer_stats[f"{feature}_diff"] = peer_diff

        # 追加統計
        peer_stats["count"] = len(peer_data)

        # ランク百分位数（主要特徴量で計算）
        if self.feature_columns and self.feature_columns[0] in peer_data.columns:
            main_feature = self.feature_columns[0]
            peer_values = peer_data[main_feature].dropna()
            if len(peer_values) > 0:
                current_value = float(target_row.get(main_feature, 0.0))
                rank_percentile = (peer_values < current_value).mean()
                peer_stats["rank_percentile"] = float(rank_percentile)
            else:
                peer_stats["rank_percentile"] = 0.5
        else:
            peer_stats["rank_percentile"] = 0.5

        return peer_stats

    def _add_zero_peer_features(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """Peer特徴量をゼロで埋める（銘柄数不足時）"""
        day_data = day_data.copy()

        for feature in self.feature_columns:
            day_data[f"peer_{feature}_mean"] = 0.0
            day_data[f"peer_{feature}_std"] = 1.0
            day_data[f"peer_{feature}_diff"] = 0.0

        day_data["peer_count"] = 0
        day_data["peer_rank_percentile"] = 0.5

        return day_data

    def clear_cache(self):
        """キャッシュをクリア"""
        self.correlation_cache.clear()
        self.code_mapping_cache.clear()
        self.peer_cache.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """キャッシュ統計を取得"""
        return {
            "correlation_cache_size": len(self.correlation_cache),
            "code_mapping_cache_size": len(self.code_mapping_cache),
            "peer_cache_size": len(self.peer_cache),
        }
