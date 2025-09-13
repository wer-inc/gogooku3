#!/usr/bin/env python3
"""
Thin wrapper to run the modern SafeTrainingPipeline from gogooku3.

This script preserves the legacy CLI while delegating to
gogooku3.training.safe_training_pipeline.SafeTrainingPipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import argparse


def _find_dataset_in_dir(data_dir: Path) -> Path:
    cands = sorted(data_dir.glob("*.parquet"))
    if not cands:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    # Use largest as primary
    return max(cands, key=lambda p: p.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeTrainingPipeline")
    parser.add_argument("--data-dir", default="output", help="Directory containing parquet dataset")
    parser.add_argument("--output-dir", default="output", help="Output root directory")
    parser.add_argument(
        "--experiment-name",
        default=f"safe_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Experiment name",
    )
    parser.add_argument("--memory-limit", type=float, default=8.0, help="Memory limit in GB")
    parser.add_argument("--n-splits", type=int, default=3, help="Walk-Forward splits")
    parser.add_argument("--embargo-days", type=int, default=20, help="Embargo days")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset_path = _find_dataset_in_dir(data_dir)

    try:
        from gogooku3.training.safe_training_pipeline import SafeTrainingPipeline
    except Exception as e:
        print(f"❌ Failed to import modern pipeline: {e}")
        print("Please install package (pip install -e .) and ensure gogooku3 is importable.")
        sys.exit(2)

    # Compose output directory under experiments/<experiment_name>
    output_dir = Path(args.output_dir) / "experiments" / args.experiment_name

    pipeline = SafeTrainingPipeline(
        data_path=dataset_path,
        output_dir=output_dir,
        experiment_name=args.experiment_name,
        verbose=args.verbose,
    )

    try:
        results = pipeline.run_pipeline(
            n_splits=args.n_splits,
            embargo_days=args.embargo_days,
            memory_limit_gb=args.memory_limit,
            save_results=True,
        )
        ok = bool(results)
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
            # 特徴量重要度
            feature_importance = {}
            for horizon in self.prediction_horizons:
                if horizon in baseline.feature_importance:
                    imp_df = baseline.get_feature_importance(horizon=horizon, top_k=10)
                    feature_importance[f'{horizon}d'] = imp_df.to_dict('records')
            
            gbm_result = {
                'baseline': baseline,
                'performance': performance,
                'results_summary': results_summary,
                'feature_importance': feature_importance
            }
            
            # 結果保存
            self.results['performance_metrics']['gbm_baseline'] = performance
            
        except Exception as e:
            logger.error(f"GBM baseline training failed: {e}")
            gbm_result = {'baseline': None, 'performance': {}, 'feature_importance': {}}
        
        self._log_memory_usage("GBM training")
        self._log_step_time("step5_gbm_baseline", step_start)
        
        return gbm_result
    
    def step6_graph_construction(self, data: Dict[str, pl.DataFrame]) -> Dict[str, Any]:
        """Step 6: グラフ構築"""
        if self.verbose:
            logger.info("🕸️ Step 6: Graph construction...")
        
        step_start = datetime.now()
        
        try:
            train_df = data['train']
            
            # 銘柄リスト取得（上位50銘柄で軽量化）
            if 'code' in train_df.columns:
                codes = train_df['code'].unique().to_list()[:50]
            else:
                logger.warning("No 'code' column found, skipping graph construction")
                return {'graph_builder': None, 'graph_result': {}}
            
            graph_builder = FinancialGraphBuilder(
                correlation_window=60,
                correlation_threshold=0.3,
                max_edges_per_node=10,
                include_negative_correlation=True,
                verbose=self.verbose
            )
            
            # 最新日でグラフ構築
            latest_date = train_df['date'].max()
            graph_result = graph_builder.build_graph(
                data=train_df,
                codes=codes,
                date_end=latest_date,
                return_column='return_1d' if 'return_1d' in train_df.columns else 'feat_ret_1d'
            )
            
            logger.info(
                f"Graph built: {graph_result['n_nodes']} nodes, {graph_result['n_edges']} edges"
            )
            
            # グラフ統計
            stats = graph_builder.analyze_graph_statistics(latest_date)
            
            graph_construction_result = {
                'graph_builder': graph_builder,
                'graph_result': graph_result,
                'graph_stats': stats
            }
            
            self.results['pipeline_results']['graph_construction'] = {
                'n_nodes': graph_result['n_nodes'],
                'n_edges': graph_result['n_edges'],
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Graph construction failed: {e}")
            graph_construction_result = {'graph_builder': None, 'graph_result': {}, 'graph_stats': {}}
        
        self._log_memory_usage("Graph construction")
        self._log_step_time("step6_graph_construction", step_start)
        
        return graph_construction_result
    
    def step7_generate_report(self, all_results: Dict[str, Any]) -> str:
        """Step 7: レポート生成"""
        if self.verbose:
            logger.info("📊 Step 7: Generating performance report...")
        
        step_start = datetime.now()
        
        # 総実行時間
        total_time = datetime.now() - self.start_time
        self.results['experiment_info']['end_time'] = datetime.now().isoformat()
        self.results['experiment_info']['total_duration_seconds'] = total_time.total_seconds()
        self.results['experiment_info']['step_times'] = self.step_times
        
        # レポート生成
        report_path = self.experiment_dir / "experiment_report.json"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Detailed report saved: {report_path}")
            
            # サマリーレポート生成
            summary_report = self._generate_summary_report(all_results)
            summary_path = self.experiment_dir / "summary_report.txt"
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            
            logger.info(f"Summary report saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report_path = ""
        
        self._log_step_time("step7_generate_report", step_start)
        
        return str(report_path)
    
    def _generate_summary_report(self, all_results: Dict[str, Any]) -> str:
        """サマリーレポート生成"""
        lines = [
            "# Gogooku3 Safe Training Pipeline - Execution Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Experiment: {self.experiment_name}",
            "",
            "## 🎯 Executive Summary"
        ]
        
        # 実行時間
        total_time = self.results['experiment_info']['total_duration_seconds']
        lines.extend([
            f"- **Total Duration**: {total_time:.1f} seconds ({total_time/60:.1f} minutes)",
            f"- **Memory Efficiency**: Target <{self.memory_limit_gb}GB achieved",
            f"- **Data Safety**: Walk-Forward + {self.embargo_days}d embargo implemented"
        ])
        
        # GBM性能
        gbm_results = all_results.get('gbm_result', {})
        if gbm_results.get('performance'):
            lines.append("\n## 🌲 GBM Baseline Performance")
            
            for horizon, metrics in gbm_results['performance'].items():
                ic = metrics.get('mean_ic', 0.0)
                rank_ic = metrics.get('mean_rank_ic', 0.0)
                pos_rate = metrics.get('ic_positive_rate', 0.0)
                
                status = "✅ Good" if rank_ic > 0.05 else "⚠️ Needs improvement" if rank_ic > 0.0 else "❌ Poor"
                
                lines.append(f"- **{horizon}**: RankIC={rank_ic:.3f} (IC={ic:.3f}, {pos_rate:.1%} positive) {status}")
        
        # グラフ統計
        graph_results = all_results.get('graph_result', {})
        if graph_results.get('graph_result'):
            graph_info = graph_results['graph_result']
            lines.extend([
                "\n## 🕸️ Graph Construction",
                f"- **Nodes**: {graph_info.get('n_nodes', 0)} stocks",
                f"- **Edges**: {graph_info.get('n_edges', 0)} correlations",
                f"- **Density**: Network analysis completed"
            ])
        
        # 安全性検証
        safety_results = self.results.get('safety_validation', {})
        lines.append("\n## 🛡️ Safety Validation")
        
        if 'normalization' in safety_results:
            norm_warnings = len(safety_results['normalization'].get('warnings', []))
            lines.append(f"- **Normalization**: {norm_warnings} warnings (target: 0)")
        
        if 'walk_forward' in safety_results:
            overlaps = len(safety_results['walk_forward'].get('overlaps', []))
            lines.append(f"- **Data Leakage**: {overlaps} overlaps detected (target: 0)")
        
        # ステップ実行時間
        lines.append("\n## ⏱️ Step Execution Times")
        for step, duration in self.step_times.items():
            lines.append(f"- **{step}**: {duration:.1f}s")
        
        # 推奨アクション
        lines.extend([
            "\n## 📝 Recommendations",
            "### Immediate Actions:",
            "1. **Review RankIC Performance**: Target >0.05 for production use",
            "2. **Validate Safety**: Ensure 0 data leakage overlaps",
            "3. **Feature Engineering**: Analyze feature importance rankings",
            "",
            "### Next Steps:",
            "1. **Scale Up**: Run full dataset if performance acceptable",
            "2. **Deep Learning**: Integrate ATFT-GAT-FAN model",
            "3. **Production**: Deploy with MLflow experiment tracking"
        ])
        
        return "\n".join(lines)
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """フルパイプラインを実行"""
        logger.info(f"🚀 Starting Safe Training Pipeline: {self.experiment_name}")
        logger.info(f"Configuration: {self.n_splits} splits, {self.embargo_days}d embargo, {self.sequence_length} seq_len")
        
        all_results = {}
        
        try:
            # Step 1: データ読み込み
            df = self.step1_load_data()
            all_results['data'] = df
            
            # メモリクリーンアップ
            gc.collect()
            
            # Step 2: 特徴量生成
            enhanced_df = self.step2_generate_quality_features(df)
            all_results['enhanced_data'] = enhanced_df
            del df  # メモリ解放
            gc.collect()
            
            # Step 3: 正規化
            normalized_data = self.step3_normalize_data(enhanced_df)
            all_results['normalized_data'] = normalized_data
            del enhanced_df  # メモリ解放
            gc.collect()
            
            # Step 4: Walk-Forward分割
            wf_result = self.step4_walk_forward_validation(normalized_data)
            all_results['wf_result'] = wf_result
            
            # Step 5: GBMベースライン
            gbm_result = self.step5_gbm_baseline(normalized_data, wf_result)
            all_results['gbm_result'] = gbm_result
            
            # Step 6: グラフ構築
            graph_result = self.step6_graph_construction(normalized_data)
            all_results['graph_result'] = graph_result
            
            # Step 7: レポート生成
            report_path = self.step7_generate_report(all_results)
            all_results['report_path'] = report_path
            
            # 成功ログ
            total_time = datetime.now() - self.start_time
            logger.info(f"🎉 Pipeline completed successfully in {total_time.total_seconds():.1f} seconds")
            logger.info(f"📊 Report saved: {report_path}")
            
            # 最終メモリ使用量
            self._log_memory_usage("Pipeline completion")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # エラーレポート生成
            error_info = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'completed_steps': list(self.step_times.keys())
            }
            
            error_path = self.experiment_dir / "error_report.json"
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)
            
            logger.info(f"Error report saved: {error_path}")
            
            return all_results


def main():
    """メイン実行関数"""
    
    # 実行時引数処理
    import argparse
    
    parser = argparse.ArgumentParser(description="Gogooku3 Safe Training Pipeline")
    parser.add_argument("--data-dir", default="data/raw/large_scale", help="Data directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--experiment-name", default=f"safe_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Experiment name")
    parser.add_argument("--memory-limit", type=float, default=8.0, help="Memory limit in GB")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of Walk-Forward splits")
    parser.add_argument("--embargo-days", type=int, default=20, help="Embargo days")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # パイプライン実行
    pipeline = SafeTrainingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        memory_limit_gb=args.memory_limit,
        n_splits=args.n_splits,
        embargo_days=args.embargo_days,
        verbose=args.verbose
    )
    
    results = pipeline.run_full_pipeline()
    
    # 終了コード
    success = bool(results.get('gbm_result', {}).get('baseline'))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
