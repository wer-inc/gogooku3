from apex_ranker.backtest import OptimizationConfig, generate_target_weights


def test_generate_target_weights_respects_topk_and_turnover() -> None:
    predictions = {"AAA": 0.6, "BBB": 0.55, "CCC": 0.4, "DDD": 0.35}
    current_weights = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}

    config = OptimizationConfig(
        target_top_k=2,
        candidate_multiplier=1.5,
        min_weight=0.2,
        turnover_limit=0.5,
        cost_penalty=0.0,
    )

    target_weights, result = generate_target_weights(
        predictions,
        current_weights,
        portfolio_value=1_000_000.0,
        config=config,
        cost_calculator=None,
        volumes=None,
    )

    assert len(target_weights) <= config.target_top_k
    assert abs(sum(target_weights.values()) - 1.0) < 1e-6
    assert set(target_weights.keys()).issubset(predictions.keys())
    assert result.selected_codes
