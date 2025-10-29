import torch
import pytest

from src.atft_gat_fan.models.components.adaptive_normalization import (
    FrequencyAdaptiveNorm,
    SliceAdaptiveNorm,
)


@pytest.mark.unit
def test_fan_weighted_mean_no_nan():
    torch.manual_seed(0)
    fan = FrequencyAdaptiveNorm(
        num_features=4,
        window_sizes=(3, 5, 7),
        aggregation="weighted_mean",
        learn_weights=True,
    )
    x = torch.randn(2, 10, 4)
    x[0, 3, 1] = float("nan")

    out = fan(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    assert fan._last_weights is not None
    weights = fan._last_weights
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))

    x.requires_grad_(True)
    out = fan(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.unit
def test_fan_mean_aggregation():
    fan = FrequencyAdaptiveNorm(
        num_features=3,
        window_sizes=(2, 4),
        aggregation="mean",
        learn_weights=False,
    )
    x = torch.randn(1, 5, 3)
    out = fan(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


@pytest.mark.unit
def test_san_learned_weights_no_nan():
    torch.manual_seed(1)
    san = SliceAdaptiveNorm(
        num_features=4,
        num_slices=3,
        overlap=0.25,
        slice_aggregation="learned",
    )
    x = torch.randn(2, 12, 4)
    x[1, 5, 2] = float("nan")

    out = san(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    assert san._last_weights is not None
    weights = san._last_weights
    torch.testing.assert_close(
        weights.sum(dim=1), torch.ones(weights.size(0), dtype=weights.dtype)
    )

    x.requires_grad_(True)
    out = san(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.unit
def test_san_mean_aggregation():
    san = SliceAdaptiveNorm(
        num_features=2,
        num_slices=2,
        overlap=0.5,
        slice_aggregation="mean",
    )
    x = torch.randn(1, 6, 2)
    out = san(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
