from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .patchtst import PatchTSTEncoder


class MultiHorizonHead(nn.Module):
    """Linear heads for each forecasting horizon."""

    def __init__(self, d_model: int, horizons: Iterable[int]) -> None:
        super().__init__()
        self.horizons = [int(h) for h in horizons]
        self.heads = nn.ModuleDict({self._key(h): nn.Linear(d_model, 1) for h in self.horizons})

    @staticmethod
    def _key(horizon: int) -> str:
        return f"h{int(horizon)}"

    def forward(self, z: torch.Tensor) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        for h in self.horizons:
            head = self.heads[self._key(h)]
            outputs[h] = head(z).squeeze(-1)
        return outputs


class APEXRankerV0(nn.Module):
    """v0 baseline model (PatchTST + multi-horizon linear heads)."""

    def __init__(
        self,
        in_features: int,
        horizons: Iterable[int],
        *,
        d_model: int = 192,
        depth: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        patch_multiplier: int | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features  # FIX: Store for dimension validation
        self.horizons = [int(h) for h in horizons]

        self.encoder = PatchTSTEncoder(
            in_feats=in_features,
            d_model=d_model,
            depth=depth,
            patch_len=patch_len,
            stride=stride,
            n_heads=n_heads,
            dropout=dropout,
            patch_multiplier=patch_multiplier,  # Explicit from config
        )
        self.head = MultiHorizonHead(d_model, self.horizons)
        self.loss_fn = loss_fn

    def forward(self, X: torch.Tensor) -> dict[int, torch.Tensor]:
        pooled, _ = self.encoder(X)
        return self.head(pooled)

    def compute_loss(self, scores: dict[int, torch.Tensor], targets: torch.Tensor) -> torch.Tensor | None:
        """Compute multi-horizon loss.

        Returns:
            Loss tensor with requires_grad=True, or None if all horizons skipped.
        """
        if self.loss_fn is None:
            raise RuntimeError("loss_fn was not provided to APEXRankerV0")

        # GRADIENT-SAFE: Collect loss terms as list, then stack
        # Avoids leaf tensor in-place update (torch.tensor(0.0) + ...)
        loss_terms: list[torch.Tensor] = []

        for idx, horizon in enumerate(self.horizons):
            if idx >= targets.shape[1]:
                break
            y = targets[:, idx]
            # Skip low-variance targets
            if torch.std(y) < 1e-6:
                continue

            loss_h = self.loss_fn(scores[horizon], y)
            # loss_fn may return None if invalid (e.g., CompositeLoss)
            if loss_h is not None:
                loss_terms.append(loss_h)

        # Empty aggregation → return None (training loop should skip)
        if not loss_terms:
            return None

        # Stack and reduce → maintains gradient graph
        return torch.stack(loss_terms).mean()


# ============================================================================
# Feature-ABI Validation (Phase 1.1 - Production Safety)
# ============================================================================


def save_with_metadata(
    model: nn.Module,
    path: str,
    feature_names: list[str],
    config: dict | None = None,
) -> None:
    """Save checkpoint with feature metadata for ABI validation.

    Args:
        model: Model to save
        path: Checkpoint file path
        feature_names: List of feature names in exact order
        config: Optional config dict to embed in checkpoint
    """
    import hashlib

    # Generate feature hash (SHA256, first 16 chars)
    feature_hash = hashlib.sha256("|".join(feature_names).encode()).hexdigest()[:16]

    state = {
        "model_state_dict": model.state_dict(),
        "feature_names": feature_names,
        "feature_hash": feature_hash,
        "feature_count": len(feature_names),
        "config": config,
    }

    torch.save(state, path)
    print("✅ Saved checkpoint with Feature-ABI metadata:")
    print(f"   Features: {len(feature_names)}")
    print(f"   Hash: {feature_hash}")


def load_with_validation(
    path: str,
    expected_features: list[str],
    *,
    strict: bool = True,
) -> dict:
    """Load checkpoint and validate feature compatibility.

    Args:
        path: Checkpoint file path
        expected_features: Expected feature names in exact order
        strict: If True, raises ValueError on mismatch. If False, only warns.

    Returns:
        Loaded checkpoint dict with keys: model_state_dict, feature_names, etc.

    Raises:
        ValueError: If feature mismatch detected and strict=True
    """
    import hashlib

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Backward compatibility: Old checkpoints without metadata
    if "feature_names" not in ckpt:
        if strict:
            raise ValueError(
                f"⚠️  Checkpoint missing Feature-ABI metadata!\n"
                f"   Path: {path}\n"
                f"   This checkpoint was saved without feature_names.\n"
                f"   Set strict=False to load anyway (unsafe)."
            )
        else:
            print("⚠️  WARNING: Checkpoint missing Feature-ABI metadata (loading anyway)")
            return ckpt

    ckpt_names = ckpt["feature_names"]
    ckpt_hash = ckpt.get("feature_hash", "unknown")

    # Generate expected hash
    expected_hash = hashlib.sha256("|".join(expected_features).encode()).hexdigest()[:16]

    # Fast check: Hash comparison
    if ckpt_hash != expected_hash:
        mismatch_msg = (
            f"❌ Feature-ABI mismatch detected!\n"
            f"   Checkpoint: {path}\n"
            f"   Checkpoint hash: {ckpt_hash} ({len(ckpt_names)} features)\n"
            f"   Expected hash:   {expected_hash} ({len(expected_features)} features)\n"
            f"\n"
            f"   First 5 checkpoint features: {ckpt_names[:5]}\n"
            f"   First 5 expected features:   {expected_features[:5]}\n"
        )

        # Detailed comparison (slow, only on mismatch)
        if ckpt_names != expected_features:
            # Find first difference
            for i, (ckpt_feat, exp_feat) in enumerate(zip(ckpt_names, expected_features)):
                if ckpt_feat != exp_feat:
                    mismatch_msg += (
                        f"\n"
                        f"   First difference at index {i}:\n"
                        f"     Checkpoint: {ckpt_feat}\n"
                        f"     Expected:   {exp_feat}\n"
                    )
                    break
            if len(ckpt_names) != len(expected_features):
                mismatch_msg += f"\n" f"   Count mismatch: {len(ckpt_names)} vs {len(expected_features)}\n"

        if strict:
            raise ValueError(mismatch_msg)
        else:
            print(f"⚠️  WARNING: {mismatch_msg}")
            print("   Proceeding anyway (strict=False)")

    else:
        print("✅ Feature-ABI validation passed:")
        print(f"   Hash: {ckpt_hash}")
        print(f"   Features: {len(ckpt_names)}")

    return ckpt
