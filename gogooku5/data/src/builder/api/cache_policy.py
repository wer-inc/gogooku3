"""Source cache policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional


def _parse_asof(value: str | None) -> Optional[date]:
    if not value:
        return None
    value = value.strip()
    if value.lower() in {"today", "now", "latest"}:
        return datetime.utcnow().date()
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        raise ValueError(f"Invalid SOURCE_CACHE_ASOF value: {value!r}. Use YYYY-MM-DD or 'today'.") from None


@dataclass
class SourceCachePolicy:
    """Policy describing how builder data sources should interact with the cache."""

    dataset: str
    ttl_days: Optional[int]
    enable_read: bool = True
    enable_write: bool = True
    force_refresh: bool = False
    asof: Optional[date] = None
    tag: Optional[str] = None

    def decorate_key(self, base_key: str) -> str:
        """Append as-of/tag segments to the provided cache key."""

        suffix = []
        if self.asof:
            suffix.append(f"asof-{self.asof.isoformat()}")
        if self.tag:
            suffix.append(self.tag)
        if not suffix:
            return base_key
        return "_".join([base_key] + suffix)

    def metadata(self) -> Dict[str, str]:
        """Return metadata recorded alongside cache entries."""

        meta: Dict[str, str] = {"dataset": self.dataset}
        if self.asof:
            meta["asof"] = self.asof.isoformat()
        if self.tag:
            meta["tag"] = self.tag
        return meta

    @classmethod
    def from_settings(
        cls,
        *,
        dataset: str,
        ttl_days: Optional[int],
        mode: str,
        force_refresh: bool,
        asof_value: Optional[str],
        tag: Optional[str],
        ttl_override: Optional[int] = None,
    ) -> "SourceCachePolicy":
        """Construct a policy from DatasetBuilderSettings."""

        normalized_mode = (mode or "read_write").lower()
        enable_read = normalized_mode in {"read", "read_write"}
        enable_write = normalized_mode == "read_write"
        resolved_ttl = ttl_override if ttl_override is not None else ttl_days
        return cls(
            dataset=dataset,
            ttl_days=resolved_ttl,
            enable_read=enable_read,
            enable_write=enable_write,
            force_refresh=force_refresh,
            asof=_parse_asof(asof_value),
            tag=tag,
        )
