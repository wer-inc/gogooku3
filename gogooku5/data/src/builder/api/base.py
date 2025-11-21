"""Base utilities for API clients used in dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import requests
from requests import Response, Session

from ..config import DatasetBuilderSettings, get_settings

DEFAULT_HEADERS: Mapping[str, str] = {
    "User-Agent": "gogooku5-dataset-builder/0.1",
    "Accept": "application/json",
}


@dataclass
class APIClient:
    """Reusable base client that wraps a `requests.Session` instance."""

    base_url: str
    settings: DatasetBuilderSettings = field(default_factory=get_settings)
    session: Session = field(default_factory=requests.Session)

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Response:
        """Execute an HTTP request with standardized defaults."""

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        merged_headers = {**DEFAULT_HEADERS, **(headers or {})}
        response = self.session.request(
            method=method.upper(),
            url=url,
            params=params,
            json=json,
            headers=merged_headers,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        return response

    def close(self) -> None:
        """Close the underlying session."""

        self.session.close()

    def __enter__(self) -> "APIClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.close()
