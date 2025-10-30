"""Detect corporate actions or market events."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

EventRecord = Mapping[str, object]


@dataclass
class EventDetector:
    """Simple event detector placeholder."""

    event_types: List[str]

    def detect(self, events: Iterable[EventRecord]) -> List[EventRecord]:
        """Filter events to those we currently care about."""

        return [event for event in events if event.get("eventType") in self.event_types]

    def add_event_type(self, event_type: str) -> None:
        """Register an additional event type."""

        if event_type not in self.event_types:
            self.event_types.append(event_type)
