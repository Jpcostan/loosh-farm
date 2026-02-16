"""Snapshot data model — the final output structure."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class MoodSnapshot:
    """Complete mood snapshot at a point in time."""

    timestamp_utc: str = ""
    items_analyzed: int = 0
    sentiment: dict[str, float] = field(default_factory=dict)
    emotions: dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = "neutral"
    loosh_index: float = 50.0
    loosh_label: str = "Neutral"
    loosh_components: dict = field(default_factory=dict)
    trending_topics: dict = field(default_factory=dict)
    source_breakdown: dict = field(default_factory=dict)
    analysis_mode: str = "lightweight"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def create(
        items_analyzed: int,
        sentiment: dict,
        emotions: dict,
        dominant_emotion: str,
        loosh_index: float,
        loosh_label: str,
        loosh_components: dict,
        trending_topics: dict,
        source_breakdown: dict,
        analysis_mode: str,
    ) -> MoodSnapshot:
        return MoodSnapshot(
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            items_analyzed=items_analyzed,
            sentiment=sentiment,
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            loosh_index=loosh_index,
            loosh_label=loosh_label,
            loosh_components=loosh_components,
            trending_topics=trending_topics,
            source_breakdown=source_breakdown,
            analysis_mode=analysis_mode,
        )
