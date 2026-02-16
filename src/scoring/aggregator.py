"""Aggregates per-item analysis results into a global snapshot."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.collectors.base import CollectedItem
from src.models.sentiment import SentimentResult
from src.models.emotion import EmotionResult, EMOTION_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class AggregatedResult:
    """Aggregated analysis across all items."""

    items_analyzed: int = 0

    # Global sentiment (averaged)
    sentiment: dict[str, float] = field(default_factory=dict)

    # Global emotion distribution (averaged)
    emotions: dict[str, float] = field(default_factory=dict)

    # Per-source breakdown
    source_breakdown: dict[str, dict] = field(default_factory=dict)

    # Dominant emotion
    dominant_emotion: str = "neutral"

    # Average compound sentiment
    avg_compound: float = 0.0


class Aggregator:
    """Aggregates individual results into a snapshot."""

    def __init__(self, config: dict):
        self.source_weights = config.get("source_weights", {})

    def aggregate(
        self,
        items: list[CollectedItem],
        sentiments: list[SentimentResult],
        emotions: list[EmotionResult],
    ) -> AggregatedResult:
        if not items:
            return AggregatedResult()

        n = len(items)

        # Global sentiment averages
        avg_pos = sum(s.positive for s in sentiments) / n
        avg_neu = sum(s.neutral for s in sentiments) / n
        avg_neg = sum(s.negative for s in sentiments) / n
        avg_compound = sum(s.compound for s in sentiments) / n

        global_sentiment = {
            "positive": round(avg_pos, 4),
            "neutral": round(avg_neu, 4),
            "negative": round(avg_neg, 4),
        }

        # Global emotion averages
        global_emotions: dict[str, float] = {}
        for emotion in EMOTION_CATEGORIES:
            avg = sum(e.scores.get(emotion, 0.0) for e in emotions) / n
            global_emotions[emotion] = round(avg, 4)

        # Normalize emotion scores to sum to 1.0
        emo_total = sum(global_emotions.values())
        if emo_total > 0:
            global_emotions = {k: round(v / emo_total, 4) for k, v in global_emotions.items()}

        dominant = max(global_emotions, key=global_emotions.get)

        # Per-source breakdown
        source_groups: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(items):
            source_groups[item.source_category].append(i)

        source_breakdown = {}
        for source_cat, indices in source_groups.items():
            src_n = len(indices)
            src_sentiment = {
                "positive": round(sum(sentiments[i].positive for i in indices) / src_n, 4),
                "neutral": round(sum(sentiments[i].neutral for i in indices) / src_n, 4),
                "negative": round(sum(sentiments[i].negative for i in indices) / src_n, 4),
            }
            src_emotions = {}
            for emotion in EMOTION_CATEGORIES:
                src_emotions[emotion] = round(
                    sum(emotions[i].scores.get(emotion, 0.0) for i in indices) / src_n, 4
                )
            source_breakdown[source_cat] = {
                "count": src_n,
                "sentiment": src_sentiment,
                "emotions": src_emotions,
                "weight": self.source_weights.get(source_cat, 0.0),
            }

        return AggregatedResult(
            items_analyzed=n,
            sentiment=global_sentiment,
            emotions=global_emotions,
            source_breakdown=source_breakdown,
            dominant_emotion=dominant,
            avg_compound=round(avg_compound, 4),
        )
