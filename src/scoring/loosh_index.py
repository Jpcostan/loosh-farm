"""Loosh Index calculation — a single numeric score for global emotional temperature."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .aggregator import AggregatedResult

logger = logging.getLogger(__name__)


@dataclass
class LooshScore:
    index: float  # 0-100
    label: str  # human-readable label
    components: dict  # breakdown of what contributed


class LooshIndexCalculator:
    """
    Calculates a 0-100 Loosh Index.

    The index represents aggregate emotional intensity/negativity.
    Higher = more intense negative emotions detected globally.
    Lower = calmer, more positive emotional landscape.

    Formula:
      1. Weighted emotion contributions (negative emotions push up, positive push down)
      2. Sentiment negativity ratio amplifier
      3. Source-weighted adjustment
      4. Clamped to [0, 100]
    """

    def __init__(self, config: dict):
        loosh_config = config.get("loosh_index", {})
        self.emotion_polarity = loosh_config.get("emotion_polarity", {
            "joy": -1.0,
            "anger": 1.5,
            "fear": 1.8,
            "sadness": 1.2,
            "surprise": 0.3,
            "disgust": 1.0,
            "trust": -0.8,
            "anticipation": 0.2,
        })
        self.baseline = loosh_config.get("baseline", 50)
        self.scale_min = loosh_config.get("scale_min", 0)
        self.scale_max = loosh_config.get("scale_max", 100)
        self.source_weights = config.get("source_weights", {})

    def calculate(self, aggregated: AggregatedResult) -> LooshScore:
        if aggregated.items_analyzed == 0:
            return LooshScore(index=self.baseline, label="No Data", components={})

        # Step 1: Emotion-weighted contribution
        emotion_signal = 0.0
        emotion_components = {}
        for emotion, score in aggregated.emotions.items():
            polarity = self.emotion_polarity.get(emotion, 0.0)
            contribution = score * polarity
            emotion_signal += contribution
            emotion_components[emotion] = {
                "score": score,
                "polarity": polarity,
                "contribution": round(contribution, 4),
            }

        # Step 2: Sentiment negativity amplifier
        neg_ratio = aggregated.sentiment.get("negative", 0.0)
        pos_ratio = aggregated.sentiment.get("positive", 0.0)
        sentiment_amplifier = 1.0 + (neg_ratio - pos_ratio) * 0.5

        # Step 3: Source-weighted adjustment (use per-source if available)
        if aggregated.source_breakdown and self.source_weights:
            weighted_signal = 0.0
            total_weight = 0.0
            for source_cat, breakdown in aggregated.source_breakdown.items():
                weight = self.source_weights.get(source_cat, 0.0)
                if weight <= 0:
                    continue
                # Per-source emotion signal
                src_signal = 0.0
                for emotion, score in breakdown.get("emotions", {}).items():
                    polarity = self.emotion_polarity.get(emotion, 0.0)
                    src_signal += score * polarity
                weighted_signal += src_signal * weight
                total_weight += weight

            if total_weight > 0:
                emotion_signal = weighted_signal / total_weight

        # Step 4: Compute raw index
        raw_index = self.baseline + (emotion_signal * sentiment_amplifier * 25)

        # Step 5: Clamp
        index = max(self.scale_min, min(self.scale_max, round(raw_index, 1)))

        # Human-readable label
        label = self._label(index)

        return LooshScore(
            index=index,
            label=label,
            components={
                "emotion_signal": round(emotion_signal, 4),
                "sentiment_amplifier": round(sentiment_amplifier, 4),
                "baseline": self.baseline,
                "raw_index": round(raw_index, 2),
                "emotion_breakdown": emotion_components,
            },
        )

    def _label(self, index: float) -> str:
        if index < 20:
            return "Very Calm"
        elif index < 35:
            return "Calm"
        elif index < 45:
            return "Slightly Calm"
        elif index < 55:
            return "Neutral"
        elif index < 65:
            return "Slightly Elevated"
        elif index < 75:
            return "Elevated"
        elif index < 85:
            return "High"
        else:
            return "Very High"
