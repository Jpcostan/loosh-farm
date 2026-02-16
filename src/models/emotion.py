"""Emotion classification using NRC lexicon (lightweight) or transformers (deep)."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

EMOTION_CATEGORIES = [
    "joy", "anger", "fear", "sadness",
    "surprise", "disgust", "trust", "anticipation",
]


@dataclass
class EmotionResult:
    scores: dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = "neutral"


# Curated emotion lexicon (subset of NRC-style words mapped to emotions)
# This avoids requiring a large external file while still being useful
_EMOTION_LEXICON: dict[str, list[str]] = {
    "joy": [
        "happy", "love", "wonderful", "great", "amazing", "excellent", "beautiful",
        "celebrate", "delight", "fantastic", "glad", "cheerful", "pleased", "joy",
        "thrilled", "ecstatic", "bliss", "elated", "grateful", "optimistic",
        "success", "win", "triumph", "paradise", "brilliant", "superb", "awesome",
        "hope", "proud", "laugh", "smile", "fun", "enjoy", "peace", "comfort",
        "bright", "radiant", "warm", "kind", "generous", "uplift", "inspire",
    ],
    "anger": [
        "angry", "furious", "outrage", "rage", "hate", "hostile", "violent",
        "attack", "destroy", "war", "fight", "aggression", "resent", "bitter",
        "irritate", "annoy", "frustrate", "provoke", "enrage", "infuriate",
        "condemn", "denounce", "protest", "revolt", "clash", "conflict",
        "abuse", "exploit", "corruption", "injustice", "oppression", "tyranny",
        "bully", "harass", "threaten", "demand", "blame", "accuse", "betray",
    ],
    "fear": [
        "afraid", "fear", "terror", "panic", "scare", "dread", "horror",
        "threat", "danger", "risk", "crisis", "emergency", "alarm", "anxiety",
        "worry", "nervous", "concern", "caution", "warn", "hazard", "peril",
        "catastrophe", "disaster", "devastation", "collapse", "doom", "chaos",
        "plague", "epidemic", "pandemic", "vulnerable", "helpless", "desperate",
        "uncertain", "unstable", "volatile", "flee", "escape", "survive",
    ],
    "sadness": [
        "sad", "grief", "sorrow", "mourn", "tragic", "loss", "suffer",
        "pain", "hurt", "cry", "tears", "misery", "despair", "hopeless",
        "depress", "lonely", "abandon", "neglect", "regret", "disappoint",
        "fail", "defeat", "decline", "deteriorate", "poverty", "famine",
        "death", "die", "kill", "victim", "casualty", "funeral", "memorial",
        "devastate", "heartbreak", "anguish", "agony", "lament", "woe",
    ],
    "surprise": [
        "surprise", "shock", "astonish", "amaze", "stun", "unexpected",
        "sudden", "breaking", "unprecedented", "remarkable", "extraordinary",
        "incredible", "unbelievable", "dramatic", "revelation", "discover",
        "breakthrough", "miracle", "twist", "upset", "bombshell", "jaw-dropping",
        "alert", "flash", "interrupt", "announce", "reveal", "unveil",
    ],
    "disgust": [
        "disgust", "repulsive", "gross", "vile", "sick", "revolting",
        "nasty", "foul", "corrupt", "scandal", "fraud", "cheat", "deceive",
        "toxic", "contaminate", "pollute", "waste", "filth", "sleaze",
        "hypocrisy", "shameful", "despicable", "contempt", "loathe", "abhor",
        "obscene", "vulgar", "offensive", "grotesque", "perverse",
    ],
    "trust": [
        "trust", "reliable", "honest", "integrity", "loyal", "faithful",
        "secure", "safe", "stable", "protect", "support", "ally", "partner",
        "cooperate", "unite", "together", "solidarity", "agreement", "treaty",
        "peace", "harmony", "respect", "honor", "commit", "promise", "pledge",
        "transparent", "accountable", "responsible", "fair", "just", "equitable",
    ],
    "anticipation": [
        "expect", "anticipate", "predict", "forecast", "plan", "prepare",
        "upcoming", "future", "tomorrow", "next", "soon", "await", "pending",
        "potential", "prospect", "opportunity", "launch", "release", "debut",
        "election", "vote", "decision", "deadline", "schedule", "target",
        "goal", "ambition", "aspire", "strategy", "initiative", "proposal",
    ],
}

# Flatten for fast lookups: word -> list of emotions
_WORD_TO_EMOTIONS: dict[str, list[str]] = {}
for _emotion, _words in _EMOTION_LEXICON.items():
    for _word in _words:
        _WORD_TO_EMOTIONS.setdefault(_word, []).append(_emotion)

_WORD_RE = re.compile(r"\b[a-z]+\b")


class EmotionAnalyzer:
    """Dual-mode emotion analyzer."""

    def __init__(self, mode: str = "lightweight"):
        self.mode = mode
        self._transformer_pipeline = None
        if mode == "deep":
            self._init_transformer()

    def _init_transformer(self):
        try:
            from transformers import pipeline
            self._transformer_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                truncation=True,
                max_length=512,
            )
            logger.info("Transformer emotion pipeline initialized")
        except ImportError:
            logger.warning("transformers not available, falling back to lexicon")
            self.mode = "lightweight"

    def analyze(self, text: str) -> EmotionResult:
        if self.mode == "lightweight":
            return self._analyze_lexicon(text)
        return self._analyze_transformer(text)

    def _analyze_lexicon(self, text: str) -> EmotionResult:
        """Lexicon-based emotion detection."""
        words = _WORD_RE.findall(text.lower())
        emotion_counts: Counter = Counter()
        total_hits = 0

        for word in words:
            if word in _WORD_TO_EMOTIONS:
                for emotion in _WORD_TO_EMOTIONS[word]:
                    emotion_counts[emotion] += 1
                    total_hits += 1

        # Normalize to probabilities
        scores = {}
        for emotion in EMOTION_CATEGORIES:
            if total_hits > 0:
                scores[emotion] = round(emotion_counts.get(emotion, 0) / total_hits, 4)
            else:
                scores[emotion] = round(1.0 / len(EMOTION_CATEGORIES), 4)

        dominant = max(scores, key=scores.get) if total_hits > 0 else "neutral"

        return EmotionResult(scores=scores, dominant_emotion=dominant)

    def _analyze_transformer(self, text: str) -> EmotionResult:
        """Transformer-based emotion detection."""
        truncated = text[:512]
        results = self._transformer_pipeline(truncated)[0]

        # Map model labels to our categories
        label_map = {
            "joy": "joy",
            "anger": "anger",
            "fear": "fear",
            "sadness": "sadness",
            "surprise": "surprise",
            "disgust": "disgust",
            "neutral": None,
        }

        scores = {e: 0.0 for e in EMOTION_CATEGORIES}
        for r in results:
            mapped = label_map.get(r["label"])
            if mapped and mapped in scores:
                scores[mapped] = round(r["score"], 4)

        # Fill unmapped categories
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 4) for k, v in scores.items()}
        else:
            scores = {k: round(1.0 / len(EMOTION_CATEGORIES), 4) for k in EMOTION_CATEGORIES}

        dominant = max(scores, key=scores.get)
        return EmotionResult(scores=scores, dominant_emotion=dominant)

    def analyze_batch(self, texts: list[str]) -> list[EmotionResult]:
        return [self.analyze(t) for t in texts]
