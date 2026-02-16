"""Sentiment analysis using VADER (lightweight) and transformers (deep)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    positive: float
    neutral: float
    negative: float
    compound: float


class SentimentAnalyzer:
    """Dual-mode sentiment analyzer."""

    def __init__(self, mode: str = "lightweight"):
        self.mode = mode
        self._vader = None
        self._transformer_pipeline = None
        self._initialize()

    def _initialize(self):
        if self.mode == "lightweight":
            self._init_vader()
        else:
            self._init_transformer()

    def _init_vader(self):
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        except ImportError:
            logger.error("vaderSentiment not installed. Run: pip install vaderSentiment")
            raise

    def _init_transformer(self):
        try:
            from transformers import pipeline
            self._transformer_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )
            logger.info("Transformer sentiment pipeline initialized")
        except ImportError:
            logger.warning("transformers not available, falling back to VADER")
            self.mode = "lightweight"
            self._init_vader()

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        if self.mode == "lightweight":
            return self._analyze_vader(text)
        return self._analyze_transformer(text)

    def _analyze_vader(self, text: str) -> SentimentResult:
        scores = self._vader.polarity_scores(text)
        return SentimentResult(
            positive=scores["pos"],
            neutral=scores["neu"],
            negative=scores["neg"],
            compound=scores["compound"],
        )

    def _analyze_transformer(self, text: str) -> SentimentResult:
        # Truncate for transformer input
        truncated = text[:512]
        result = self._transformer_pipeline(truncated)[0]
        label = result["label"]
        score = result["score"]

        if label == "POSITIVE":
            return SentimentResult(
                positive=score,
                neutral=0.0,
                negative=1.0 - score,
                compound=score * 2 - 1,  # Map to [-1, 1]
            )
        else:
            return SentimentResult(
                positive=1.0 - score,
                neutral=0.0,
                negative=score,
                compound=-(score * 2 - 1),
            )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze a batch of texts."""
        return [self.analyze(t) for t in texts]
