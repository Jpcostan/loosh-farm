"""Tests for sentiment analyzer."""

import pytest
from src.models.sentiment import SentimentAnalyzer, SentimentResult


class TestSentimentAnalyzer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.analyzer = SentimentAnalyzer(mode="lightweight")

    def test_positive_text(self):
        result = self.analyzer.analyze("I love this amazing wonderful product!")
        assert result.positive > result.negative
        assert result.compound > 0

    def test_negative_text(self):
        result = self.analyzer.analyze("This is terrible, horrible, and disgusting.")
        assert result.negative > result.positive
        assert result.compound < 0

    def test_neutral_text(self):
        result = self.analyzer.analyze("The meeting is scheduled for 3 PM today.")
        assert result.neutral > 0.5

    def test_returns_sentiment_result(self):
        result = self.analyzer.analyze("Some text here for testing purposes.")
        assert isinstance(result, SentimentResult)
        assert 0 <= result.positive <= 1
        assert 0 <= result.neutral <= 1
        assert 0 <= result.negative <= 1

    def test_batch_analysis(self):
        texts = ["Great day!", "Terrible news.", "The sky is blue."]
        results = self.analyzer.analyze_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)
