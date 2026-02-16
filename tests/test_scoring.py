"""Tests for aggregator and loosh index."""

from src.collectors.base import CollectedItem
from src.models.sentiment import SentimentResult
from src.models.emotion import EmotionResult
from src.scoring.aggregator import Aggregator
from src.scoring.loosh_index import LooshIndexCalculator


def _make_item(category: str = "rss_news") -> CollectedItem:
    return CollectedItem(text="test", title="test", source_name="test", source_category=category)


def _make_sentiment(pos: float, neu: float, neg: float, compound: float) -> SentimentResult:
    return SentimentResult(positive=pos, neutral=neu, negative=neg, compound=compound)


def _make_emotion(**kwargs) -> EmotionResult:
    scores = {
        "joy": 0.125, "anger": 0.125, "fear": 0.125, "sadness": 0.125,
        "surprise": 0.125, "disgust": 0.125, "trust": 0.125, "anticipation": 0.125,
    }
    scores.update(kwargs)
    return EmotionResult(scores=scores, dominant_emotion=max(scores, key=scores.get))


class TestAggregator:
    def setup_method(self):
        self.aggregator = Aggregator({"source_weights": {"rss_news": 0.5, "reddit": 0.5}})

    def test_aggregates_sentiment(self):
        items = [_make_item(), _make_item()]
        sentiments = [
            _make_sentiment(0.6, 0.3, 0.1, 0.5),
            _make_sentiment(0.4, 0.3, 0.3, 0.1),
        ]
        emotions = [_make_emotion(), _make_emotion()]
        result = self.aggregator.aggregate(items, sentiments, emotions)
        assert result.sentiment["positive"] == 0.5
        assert result.items_analyzed == 2

    def test_empty_items(self):
        result = self.aggregator.aggregate([], [], [])
        assert result.items_analyzed == 0

    def test_source_breakdown(self):
        items = [_make_item("rss_news"), _make_item("reddit")]
        sentiments = [
            _make_sentiment(0.8, 0.1, 0.1, 0.7),
            _make_sentiment(0.2, 0.3, 0.5, -0.3),
        ]
        emotions = [_make_emotion(), _make_emotion()]
        result = self.aggregator.aggregate(items, sentiments, emotions)
        assert "rss_news" in result.source_breakdown
        assert "reddit" in result.source_breakdown


class TestLooshIndex:
    def setup_method(self):
        self.config = {
            "source_weights": {"rss_news": 1.0},
            "loosh_index": {
                "emotion_polarity": {
                    "joy": -1.0, "anger": 1.5, "fear": 1.8, "sadness": 1.2,
                    "surprise": 0.3, "disgust": 1.0, "trust": -0.8, "anticipation": 0.2,
                },
                "baseline": 50,
                "scale_min": 0,
                "scale_max": 100,
            },
        }
        self.calculator = LooshIndexCalculator(self.config)

    def test_uniform_emotions_returns_slightly_elevated(self):
        # With uniform emotions, the asymmetric polarity weights (negative emotions
        # weighted higher) intentionally produce a slightly-above-baseline score
        agg = Aggregator(self.config)
        items = [_make_item()]
        sentiments = [_make_sentiment(0.33, 0.34, 0.33, 0.0)]
        emotions = [_make_emotion()]
        result = agg.aggregate(items, sentiments, emotions)
        score = self.calculator.calculate(result)
        assert 50 <= score.index <= 70

    def test_high_negative_increases_index(self):
        agg = Aggregator(self.config)
        items = [_make_item()]
        sentiments = [_make_sentiment(0.1, 0.1, 0.8, -0.8)]
        emotions = [_make_emotion(anger=0.5, fear=0.3, joy=0.0, trust=0.0)]
        result = agg.aggregate(items, sentiments, emotions)
        score = self.calculator.calculate(result)
        assert score.index > 60

    def test_high_positive_decreases_index(self):
        agg = Aggregator(self.config)
        items = [_make_item()]
        sentiments = [_make_sentiment(0.8, 0.1, 0.1, 0.8)]
        emotions = [_make_emotion(joy=0.6, trust=0.3, anger=0.0, fear=0.0)]
        result = agg.aggregate(items, sentiments, emotions)
        score = self.calculator.calculate(result)
        assert score.index < 50

    def test_clamped_to_range(self):
        agg = Aggregator(self.config)
        items = [_make_item()]
        sentiments = [_make_sentiment(0.0, 0.0, 1.0, -1.0)]
        emotions = [_make_emotion(fear=0.9, anger=0.1, joy=0.0, trust=0.0,
                                   sadness=0.0, surprise=0.0, disgust=0.0, anticipation=0.0)]
        result = agg.aggregate(items, sentiments, emotions)
        score = self.calculator.calculate(result)
        assert 0 <= score.index <= 100

    def test_label_assigned(self):
        agg = Aggregator(self.config)
        items = [_make_item()]
        sentiments = [_make_sentiment(0.33, 0.34, 0.33, 0.0)]
        emotions = [_make_emotion()]
        result = agg.aggregate(items, sentiments, emotions)
        score = self.calculator.calculate(result)
        assert score.label != ""

    def test_no_data(self):
        from src.scoring.aggregator import AggregatedResult
        score = self.calculator.calculate(AggregatedResult())
        assert score.index == 50
        assert score.label == "No Data"
