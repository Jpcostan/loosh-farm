"""Tests for emotion analyzer."""

from src.models.emotion import EmotionAnalyzer, EmotionResult, EMOTION_CATEGORIES


class TestEmotionAnalyzer:
    def setup_method(self):
        self.analyzer = EmotionAnalyzer(mode="lightweight")

    def test_detects_joy(self):
        result = self.analyzer.analyze("I am so happy and delighted, this is wonderful!")
        assert result.scores["joy"] > 0

    def test_detects_anger(self):
        result = self.analyzer.analyze("I am furious and outraged by this hostile attack!")
        assert result.scores["anger"] > 0

    def test_detects_fear(self):
        result = self.analyzer.analyze("The crisis is terrifying, panic and danger everywhere!")
        assert result.scores["fear"] > 0

    def test_all_categories_present(self):
        result = self.analyzer.analyze("Today we celebrate with joy and anticipation for the future.")
        for cat in EMOTION_CATEGORIES:
            assert cat in result.scores

    def test_scores_sum_approximately_one(self):
        result = self.analyzer.analyze("A mix of emotions in this complex situation we face.")
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.01

    def test_returns_emotion_result(self):
        result = self.analyzer.analyze("Some text")
        assert isinstance(result, EmotionResult)

    def test_batch_analysis(self):
        texts = ["Happy day!", "Angry protest!", "Fearful crisis!"]
        results = self.analyzer.analyze_batch(texts)
        assert len(results) == 3
