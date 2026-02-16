"""Tests for topic extraction."""

from src.models.topics import TopicExtractor


class TestTopicExtractor:
    def setup_method(self):
        self.extractor = TopicExtractor(top_n=5)

    def test_extracts_unigrams(self):
        texts = [
            "climate change affects global weather patterns significantly",
            "climate change is the biggest threat facing humanity today",
            "global warming and climate change require immediate action now",
        ]
        result = self.extractor.extract(texts)
        top_words = [w for w, _ in result.top_unigrams]
        assert "climate" in top_words
        assert "change" in top_words

    def test_extracts_bigrams(self):
        texts = [
            "artificial intelligence is transforming modern technology rapidly",
            "artificial intelligence applications grow every single day",
            "the future of artificial intelligence looks very promising",
        ]
        result = self.extractor.extract(texts)
        top_bigrams = [b for b, _ in result.top_bigrams]
        assert "artificial intelligence" in top_bigrams

    def test_filters_stop_words(self):
        texts = ["the and or but is are was were will would could should"]
        result = self.extractor.extract(texts)
        assert len(result.top_unigrams) == 0

    def test_empty_input(self):
        result = self.extractor.extract([])
        assert result.top_unigrams == []
        assert result.top_bigrams == []
