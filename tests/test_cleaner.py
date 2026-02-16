"""Tests for text cleaner."""

from src.processors.cleaner import TextCleaner


class TestTextCleaner:
    def setup_method(self):
        self.cleaner = TextCleaner(min_length=5)

    def test_strips_html_tags(self):
        result = self.cleaner.clean("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_urls(self):
        result = self.cleaner.clean("Check https://example.com for details today")
        assert "https" not in result
        assert "example.com" not in result
        assert "details" in result

    def test_decodes_html_entities(self):
        result = self.cleaner.clean("Tom &amp; Jerry are great characters")
        assert "&amp;" not in result
        assert "Tom" in result

    def test_returns_none_for_short_text(self):
        result = self.cleaner.clean("Hi")
        assert result is None

    def test_returns_none_for_empty_text(self):
        assert self.cleaner.clean("") is None
        assert self.cleaner.clean(None) is None

    def test_normalizes_whitespace(self):
        result = self.cleaner.clean("Hello    world   this   is   a   test")
        assert "  " not in result

    def test_truncates_long_text(self):
        cleaner = TextCleaner(max_length=50)
        long_text = "a " * 100
        result = cleaner.clean(long_text)
        assert len(result) <= 50

    def test_clean_batch(self):
        texts = ["Hello world today", "Hi", "<p>Good morning everyone</p>", ""]
        results = self.cleaner.clean_batch(texts)
        assert len(results) == 2  # "Hi" and "" filtered out
