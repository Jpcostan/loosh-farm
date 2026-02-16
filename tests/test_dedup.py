"""Tests for deduplication."""

from src.collectors.base import CollectedItem
from src.processors.dedup import Deduplicator


def _make_item(text: str, title: str = "") -> CollectedItem:
    return CollectedItem(
        text=text,
        title=title,
        source_name="test",
        source_category="test",
    )


class TestDeduplicator:
    def setup_method(self):
        self.dedup = Deduplicator()

    def test_removes_exact_duplicates(self):
        items = [
            _make_item("This is a test sentence about the weather"),
            _make_item("This is a test sentence about the weather"),
            _make_item("Something completely different here today"),
        ]
        result = self.dedup.deduplicate(items)
        assert len(result) == 2

    def test_removes_near_duplicates(self):
        items = [
            _make_item("The president announced new policy changes today in Washington"),
            _make_item("The president announced new policy changes today in Washington DC"),
            _make_item("Stock markets crash amid global uncertainty and fears"),
        ]
        result = self.dedup.deduplicate(items)
        assert len(result) == 2

    def test_keeps_unique_items(self):
        items = [
            _make_item("First unique article about technology and AI"),
            _make_item("Second unique article about climate change issues"),
            _make_item("Third unique article about sports championships"),
        ]
        result = self.dedup.deduplicate(items)
        assert len(result) == 3

    def test_empty_input(self):
        result = self.dedup.deduplicate([])
        assert result == []

    def test_reset(self):
        items = [_make_item("Testing dedup reset functionality here")]
        self.dedup.deduplicate(items)
        assert len(self.dedup._seen_hashes) > 0
        self.dedup.reset()
        assert len(self.dedup._seen_hashes) == 0
