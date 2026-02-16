"""Processing pipeline that chains cleaner and deduplicator."""

from __future__ import annotations

import logging

from src.collectors.base import CollectedItem
from .cleaner import TextCleaner
from .dedup import Deduplicator

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Orchestrates text cleaning and deduplication."""

    def __init__(self, config: dict | None = None):
        self.cleaner = TextCleaner()
        self.dedup = Deduplicator()

    def process(self, items: list[CollectedItem]) -> list[CollectedItem]:
        """Clean text and remove duplicates."""
        logger.info("Processing pipeline: %d raw items", len(items))

        # Clean text in each item
        cleaned_items: list[CollectedItem] = []
        for item in items:
            cleaned_text = self.cleaner.clean(item.text)
            cleaned_title = self.cleaner.clean(item.title) if item.title else item.title

            # Keep the item if at least title or text survived cleaning
            if cleaned_text is None and (cleaned_title is None or not cleaned_title):
                continue

            item.text = cleaned_text or ""
            if cleaned_title is not None:
                item.title = cleaned_title
            cleaned_items.append(item)

        logger.info("After cleaning: %d items", len(cleaned_items))

        # Deduplicate
        unique_items = self.dedup.deduplicate(cleaned_items)

        logger.info("Pipeline output: %d items", len(unique_items))
        return unique_items
