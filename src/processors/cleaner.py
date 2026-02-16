"""Text cleaning and normalization."""

from __future__ import annotations

import html
import re
import logging

logger = logging.getLogger(__name__)

# Pre-compiled patterns for performance
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s.,!?;:'\"-]", re.UNICODE)


class TextCleaner:
    """Cleans and normalizes raw text from collectors."""

    def __init__(self, min_length: int = 10, max_length: int = 5000):
        self.min_length = min_length
        self.max_length = max_length

    def clean(self, text: str) -> str | None:
        """Clean a single text string. Returns None if text is too short after cleaning."""
        if not text:
            return None

        # Decode HTML entities
        text = html.unescape(text)

        # Strip HTML tags
        text = _HTML_TAG_RE.sub(" ", text)

        # Remove URLs
        text = _URL_RE.sub("", text)

        # Remove special characters but keep punctuation useful for sentiment
        text = _SPECIAL_CHARS_RE.sub(" ", text)

        # Normalize whitespace
        text = _MULTI_SPACE_RE.sub(" ", text).strip()

        # Truncate if too long
        if len(text) > self.max_length:
            text = text[: self.max_length]

        # Check minimum length
        if len(text) < self.min_length:
            return None

        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a batch of texts, filtering out None results."""
        results = []
        for t in texts:
            cleaned = self.clean(t)
            if cleaned is not None:
                results.append(cleaned)
        return results
