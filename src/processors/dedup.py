"""Deduplication via content hashing and similarity detection."""

from __future__ import annotations

import hashlib
import logging
import re

from src.collectors.base import CollectedItem

logger = logging.getLogger(__name__)

_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]")
_MULTI_SPACE_RE = re.compile(r"\s+")


class Deduplicator:
    """Removes duplicate items using exact and near-duplicate detection."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self._seen_hashes: set[str] = set()

    def _normalize_for_hash(self, text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace for hashing."""
        text = text.lower()
        text = _NORMALIZE_RE.sub(" ", text)
        text = _MULTI_SPACE_RE.sub(" ", text).strip()
        return text

    def _content_hash(self, text: str) -> str:
        normalized = self._normalize_for_hash(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _ngram_fingerprint(self, text: str, n: int = 3) -> set[str]:
        """Generate character n-gram set for near-duplicate detection."""
        normalized = self._normalize_for_hash(text)
        if len(normalized) < n:
            return {normalized}
        return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}

    def _jaccard_similarity(self, set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def deduplicate(self, items: list[CollectedItem]) -> list[CollectedItem]:
        """Remove exact and near-duplicate items."""
        unique: list[CollectedItem] = []
        fingerprints: list[set[str]] = []
        duplicates_removed = 0

        for item in items:
            combined = item.combined_text
            if not combined.strip():
                continue

            # Exact duplicate check
            content_hash = self._content_hash(combined)
            if content_hash in self._seen_hashes:
                duplicates_removed += 1
                continue

            # Near-duplicate check via n-gram Jaccard similarity
            fp = self._ngram_fingerprint(combined)
            is_near_dup = False
            for existing_fp in fingerprints:
                if self._jaccard_similarity(fp, existing_fp) >= self.similarity_threshold:
                    is_near_dup = True
                    duplicates_removed += 1
                    break

            if is_near_dup:
                continue

            self._seen_hashes.add(content_hash)
            fingerprints.append(fp)
            unique.append(item)

        logger.info(
            "Deduplication: %d items in, %d unique, %d duplicates removed",
            len(items),
            len(unique),
            duplicates_removed,
        )
        return unique

    def reset(self):
        """Clear seen hashes for a fresh run."""
        self._seen_hashes.clear()
