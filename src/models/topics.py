"""Trending topic and phrase extraction."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Common English stop words
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "as", "until",
    "while", "if", "then", "also", "about", "up", "out", "over", "after",
    "before", "between", "under", "again", "further", "once", "here",
    "there", "any", "into", "through", "during", "above", "below",
    "new", "said", "says", "like", "get", "got", "go", "going", "one",
    "two", "first", "last", "long", "great", "little", "just", "even",
    "still", "also", "back", "well", "way", "many", "much", "now", "old",
    "see", "time", "make", "know", "take", "come", "think", "look", "want",
    "give", "use", "find", "tell", "ask", "work", "seem", "feel", "try",
    "leave", "call", "good", "right", "man", "woman", "day", "year",
    "people", "part", "place", "case", "week", "company", "system",
    "program", "question", "home", "government", "number", "night",
    "point", "hand", "high", "keep", "let", "begin", "life", "thing",
    "dont", "doesnt", "didnt", "wont", "cant", "isnt", "arent", "wasnt",
    "amp", "per", "via", "etc", "vs", "re", "de",
})

_WORD_RE = re.compile(r"\b[a-z][a-z'-]*[a-z]\b|\b[a-z]\b")


@dataclass
class TopicResult:
    top_unigrams: list[tuple[str, int]] = field(default_factory=list)
    top_bigrams: list[tuple[str, int]] = field(default_factory=list)
    top_trigrams: list[tuple[str, int]] = field(default_factory=list)


class TopicExtractor:
    """Extract trending topics and phrases from collected text."""

    def __init__(self, top_n: int = 15, min_word_length: int = 3):
        self.top_n = top_n
        self.min_word_length = min_word_length

    def _tokenize(self, text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        return [
            w for w in words
            if w not in _STOP_WORDS and len(w) >= self.min_word_length
        ]

    def _get_ngrams(self, tokens: list[str], n: int) -> list[str]:
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def extract(self, texts: list[str]) -> TopicResult:
        """Extract top topics from a list of texts."""
        all_unigrams: Counter = Counter()
        all_bigrams: Counter = Counter()
        all_trigrams: Counter = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            all_unigrams.update(tokens)
            all_bigrams.update(self._get_ngrams(tokens, 2))
            all_trigrams.update(self._get_ngrams(tokens, 3))

        return TopicResult(
            top_unigrams=all_unigrams.most_common(self.top_n),
            top_bigrams=all_bigrams.most_common(self.top_n),
            top_trigrams=all_trigrams.most_common(self.top_n),
        )
