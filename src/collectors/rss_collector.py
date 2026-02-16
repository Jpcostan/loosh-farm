"""RSS/Atom feed collector using feedparser."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import mktime

import feedparser

from .base import BaseCollector, CollectedItem

logger = logging.getLogger(__name__)


class RSSCollector(BaseCollector):
    """Collects text from RSS/Atom feeds."""

    def __init__(self, config: dict, category: str = "rss_news", **kwargs):
        super().__init__(config, **kwargs)
        self._category = category
        self._feeds = self._resolve_feeds()

    def _resolve_feeds(self) -> list[dict]:
        """Pull feed URLs from config based on category."""
        rss_config = self.config.get("rss_feeds", {})
        # Map category to config key
        key_map = {
            "rss_news": "news",
            "rss_tech": "tech",
            "rss_world": "world",
        }
        key = key_map.get(self._category, self._category)
        feeds = rss_config.get(key, [])
        if not feeds:
            logger.warning("No feeds configured for category '%s'", self._category)
        return feeds

    @property
    def source_category(self) -> str:
        return self._category

    def collect(self) -> list[CollectedItem]:
        items: list[CollectedItem] = []

        for feed_info in self._feeds:
            url = feed_info["url"]
            name = feed_info.get("name", url)
            logger.info("Fetching RSS feed: %s", name)

            try:
                parsed = feedparser.parse(url)
            except Exception as e:
                logger.warning("Failed to parse feed %s: %s", name, e)
                self._rate_limit()
                continue

            if parsed.bozo and not parsed.entries:
                logger.warning("Feed %s returned no entries (bozo=%s)", name, parsed.bozo_exception)
                self._rate_limit()
                continue

            count = 0
            for entry in parsed.entries:
                if count >= self.max_items:
                    break

                title = entry.get("title", "")
                # Try summary, then description, then content
                text = ""
                if "summary" in entry:
                    text = entry.summary
                elif "description" in entry:
                    text = entry.description
                elif "content" in entry and entry.content:
                    text = entry.content[0].get("value", "")

                if not title and not text:
                    continue

                published = None
                if "published_parsed" in entry and entry.published_parsed:
                    try:
                        published = datetime.fromtimestamp(
                            mktime(entry.published_parsed), tz=timezone.utc
                        )
                    except (ValueError, OverflowError):
                        pass

                items.append(
                    CollectedItem(
                        text=text,
                        title=title,
                        source_name=name,
                        source_category=self._category,
                        url=entry.get("link"),
                        published=published,
                        metadata={"feed_url": url},
                    )
                )
                count += 1

            logger.info("Collected %d items from %s", count, name)
            self._rate_limit()

        logger.info("Total RSS items collected for '%s': %d", self._category, len(items))
        return items
