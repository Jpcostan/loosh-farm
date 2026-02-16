"""Reddit public JSON collector (no authentication required)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .base import BaseCollector, CollectedItem

logger = logging.getLogger(__name__)


class RedditCollector(BaseCollector):
    """Collects post titles and text from Reddit's public JSON API."""

    BASE_URL = "https://www.reddit.com"

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        reddit_config = config.get("reddit", {})
        self.subreddits = reddit_config.get("subreddits", [])
        self.sort = reddit_config.get("sort", "hot")
        self.limit = min(reddit_config.get("limit", 50), self.max_items)

    @property
    def source_category(self) -> str:
        return "reddit"

    def collect(self) -> list[CollectedItem]:
        items: list[CollectedItem] = []

        for subreddit in self.subreddits:
            url = f"{self.BASE_URL}/r/{subreddit}/{self.sort}.json"
            params = {"limit": self.limit, "raw_json": 1}
            logger.info("Fetching Reddit r/%s (%s)", subreddit, self.sort)

            resp = self._safe_get(url, params=params)
            if resp is None:
                self._rate_limit()
                continue

            try:
                data = resp.json()
            except ValueError:
                logger.warning("Invalid JSON from r/%s", subreddit)
                self._rate_limit()
                continue

            posts = data.get("data", {}).get("children", [])
            count = 0

            for post in posts:
                if count >= self.max_items:
                    break

                post_data = post.get("data", {})

                # Skip stickied/pinned posts
                if post_data.get("stickied"):
                    continue

                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")

                if not title and not selftext:
                    continue

                created_utc = post_data.get("created_utc")
                published = None
                if created_utc:
                    try:
                        published = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                    except (ValueError, OSError):
                        pass

                permalink = post_data.get("permalink", "")
                full_url = f"{self.BASE_URL}{permalink}" if permalink else None

                items.append(
                    CollectedItem(
                        text=selftext,
                        title=title,
                        source_name=f"r/{subreddit}",
                        source_category="reddit",
                        url=full_url,
                        published=published,
                        metadata={
                            "score": post_data.get("score", 0),
                            "num_comments": post_data.get("num_comments", 0),
                            "subreddit": subreddit,
                        },
                    )
                )
                count += 1

            logger.info("Collected %d items from r/%s", count, subreddit)
            self._rate_limit()

        logger.info("Total Reddit items collected: %d", len(items))
        return items
