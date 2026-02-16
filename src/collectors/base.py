"""Base collector interface and shared data model."""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class CollectedItem:
    """A single text item collected from a public source."""

    text: str
    source_name: str
    source_category: str  # e.g. "rss_news", "reddit", "rss_tech"
    url: Optional[str] = None
    title: Optional[str] = None
    published: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    @property
    def combined_text(self) -> str:
        """Title + text for analysis."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.text:
            parts.append(self.text)
        return " ".join(parts)


class BaseCollector(ABC):
    """Abstract base for all data collectors."""

    def __init__(
        self,
        config: dict,
        max_items: int = 100,
        timeout: int = 15,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        rate_limit_delay: float = 1.0,
    ):
        self.config = config
        self.max_items = max_items
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": "loosh-farm/1.0 (emotional-temperature-research; +https://github.com/loosh-farm)",
                "Accept": "application/json, application/xml, text/html, */*",
            }
        )
        return session

    def _rate_limit(self):
        """Sleep between requests to respect rate limits."""
        time.sleep(self.rate_limit_delay)

    def _safe_get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """GET with error handling."""
        try:
            resp = self._session.get(url, timeout=self.timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.warning("Request failed for %s: %s", url, e)
            return None

    @abstractmethod
    def collect(self) -> list[CollectedItem]:
        """Collect items from the source. Must be implemented by subclasses."""
        ...

    @property
    @abstractmethod
    def source_category(self) -> str:
        """Return the source category string (e.g. 'rss_news')."""
        ...
