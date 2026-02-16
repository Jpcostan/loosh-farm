"""JSON snapshot writer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .snapshot import MoodSnapshot

logger = logging.getLogger(__name__)


class JSONWriter:
    """Writes mood snapshot to a JSON file."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, snapshot: MoodSnapshot) -> Path:
        timestamp = snapshot.timestamp_utc.replace(":", "-").replace("T", "_").rstrip("Z")
        filename = f"loosh_snapshot_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("JSON snapshot written to %s", filepath)
        return filepath
