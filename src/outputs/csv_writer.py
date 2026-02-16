"""CSV snapshot writer."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .snapshot import MoodSnapshot

logger = logging.getLogger(__name__)


class CSVWriter:
    """Writes mood snapshot to a CSV file (flat row format for time-series tracking)."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, snapshot: MoodSnapshot) -> Path:
        filepath = self.output_dir / "loosh_history.csv"
        is_new = not filepath.exists()

        row = {
            "timestamp_utc": snapshot.timestamp_utc,
            "items_analyzed": snapshot.items_analyzed,
            "sentiment_positive": snapshot.sentiment.get("positive", 0),
            "sentiment_neutral": snapshot.sentiment.get("neutral", 0),
            "sentiment_negative": snapshot.sentiment.get("negative", 0),
            "dominant_emotion": snapshot.dominant_emotion,
            "loosh_index": snapshot.loosh_index,
            "loosh_label": snapshot.loosh_label,
            "analysis_mode": snapshot.analysis_mode,
        }

        # Add emotion columns
        for emotion, score in snapshot.emotions.items():
            row[f"emotion_{emotion}"] = score

        fieldnames = list(row.keys())

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

        logger.info("CSV row appended to %s", filepath)
        return filepath
