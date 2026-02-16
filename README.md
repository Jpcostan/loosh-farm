<div align="center">

# LOOSH FARM

### The Emotional Temperature of the Internet

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen?style=for-the-badge)](#testing)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-blue?style=for-the-badge)](https://peps.python.org/pep-0008/)

**A collect + infer + snapshot CLI that estimates the public emotional state of the internet at the moment it's run.**

*Loosh* is used here as a tongue-in-cheek label for **aggregate human emotional output** &mdash; collected from public text signals across the web, analyzed through NLP, and distilled into a single timestamped mood snapshot.

---

[Getting Started](#getting-started) &bull; [How It Works](#how-it-works) &bull; [Architecture](#architecture) &bull; [Configuration](#configuration) &bull; [Example Output](#example-output) &bull; [Deep Mode](#deep-mode)

</div>

---

## What It Does

Every time you run `loosh-farm`, it reaches across the public internet &mdash; pulling from **21+ live data sources** including major news outlets, tech feeds, and community forums &mdash; then pipes everything through a multi-stage NLP pipeline to produce:

| Output | Description |
|--------|-------------|
| **Sentiment** | Positive / neutral / negative ratio across all collected text |
| **Emotion Distribution** | Plutchik's 8 basic emotions &mdash; joy, anger, fear, sadness, surprise, disgust, trust, anticipation |
| **Loosh Index** | A single 0&ndash;100 score representing global emotional intensity (higher = more intense/negative) |
| **Trending Topics** | Top words, bigrams, and trigrams extracted from the corpus |
| **Source Breakdown** | Per-source sentiment and emotion with configurable weights |

Results are written to **JSON** (timestamped snapshots), **CSV** (append-mode time-series), and a **rich terminal dashboard**.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Internet connection (for live data collection)

### Install

```bash
git clone https://github.com/jpcostan/loosh-farm.git
cd loosh-farm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python -m src.main run
```

That's it. The pipeline will collect from all configured sources, analyze the text, and output a full mood snapshot.

### Other Commands

```bash
# List all configured data sources
python -m src.main sources

# Validate config and check dependencies
python -m src.main validate

# Run with verbose logging
python -m src.main run -v

# Skip specific outputs
python -m src.main run --no-csv --no-json

# Use deep analysis mode (requires transformers + torch)
python -m src.main run --mode deep
```

---

## How It Works

The pipeline executes six stages in sequence:

```
                    +------------------+
                    |   COLLECT        |    RSS feeds, Reddit JSON API
                    |   21+ sources    |    Rate-limited, retries, sessions
                    +--------+---------+
                             |
                    +--------v---------+
                    |   CLEAN          |    HTML stripping, URL removal
                    |   Normalize      |    Entity decoding, whitespace
                    +--------+---------+
                             |
                    +--------v---------+
                    |   DEDUPLICATE    |    SHA-256 exact match
                    |                  |    N-gram Jaccard similarity
                    +--------+---------+
                             |
                    +--------v---------+
                    |   ANALYZE        |    VADER / DistilBERT sentiment
                    |   NLP            |    Lexicon / DistilRoBERTa emotion
                    +--------+---------+    N-gram topic extraction
                             |
                    +--------v---------+
                    |   AGGREGATE      |    Per-source + global averages
                    |   + SCORE        |    Weighted Loosh Index formula
                    +--------+---------+
                             |
                    +--------v---------+
                    |   OUTPUT         |    JSON snapshot
                    |                  |    CSV time-series
                    |                  |    Rich terminal dashboard
                    +------------------+
```

### The Loosh Index Formula

The Loosh Index is a composite score from 0 to 100:

1. **Emotion Signal** &mdash; Each of the 8 emotions is multiplied by a polarity weight (negative emotions like fear and anger push the score up; positive emotions like joy and trust pull it down)
2. **Sentiment Amplifier** &mdash; The negativity-to-positivity ratio amplifies the signal
3. **Source Weighting** &mdash; Per-source signals are combined using configurable weights
4. **Clamped** to [0, 100] with a baseline of 50 (neutral)

| Range | Label |
|-------|-------|
| 0 &ndash; 19 | Very Calm |
| 20 &ndash; 34 | Calm |
| 35 &ndash; 44 | Slightly Calm |
| 45 &ndash; 54 | Neutral |
| 55 &ndash; 64 | Slightly Elevated |
| 65 &ndash; 74 | Elevated |
| 75 &ndash; 84 | High |
| 85 &ndash; 100 | Very High |

---

## Architecture

```
loosh-farm/
├── config.yaml                  # Source URLs, weights, parameters
├── pyproject.toml               # Package metadata + optional extras
├── requirements.txt
│
├── src/
│   ├── main.py                  # Click CLI (run / sources / validate)
│   │
│   ├── collectors/
│   │   ├── base.py              # BaseCollector — sessions, retries, rate limiting
│   │   ├── rss_collector.py     # RSS/Atom via feedparser (13 feeds)
│   │   └── reddit_collector.py  # Reddit public JSON API (8 subreddits)
│   │
│   ├── processors/
│   │   ├── cleaner.py           # HTML strip, URL removal, normalization
│   │   ├── dedup.py             # SHA-256 + n-gram Jaccard deduplication
│   │   └── pipeline.py          # Chains cleaner → deduplicator
│   │
│   ├── models/
│   │   ├── sentiment.py         # VADER (lightweight) / DistilBERT (deep)
│   │   ├── emotion.py           # 300+ word lexicon / DistilRoBERTa (deep)
│   │   └── topics.py            # N-gram extraction with stop word filtering
│   │
│   ├── scoring/
│   │   ├── aggregator.py        # Per-source + global averaging
│   │   └── loosh_index.py       # Weighted composite 0–100 index
│   │
│   └── outputs/
│       ├── snapshot.py          # MoodSnapshot dataclass
│       ├── json_writer.py       # Timestamped JSON files
│       ├── csv_writer.py        # Append-mode CSV for history tracking
│       └── console_writer.py    # Rich terminal dashboard
│
├── output/                      # Generated snapshots (gitignored)
└── tests/                       # 38 unit tests
```

### Design Principles

- **Pluggable collectors** &mdash; Add a new source by extending `BaseCollector` with a single `collect()` method
- **Dual-mode analysis** &mdash; Lightweight (VADER + lexicon, no GPU) or deep (HuggingFace transformers)
- **Defensive networking** &mdash; Exponential backoff retries, rate limiting, per-source error isolation
- **Near-duplicate detection** &mdash; Character n-gram Jaccard similarity on top of exact SHA-256 hashing
- **Append-mode CSV** &mdash; Each run adds a row, building a time-series for trend analysis

---

## Data Sources

### RSS / Atom Feeds (13 feeds across 3 categories)

| Category | Sources |
|----------|---------|
| **News** | NYT Homepage, BBC News, CNN, Reuters, ABC News, Al Jazeera |
| **Tech** | Hacker News, r/technology RSS, Ars Technica, The Verge |
| **World** | BBC World, NYT World, The Guardian World |

### Reddit Public JSON API (8 subreddits)

`r/worldnews` &bull; `r/news` &bull; `r/politics` &bull; `r/UpliftingNews` &bull; `r/technology` &bull; `r/science` &bull; `r/AskReddit` &bull; `r/todayilearned`

> All sources use public APIs and feeds. No authentication tokens required for the default configuration.

---

## Configuration

All settings live in [`config.yaml`](config.yaml). Key options:

```yaml
# Switch between fast (VADER) and accurate (transformers) analysis
analysis_mode: lightweight   # or "deep"

# How much each source category contributes to the Loosh Index
source_weights:
  rss_news: 0.35
  reddit:   0.30
  rss_world: 0.20
  rss_tech: 0.15

# Collection limits
max_items_per_source: 100
request_timeout: 15
rate_limit_delay: 1.0

# Emotion polarity weights (negative emotions push the index up)
loosh_index:
  emotion_polarity:
    fear:   1.8
    anger:  1.5
    sadness: 1.2
    disgust: 1.0
    surprise: 0.3
    anticipation: 0.2
    trust: -0.8
    joy:   -1.0
```

You can add new RSS feeds or subreddits directly in the config without touching any code.

---

## Example Output

### Terminal Dashboard

```
╔══════════════════════════════════════════════════════════════════╗
║ LOOSH FARM — Global Emotional Temperature                      ║
╚══════════════════════════════════════════════════════════════════╝
  Timestamp:  2026-02-16T05:44:22Z
  Items:      694
  Mode:       lightweight

╭──────────────────────── Loosh Index ─────────────────────────╮
│   62.0 / 100  [Slightly Elevated]                            │
╰──────────────────────────────────────────────────────────────╯

╭──────────────────────── Sentiment ───────────────────────────╮
│   Positive    8.3%   █░░░░░░░░░░░░░░░░░░░                   │
│   Neutral    81.8%   ████████████████░░░░                    │
│   Negative    9.9%   █░░░░░░░░░░░░░░░░░░░                   │
╰──────────────────────────────────────────────────────────────╯

╭──────────────────────── Emotions ────────────────────────────╮
│   Anticipation  16.5%   ███░░░░░░░░░░░░░░░░░                │
│   Anger         14.4%   ██░░░░░░░░░░░░░░░░░░                │
│   Trust         13.2%   ██░░░░░░░░░░░░░░░░░░                │
│   Joy           13.0%   ██░░░░░░░░░░░░░░░░░░                │
│   Fear          12.2%   ██░░░░░░░░░░░░░░░░░░                │
│   Sadness       12.0%   ██░░░░░░░░░░░░░░░░░░                │
│   Surprise      10.1%   ██░░░░░░░░░░░░░░░░░░                │
│   Disgust        8.6%   █░░░░░░░░░░░░░░░░░░░                │
╰──────────────────────────────────────────────────────────────╯

╭──────────────────────── Sources ─────────────────────────────╮
│   rss_news     113    Pos 9.3%    Neg 7.7%    Weight 35%     │
│   rss_tech      85    Pos 6.3%    Neg 6.5%    Weight 15%     │
│   rss_world    114    Pos 7.9%    Neg 10.9%   Weight 20%     │
│   reddit       382    Pos 8.5%    Neg 11.0%   Weight 30%     │
╰──────────────────────────────────────────────────────────────╯
```

### JSON Snapshot

<details>
<summary>Click to expand full JSON output</summary>

```json
{
  "timestamp_utc": "2026-02-16T05:44:22Z",
  "items_analyzed": 694,
  "sentiment": {
    "positive": 0.0827,
    "neutral": 0.8184,
    "negative": 0.099
  },
  "emotions": {
    "joy": 0.1298,
    "anger": 0.1445,
    "fear": 0.1218,
    "sadness": 0.1205,
    "surprise": 0.1008,
    "disgust": 0.0861,
    "trust": 0.132,
    "anticipation": 0.1646
  },
  "dominant_emotion": "anticipation",
  "loosh_index": 62.0,
  "loosh_label": "Slightly Elevated",
  "trending_topics": {
    "words": [["trump", 63], ["world", 37], ["state", 33]],
    "bigrams": [["epstein files", 13], ["president trump", 11], ["winter olympics", 11]],
    "trigrams": [["munich security conference", 7], ["dart frog toxin", 5]]
  },
  "source_breakdown": {
    "rss_news":  { "count": 113, "weight": 0.35 },
    "rss_tech":  { "count": 85,  "weight": 0.15 },
    "rss_world": { "count": 114, "weight": 0.2 },
    "reddit":    { "count": 382, "weight": 0.3 }
  },
  "analysis_mode": "lightweight"
}
```

</details>

### CSV Time-Series

Each run appends a row to `output/loosh_history.csv`, enabling historical trend tracking:

```
timestamp_utc,items_analyzed,sentiment_positive,sentiment_neutral,sentiment_negative,dominant_emotion,loosh_index,loosh_label,...
2026-02-16T05:44:22Z,694,0.0827,0.8184,0.099,anticipation,62.0,Slightly Elevated,...
```

---

## Deep Mode

For more accurate analysis at the cost of speed (and a GPU if available), install the transformer dependencies:

```bash
pip install transformers torch
```

Then run with the `--mode deep` flag:

```bash
python -m src.main run --mode deep
```

| | Lightweight | Deep |
|---|---|---|
| **Sentiment** | VADER (lexicon-based) | DistilBERT (fine-tuned SST-2) |
| **Emotion** | 300+ word curated lexicon | DistilRoBERTa (j-hartmann) |
| **Speed** | ~2 seconds for 700 items | ~60 seconds for 700 items |
| **GPU** | Not needed | Optional (auto-detected) |

---

## Testing

```bash
python -m pytest tests/ -v
```

```
tests/test_cleaner.py    ........                 [ 21%]
tests/test_dedup.py      .....                    [ 34%]
tests/test_emotion.py    .......                  [ 52%]
tests/test_scoring.py    ......                   [ 68%]
tests/test_sentiment.py  .....                    [ 81%]
tests/test_topics.py     ....                     [100%]

38 passed in 0.25s
```

Test coverage includes: text cleaning, HTML/URL stripping, exact and near-duplicate detection, emotion scoring, sentiment analysis, topic extraction, aggregation, and Loosh Index calculation.

---

## Extending

### Add a New Collector

Create a new file in `src/collectors/` and extend `BaseCollector`:

```python
from src.collectors.base import BaseCollector, CollectedItem

class MyCollector(BaseCollector):
    @property
    def source_category(self) -> str:
        return "my_source"

    def collect(self) -> list[CollectedItem]:
        # Use self._safe_get(url) for HTTP requests (includes retries)
        # Use self._rate_limit() between requests
        # Return a list of CollectedItem objects
        ...
```

Then register it in `src/main.py` alongside the existing collectors.

### Add a New RSS Feed

No code changes needed &mdash; just add an entry in `config.yaml`:

```yaml
rss_feeds:
  news:
    - url: "https://example.com/feed.xml"
      name: "My Feed"
```

---

## Ethics & Data Use

This project **only** collects from:

- Public RSS/Atom feeds
- Public JSON APIs (no authentication)
- Publicly accessible content permitted by each platform's Terms of Service

No scraping, no login-walled content, no private data. You are responsible for complying with each platform's terms when configuring sources.

---

## Roadmap

- [ ] Dashboard visualization (web UI)
- [ ] Historical trend charts from CSV data
- [ ] LDA / BERTopic topic modeling
- [ ] Language detection and multilingual support
- [ ] GitHub Actions for scheduled automated runs
- [ ] Webhook / Slack notifications for index spikes
- [ ] Additional collectors (Mastodon, Bluesky, public Telegram)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| CLI | [Click](https://click.palletsprojects.com/) |
| Terminal UI | [Rich](https://rich.readthedocs.io/) |
| RSS Parsing | [feedparser](https://feedparser.readthedocs.io/) |
| HTTP | [Requests](https://requests.readthedocs.io/) + urllib3 retry |
| Sentiment | [VADER](https://github.com/cjhutto/vaderSentiment) / [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) |
| Emotion | Curated lexicon / [DistilRoBERTa](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) |
| Config | [PyYAML](https://pyyaml.org/) |
| Testing | [pytest](https://pytest.org/) |

---

## License

[MIT](LICENSE)

---

<div align="center">

*This is an experimental mood estimation tool using imperfect public signals.*
*Not scientific. Not predictive. Built for exploration and curiosity.*

</div>
