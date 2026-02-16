"""loosh-farm CLI — collect, infer, snapshot."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.collectors.rss_collector import RSSCollector
from src.collectors.reddit_collector import RedditCollector
from src.collectors.base import CollectedItem
from src.processors.pipeline import ProcessingPipeline
from src.models.sentiment import SentimentAnalyzer
from src.models.emotion import EmotionAnalyzer
from src.models.topics import TopicExtractor
from src.scoring.aggregator import Aggregator
from src.scoring.loosh_index import LooshIndexCalculator
from src.outputs.snapshot import MoodSnapshot
from src.outputs.json_writer import JSONWriter
from src.outputs.csv_writer import CSVWriter
from src.outputs.console_writer import ConsoleWriter

console = Console()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
def cli():
    """loosh-farm: estimate the emotional temperature of the internet."""
    pass


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--mode", "-m", type=click.Choice(["lightweight", "deep"]), default=None,
              help="Analysis mode (overrides config)")
@click.option("--no-json", is_flag=True, help="Skip JSON output")
@click.option("--no-csv", is_flag=True, help="Skip CSV output")
@click.option("--no-console", is_flag=True, help="Skip console output")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(config: str, mode: str | None, no_json: bool, no_csv: bool, no_console: bool, verbose: bool):
    """Run the loosh-farm pipeline: collect, analyze, snapshot."""
    setup_logging(verbose)
    logger = logging.getLogger("loosh-farm")

    cfg = load_config(config)
    analysis_mode = mode or cfg.get("analysis_mode", "lightweight")

    max_items = cfg.get("max_items_per_source", 100)
    timeout = cfg.get("request_timeout", 15)
    max_retries = cfg.get("max_retries", 3)
    retry_delay = cfg.get("retry_delay", 2)
    rate_limit = cfg.get("rate_limit_delay", 1.0)

    collector_kwargs = dict(
        max_items=max_items,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        rate_limit_delay=rate_limit,
    )

    all_items: list[CollectedItem] = []

    # ── Step 1: Collect ──
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        collectors = []

        # RSS collectors for each category
        for category in ["rss_news", "rss_tech", "rss_world"]:
            feeds = cfg.get("rss_feeds", {}).get(
                {"rss_news": "news", "rss_tech": "tech", "rss_world": "world"}[category], []
            )
            if feeds:
                collectors.append(("RSS " + category, RSSCollector(cfg, category=category, **collector_kwargs)))

        # Reddit collector
        if cfg.get("reddit", {}).get("subreddits"):
            collectors.append(("Reddit", RedditCollector(cfg, **collector_kwargs)))

        collect_task = progress.add_task("Collecting data...", total=len(collectors))

        for name, collector in collectors:
            progress.update(collect_task, description=f"Collecting: {name}")
            try:
                items = collector.collect()
                all_items.extend(items)
                logger.info("Collected %d items from %s", len(items), name)
            except Exception as e:
                logger.error("Collector '%s' failed: %s", name, e)
            progress.advance(collect_task)

    console.print(f"  [dim]Raw items collected:[/dim] {len(all_items)}")

    if not all_items:
        console.print("[yellow]No items collected. Check your network connection and config.[/yellow]")
        sys.exit(1)

    # ── Step 2: Process ──
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        proc_task = progress.add_task("Processing and deduplicating...", total=None)
        pipeline = ProcessingPipeline(cfg)
        processed_items = pipeline.process(all_items)
        progress.update(proc_task, completed=True)

    console.print(f"  [dim]After processing:[/dim]    {len(processed_items)}")

    if not processed_items:
        console.print("[yellow]No items survived processing. Try adjusting config.[/yellow]")
        sys.exit(1)

    # ── Step 3: Analyze ──
    texts = [item.combined_text for item in processed_items]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        # Sentiment
        sent_task = progress.add_task(f"Analyzing sentiment ({analysis_mode})...", total=None)
        sentiment_analyzer = SentimentAnalyzer(mode=analysis_mode)
        sentiments = sentiment_analyzer.analyze_batch(texts)
        progress.update(sent_task, completed=True)

        # Emotions
        emo_task = progress.add_task(f"Analyzing emotions ({analysis_mode})...", total=None)
        emotion_analyzer = EmotionAnalyzer(mode=analysis_mode)
        emotions = emotion_analyzer.analyze_batch(texts)
        progress.update(emo_task, completed=True)

        # Topics
        topic_task = progress.add_task("Extracting trending topics...", total=None)
        topic_extractor = TopicExtractor()
        topic_result = topic_extractor.extract(texts)
        progress.update(topic_task, completed=True)

    # ── Step 4: Aggregate ──
    aggregator = Aggregator(cfg)
    aggregated = aggregator.aggregate(processed_items, sentiments, emotions)

    # ── Step 5: Loosh Index ──
    calculator = LooshIndexCalculator(cfg)
    loosh = calculator.calculate(aggregated)

    # ── Step 6: Build snapshot ──
    snapshot = MoodSnapshot.create(
        items_analyzed=aggregated.items_analyzed,
        sentiment=aggregated.sentiment,
        emotions=aggregated.emotions,
        dominant_emotion=aggregated.dominant_emotion,
        loosh_index=loosh.index,
        loosh_label=loosh.label,
        loosh_components=loosh.components,
        trending_topics={
            "words": topic_result.top_unigrams,
            "bigrams": topic_result.top_bigrams,
            "trigrams": topic_result.top_trigrams,
        },
        source_breakdown=aggregated.source_breakdown,
        analysis_mode=analysis_mode,
    )

    # ── Step 7: Output ──
    output_dir = cfg.get("output", {}).get("directory", "output")

    if not no_console:
        writer = ConsoleWriter()
        writer.write(snapshot)

    if not no_json:
        json_writer = JSONWriter(output_dir)
        json_path = json_writer.write(snapshot)
        console.print(f"  [green]JSON saved:[/green] {json_path}")

    if not no_csv:
        csv_writer = CSVWriter(output_dir)
        csv_path = csv_writer.write(snapshot)
        console.print(f"  [green]CSV appended:[/green] {csv_path}")

    console.print()


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
def sources(config: str):
    """List configured data sources."""
    cfg = load_config(config)

    console.print("\n[bold]Configured Sources[/bold]\n")

    rss_feeds = cfg.get("rss_feeds", {})
    for category, feeds in rss_feeds.items():
        console.print(f"  [bold cyan]RSS - {category}[/bold cyan]")
        for feed in feeds:
            console.print(f"    {feed['name']}: {feed['url']}")

    reddit = cfg.get("reddit", {})
    if reddit.get("subreddits"):
        console.print(f"\n  [bold cyan]Reddit[/bold cyan]")
        for sub in reddit["subreddits"]:
            console.print(f"    r/{sub}")

    console.print()


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
def validate(config: str):
    """Validate configuration and check connectivity."""
    cfg = load_config(config)

    console.print("\n[bold]Validating configuration...[/bold]\n")

    # Check required keys
    required = ["rss_feeds", "source_weights", "loosh_index"]
    for key in required:
        if key in cfg:
            console.print(f"  [green]✓[/green] {key}")
        else:
            console.print(f"  [red]✗[/red] {key} (missing)")

    # Check source weights sum
    weights = cfg.get("source_weights", {})
    total = sum(weights.values())
    if abs(total - 1.0) < 0.01:
        console.print(f"  [green]✓[/green] source_weights sum = {total:.2f}")
    else:
        console.print(f"  [yellow]![/yellow] source_weights sum = {total:.2f} (should be ~1.0)")

    # Check NLP dependencies
    console.print("\n  [bold]NLP Dependencies:[/bold]")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        console.print("  [green]✓[/green] vaderSentiment")
    except ImportError:
        console.print("  [red]✗[/red] vaderSentiment")

    try:
        import feedparser
        console.print("  [green]✓[/green] feedparser")
    except ImportError:
        console.print("  [red]✗[/red] feedparser")

    try:
        import transformers
        console.print(f"  [green]✓[/green] transformers ({transformers.__version__})")
    except ImportError:
        console.print("  [yellow]![/yellow] transformers (optional, needed for deep mode)")

    console.print()


if __name__ == "__main__":
    cli()
