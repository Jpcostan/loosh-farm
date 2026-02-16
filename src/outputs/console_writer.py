"""Rich console output for mood snapshot."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .snapshot import MoodSnapshot


class ConsoleWriter:
    """Renders a beautiful console summary of the mood snapshot."""

    def __init__(self):
        self.console = Console()

    def _index_color(self, index: float) -> str:
        if index < 30:
            return "green"
        elif index < 50:
            return "cyan"
        elif index < 65:
            return "yellow"
        elif index < 80:
            return "dark_orange"
        else:
            return "red"

    def _sentiment_bar(self, value: float, width: int = 20) -> str:
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)

    def write(self, snapshot: MoodSnapshot):
        self.console.print()

        # Header
        color = self._index_color(snapshot.loosh_index)
        header = Text()
        header.append("LOOSH FARM ", style="bold white")
        header.append("— Global Emotional Temperature", style="dim")
        self.console.print(Panel(header, box=box.DOUBLE, style="bold"))

        # Timestamp & meta
        self.console.print(f"  [dim]Timestamp:[/dim]  {snapshot.timestamp_utc}")
        self.console.print(f"  [dim]Items:[/dim]      {snapshot.items_analyzed}")
        self.console.print(f"  [dim]Mode:[/dim]       {snapshot.analysis_mode}")
        self.console.print()

        # Loosh Index (big number)
        index_display = Text()
        index_display.append(f"  {snapshot.loosh_index}", style=f"bold {color}")
        index_display.append(f" / 100  ", style="dim")
        index_display.append(f"[{snapshot.loosh_label}]", style=f"{color}")
        self.console.print(Panel(index_display, title="[bold]Loosh Index[/bold]", box=box.ROUNDED))

        # Sentiment breakdown
        sent_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        sent_table.add_column("Sentiment", width=10)
        sent_table.add_column("Score", width=8, justify="right")
        sent_table.add_column("Bar", width=22)

        pos = snapshot.sentiment.get("positive", 0)
        neu = snapshot.sentiment.get("neutral", 0)
        neg = snapshot.sentiment.get("negative", 0)

        sent_table.add_row("Positive", f"{pos:.1%}", Text(self._sentiment_bar(pos), style="green"))
        sent_table.add_row("Neutral", f"{neu:.1%}", Text(self._sentiment_bar(neu), style="cyan"))
        sent_table.add_row("Negative", f"{neg:.1%}", Text(self._sentiment_bar(neg), style="red"))

        self.console.print(Panel(sent_table, title="[bold]Sentiment[/bold]", box=box.ROUNDED))

        # Emotion distribution
        emo_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        emo_table.add_column("Emotion", width=14)
        emo_table.add_column("Score", width=8, justify="right")
        emo_table.add_column("Bar", width=22)

        emo_colors = {
            "joy": "green",
            "trust": "cyan",
            "anticipation": "blue",
            "surprise": "magenta",
            "anger": "red",
            "fear": "dark_orange",
            "sadness": "blue",
            "disgust": "yellow",
        }

        sorted_emotions = sorted(snapshot.emotions.items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_emotions:
            c = emo_colors.get(emotion, "white")
            emo_table.add_row(
                Text(emotion.capitalize(), style=c),
                f"{score:.1%}",
                Text(self._sentiment_bar(score), style=c),
            )

        self.console.print(Panel(emo_table, title="[bold]Emotions[/bold]", box=box.ROUNDED))

        # Source breakdown
        if snapshot.source_breakdown:
            src_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            src_table.add_column("Source", width=14)
            src_table.add_column("Count", width=8, justify="right")
            src_table.add_column("Pos", width=8, justify="right")
            src_table.add_column("Neg", width=8, justify="right")
            src_table.add_column("Weight", width=8, justify="right")

            for source, data in snapshot.source_breakdown.items():
                src_table.add_row(
                    source,
                    str(data.get("count", 0)),
                    f"{data.get('sentiment', {}).get('positive', 0):.1%}",
                    f"{data.get('sentiment', {}).get('negative', 0):.1%}",
                    f"{data.get('weight', 0):.0%}",
                )

            self.console.print(Panel(src_table, title="[bold]Sources[/bold]", box=box.ROUNDED))

        # Trending topics
        topics = snapshot.trending_topics
        if topics:
            topic_parts = []
            for bigram, count in topics.get("bigrams", [])[:10]:
                topic_parts.append(f"  [bold]{bigram}[/bold] [dim]({count})[/dim]")
            if topic_parts:
                topics_text = "\n".join(topic_parts)
                self.console.print(
                    Panel(topics_text, title="[bold]Trending Phrases[/bold]", box=box.ROUNDED)
                )

            word_parts = []
            for word, count in topics.get("words", [])[:10]:
                word_parts.append(f"  {word} [dim]({count})[/dim]")
            if word_parts:
                words_text = "\n".join(word_parts)
                self.console.print(
                    Panel(words_text, title="[bold]Top Words[/bold]", box=box.ROUNDED)
                )

        self.console.print()
