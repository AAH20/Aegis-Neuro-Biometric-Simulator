"""Rich terminal reporting for Aegis-Neuro."""

from __future__ import annotations

from typing import Iterable

from rich.box import HEAVY_HEAD
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agents.judge import JudgeVerdict
from .metrics import AggregateMetrics, aggregate
from .simulator import RunResult


def _banner(console: Console, *, source: str, seed: int) -> None:
    title = Text("AEGIS-NEURO", style="bold bright_white")
    subtitle = Text(
        "Multi-Agent Adversarial Biometric Simulator",
        style="italic dim",
    )
    body = Text.assemble(
        title, "\n", subtitle, "\n\n",
        ("run: ", "dim"), (source, "bright_white"), "\n",
        ("seed: ", "dim"), (str(seed), "bright_white"),
    )
    console.print(Panel(body, border_style="magenta", expand=False))


def _verdict_style(v: JudgeVerdict) -> str:
    if not v.bypassed:
        return "green"
    assert v.attempts_to_bypass is not None
    if v.attempts_to_bypass <= 50:
        return "bold white on red"
    if v.attempts_to_bypass <= 500:
        return "bright_red"
    return "yellow"


def render_run_table(
    results: Iterable[RunResult],
    *,
    console: Console | None = None,
) -> None:
    """Render a multi-run summary table."""
    console = console or Console()
    results = list(results)

    table = Table(
        title="Aegis-Neuro Run Verdicts",
        title_style="bold bright_white",
        box=HEAVY_HEAD,
        header_style="bold bright_white on magenta",
        show_lines=False,
        expand=False,
    )
    table.add_column("Modality", style="cyan", no_wrap=True, min_width=11)
    table.add_column("Strategy", style="bright_white", no_wrap=True, min_width=13)
    table.add_column("Outcome", no_wrap=True, min_width=8)
    table.add_column("Att.", justify="right", no_wrap=True, min_width=6)
    table.add_column("Wall ms", justify="right", no_wrap=True, min_width=8)
    table.add_column("Sim", justify="right", no_wrap=True, min_width=5)
    table.add_column("Thr", justify="right", no_wrap=True, min_width=5)
    table.add_column("Live Rej", justify="right", no_wrap=True, min_width=8)

    for r in results:
        v = r.verdict
        outcome = "BYPASSED" if v.bypassed else "HELD"
        outcome_text = Text(outcome, style=_verdict_style(v))

        attempts_cell = (
            str(v.attempts_to_bypass) if v.bypassed else f"{v.attempts}"
        )
        wall_ms_cell = (
            f"{(v.wall_seconds_to_bypass or 0.0) * 1000:.2f}"
            if v.bypassed
            else f"{v.total_wall_seconds * 1000:.1f}"
        )

        table.add_row(
            v.modality,
            v.strategy,
            outcome_text,
            attempts_cell,
            wall_ms_cell,
            f"{v.best_similarity:.3f}",
            f"{v.threshold:.3f}",
            str(v.liveness_rejections),
        )

    console.print(table)


def render_aggregate(metrics: AggregateMetrics, console: Console | None = None) -> None:
    """Print the bottom-line aggregate panel."""
    console = console or Console()

    if metrics.runs == 0:
        console.print(Panel("No runs.", border_style="dim"))
        return

    rate_pct = metrics.bypass_rate * 100.0
    headline_style = (
        "bold white on red"
        if metrics.bypass_rate >= 0.5
        else "bright_red"
        if metrics.bypass_rate > 0
        else "green"
    )
    headline = Text(
        f"{metrics.bypassed}/{metrics.runs} runs bypassed ({rate_pct:.0f}%)",
        style=headline_style,
    )

    detail_lines: list[str] = []
    if metrics.mean_attempts_to_bypass is not None:
        detail_lines.append(
            f"mean attempts to bypass: {metrics.mean_attempts_to_bypass:.1f}"
        )
    if metrics.median_attempts_to_bypass is not None:
        detail_lines.append(
            f"median attempts to bypass: {metrics.median_attempts_to_bypass:.1f}"
        )
    if metrics.mean_wall_ms_to_bypass is not None:
        detail_lines.append(
            f"mean wall-time to bypass: {metrics.mean_wall_ms_to_bypass:.2f} ms"
        )
    if metrics.weakest_modality:
        detail_lines.append(f"weakest modality: [bold]{metrics.weakest_modality}[/]")
    if metrics.strongest_modality:
        detail_lines.append(
            f"strongest modality: [bold]{metrics.strongest_modality}[/]"
        )

    body = Text.assemble(
        headline, "\n\n",
        Text.from_markup("\n".join(detail_lines) or "(no bypass observed)"),
    )
    console.print(
        Panel(
            body,
            border_style="red" if metrics.bypass_rate > 0 else "green",
            title="AGGREGATE",
        )
    )


def generate_report(
    results: Iterable[RunResult],
    *,
    source: str,
    seed: int,
    console: Console | None = None,
) -> AggregateMetrics:
    """Render the full Aegis-Neuro report and return the aggregate."""
    console = console or Console()
    results = list(results)

    _banner(console, source=source, seed=seed)
    render_run_table(results, console=console)
    metrics = aggregate(results)
    render_aggregate(metrics, console=console)
    return metrics


__all__ = [
    "generate_report",
    "render_aggregate",
    "render_run_table",
]
