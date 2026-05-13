"""Aegis-Neuro command-line entry point."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .agents.attacker import ATTACK_STRATEGIES
from .modalities import MODALITY_REGISTRY
from .reporter import generate_report
from .simulator import RunConfig, RunResult, run_many

_CONSOLE = Console()


def _print_strategies() -> None:
    _CONSOLE.print("[bold]Attack strategies:[/]")
    for s in ATTACK_STRATEGIES:
        _CONSOLE.print(f"  - {s}")


def _print_modalities() -> None:
    _CONSOLE.print("[bold]Biometric modalities:[/]")
    for name, cls in MODALITY_REGISTRY.items():
        inst = cls()
        _CONSOLE.print(
            f"  - {name:<11} default_threshold={inst.default_threshold:.2f}"
        )


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Aegis-Neuro — multi-agent adversarial biometric simulator.",
)
@click.version_option(__version__, prog_name="aegis-neuro")
def main() -> None:
    """Aegis-Neuro CLI root."""


@main.command("list-modalities", help="List available biometric modalities.")
def list_modalities() -> None:
    _print_modalities()


@main.command("list-strategies", help="List available attack strategies.")
def list_strategies() -> None:
    _print_strategies()


@main.command(
    "run",
    help="Run a single attacker-vs-defender simulation against one modality.",
)
@click.option(
    "--modality",
    "-m",
    type=click.Choice(sorted(MODALITY_REGISTRY.keys())),
    required=True,
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(list(ATTACK_STRATEGIES)),
    default="evolutionary",
    show_default=True,
)
@click.option("--attempts", type=int, default=1_000, show_default=True)
@click.option("--seed", type=int, default=1337, show_default=True)
@click.option(
    "--leak-fraction",
    type=float,
    default=0.20,
    show_default=True,
    help="For 'partial_leak': fraction of the enrolment exposed to attacker.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Override the modality's default decision threshold.",
)
@click.option(
    "--no-liveness",
    is_flag=True,
    default=False,
    help="Disable the defender's liveness check (weakens defender).",
)
@click.option(
    "--no-stop-on-bypass",
    is_flag=True,
    default=False,
    help="Continue after the first successful bypass (collect full curve).",
)
@click.option(
    "--export",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Write attempt log as JSONL to PATH.",
)
def run_one(
    modality: str,
    strategy: str,
    attempts: int,
    seed: int,
    leak_fraction: float,
    threshold: float | None,
    no_liveness: bool,
    no_stop_on_bypass: bool,
    export: Path | None,
) -> None:
    cfg = RunConfig(
        modality=modality,
        strategy=strategy,  # type: ignore[arg-type]
        max_attempts=attempts,
        seed=seed,
        leak_fraction=leak_fraction,
        threshold=threshold,
        check_liveness=not no_liveness,
        stop_on_bypass=not no_stop_on_bypass,
    )
    results: list[RunResult] = asyncio.run(run_many([cfg]))
    metrics = generate_report(
        results,
        source=f"run --modality {modality} --strategy {strategy}",
        seed=seed,
        console=_CONSOLE,
    )

    if export:
        _export_jsonl(results, export)
        _CONSOLE.print(f"[dim]wrote attempt log -> {export}[/]")

    sys.exit(1 if metrics.bypass_rate > 0 else 0)


@main.command(
    "run-all",
    help="Run every modality with every attacker strategy and produce a "
    "consolidated report.",
)
@click.option("--attempts", type=int, default=1_000, show_default=True)
@click.option("--seed", type=int, default=1337, show_default=True)
@click.option(
    "--leak-fraction",
    type=float,
    default=0.20,
    show_default=True,
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(list(ATTACK_STRATEGIES)),
    default=None,
    help="If set, run all modalities against just this single strategy.",
)
@click.option(
    "--modalities",
    "-m",
    multiple=True,
    type=click.Choice(sorted(MODALITY_REGISTRY.keys())),
    help="Restrict to these modalities (default: all).",
)
@click.option("--no-liveness", is_flag=True, default=False)
@click.option(
    "--export",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
)
def run_all(
    attempts: int,
    seed: int,
    leak_fraction: float,
    strategy: str | None,
    modalities: tuple[str, ...],
    no_liveness: bool,
    export: Path | None,
) -> None:
    mods = list(modalities) if modalities else list(MODALITY_REGISTRY.keys())
    strats = [strategy] if strategy else list(ATTACK_STRATEGIES)

    configs: list[RunConfig] = []
    for m in mods:
        for s in strats:
            configs.append(
                RunConfig(
                    modality=m,
                    strategy=s,  # type: ignore[arg-type]
                    max_attempts=attempts,
                    seed=seed,
                    leak_fraction=leak_fraction,
                    check_liveness=not no_liveness,
                )
            )

    results: list[RunResult] = asyncio.run(run_many(configs))
    metrics = generate_report(
        results,
        source=f"run-all (modalities={','.join(mods)}, "
        f"strategies={','.join(strats)})",
        seed=seed,
        console=_CONSOLE,
    )

    if export:
        _export_jsonl(results, export)
        _CONSOLE.print(f"[dim]wrote attempt log -> {export}[/]")

    sys.exit(1 if metrics.bypass_rate >= 0.5 else 0)


def _export_jsonl(results: list[RunResult], path: Path) -> None:
    """Write per-attempt records as JSONL plus a final verdict-summary line."""
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            for rec in r.records:
                fh.write(json.dumps(rec.to_dict()) + "\n")
            fh.write(json.dumps({"_verdict": r.verdict.to_dict()}) + "\n")


@main.command("version", help="Print version + dependency banner.")
def version() -> None:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    _CONSOLE.print(f"[bold]aegis-neuro[/] [bright_white]{__version__}[/]")
    for dep in ("numpy", "click", "rich", "pydantic"):
        try:
            _CONSOLE.print(f"  {dep:<8} {_pkg_version(dep)}")
        except PackageNotFoundError:
            _CONSOLE.print(f"  {dep:<8} [red]not installed[/]")


if __name__ == "__main__":  # pragma: no cover
    main()
