# Aegis-Neuro

> **A deterministic, local-only multi-agent adversarial simulator for multi-modal biometric systems — fingerprint, retina, EEG ("Brain Print"), ECG, and voice.**

Aegis-Neuro models the *shape* of adversarial attacks against five common biometric modalities so defenders can reason about decision thresholds, liveness detection, and multi-modal fusion at the architectural level. It runs entirely offline, with no API keys and no external services, and every run is reproducible from a single integer seed.

It is the companion tool to [`Aegis-IEEE11073-Scanner`](https://github.com/AAH20/Aegis-IEEE11073-Scanner) — the 11073-scanner audits live medical telemetry; Aegis-Neuro stress-tests the biometric authentication layer that sits above it.

---

## Scope and honest framing

Aegis-Neuro is **a simulator**, not an exploit kit. Specifically:

- It does **not** break any real biometric device, sensor, or matcher.
- It does **not** model any specific commercial product, named CVE, or vendor implementation.
- It does **not** include data from real subjects; every template is synthetic.
- It runs entirely on your laptop, on the CPU, with `numpy` math. No GPUs, no API keys, no telemetry.

What it *does* do is let a security architect ask questions like:

- *"If an attacker recovers 20% of an EEG enrolment via a side channel, how many submissions does it take a population-based optimiser to cross my decision threshold?"*
- *"How much does adding a liveness check raise the cost of bypassing my voice biometric?"*
- *"Which of my five biometric modalities is the cheapest target for an attacker with a verbose match-score oracle?"*

— and answer them in seconds with a reproducible, citable run.

---

## What's modelled

Three agents, one event loop:

| Agent | Role |
|---|---|
| **Attacker** | Generates candidate templates. Four strategies: `blind`, `partial_leak`, `hill_climb`, `evolutionary`. |
| **Defender** | Authenticates each candidate against an enrolment with a similarity score + threshold + optional liveness check. |
| **Judge** | Records every attempt and computes time-to-bypass and attempts-to-bypass. |

Five biometric modalities, each with its own feature shape, scorer, and liveness model:

| Modality | Feature shape | Scorer | Liveness check (modelled) |
|---|---|---|---|
| `fingerprint` | 30 minutiae × (x, y, θ) | nearest-neighbour spatial + orientation | reject implausibly clustered minutiae |
| `retina` | 25 bifurcations × (x, y, branch-angle) | spatial + angular | reject overly-uniform angle distributions |
| `eeg` | 8 channels × 5 spectral bands | cosine on the flattened band-power matrix | reject impossible channel-power ratios |
| `ecg` | 10 morphology + HRV features | distance kernel | reject implied HR < 40 or > 180 BPM |
| `voice` | 128-D unit-norm embedding (x-vector abstraction) | cosine | reject embeddings off the unit sphere |

All five share the same `BiometricModality` interface, so adding a sixth (gait, keystroke dynamics, vein, etc.) is a single subclass.

---

## Installation

```bash
git clone https://github.com/AAH20/Aegis-Neuro.git
cd Aegis-Neuro
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
aegis-neuro version
```

---

## Usage

### Run a single modality / strategy

```bash
aegis-neuro run --modality eeg --strategy evolutionary --attempts 1000 --seed 42
```

Options:

| Flag | Description |
|---|---|
| `-m / --modality` | One of: `fingerprint`, `retina`, `eeg`, `ecg`, `voice`. |
| `-s / --strategy` | One of: `blind`, `partial_leak`, `hill_climb`, `evolutionary`. |
| `--attempts` | Maximum attacker submissions (default 1000). |
| `--seed` | Integer seed. Same seed → identical run. |
| `--leak-fraction` | For `partial_leak`: fraction of the enrolment exposed (default 0.20). |
| `--threshold` | Override the modality's default decision threshold. |
| `--no-liveness` | Disable defender liveness check (model a weakened defender). |
| `--no-stop-on-bypass` | Continue after first bypass to collect the full curve. |
| `--export` | Write attempt log as JSONL to a path. |

### Run the full attack matrix (5 modalities × 4 strategies)

```bash
aegis-neuro run-all --attempts 500 --seed 1337 --export runs/full.jsonl
```

### List

```bash
aegis-neuro list-modalities
aegis-neuro list-strategies
```

---

## Reading the report

```text
                           Aegis-Neuro Run Verdicts
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Modality   ┃ Strategy     ┃ Outcome  ┃ Attempts┃ Wall(ms)┃ Best Sim┃ Threshold┃ Liveness Rej. ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ fingerprint│ partial_leak │ BYPASSED │ 14      │ 1.92    │ 0.812   │ 0.780    │ 0             │
│ retina     │ evolutionary │ BYPASSED │ 312     │ 28.41   │ 0.834   │ 0.820    │ 4             │
│ eeg        │ blind        │ HELD     │ 1000    │ 134.7   │ 0.711   │ 0.880    │ 0             │
│ ecg        │ hill_climb   │ HELD     │ 1000    │ 96.3    │ 0.812   │ 0.860    │ 12            │
│ voice      │ partial_leak │ BYPASSED │  87     │ 9.04    │ 0.847   │ 0.830    │ 0             │
└────────────┴──────────────┴──────────┴─────────┴─────────┴─────────┴──────────┴───────────────┘
╭──────────────────────────────── AGGREGATE ────────────────────────────────╮
│ 3/5 runs bypassed (60%)                                                   │
│                                                                           │
│ mean attempts to bypass: 137.7                                            │
│ median attempts to bypass: 87                                             │
│ weakest modality: fingerprint                                             │
│ strongest modality: eeg                                                   │
╰───────────────────────────────────────────────────────────────────────────╯
```

Exit codes: `0` no bypass / minority bypass, `1` significant bypass (≥50% of configurations).

---

## Architecture

```text
src/aegis_neuro/
├── modalities/
│   ├── base.py          # BiometricModality, Template, MatchResult
│   ├── fingerprint.py
│   ├── retina.py
│   ├── eeg.py
│   ├── ecg.py
│   └── voice.py
├── agents/
│   ├── base.py          # Agent, AttemptRecord
│   ├── attacker.py      # AttackerAgent + 4 strategies
│   ├── defender.py
│   └── judge.py
├── simulator.py         # async multi-agent event loop
├── metrics.py           # cross-run aggregation
├── reporter.py          # rich-formatted terminal report
└── cli.py               # click entry point
```

Embed in your own pipeline:

```python
import asyncio
from aegis_neuro.simulator import RunConfig, run_simulation

cfg = RunConfig(modality="voice", strategy="evolutionary", max_attempts=500, seed=42)
result = asyncio.run(run_simulation(cfg))
print(result.verdict.headline)
```

---

## What this tool is good for

- **Architectural threat modelling.** Quantify, in attempts-to-bypass, how a single threshold change or the addition of a liveness check shifts an attacker's cost.
- **Defence-in-depth sizing.** Run the same attack against multiple modalities to see which one your fusion layer should weight least.
- **Reproducible demos.** Same seed, same numbers, every time. Drop a run into a slide deck and the audit committee can re-run it themselves.
- **Teaching.** Show a junior engineer why "we use biometrics, so we're secure" is not a defensive position.

## What this tool is **not** good for

- Generating evidence about any specific commercial product.
- Claiming a real-world break of any deployed system.
- Replacing a hands-on red-team engagement with proper scope-of-work.

If you need any of those, hire qualified humans with signed contracts and a lab.

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

---

## License

MIT. See `pyproject.toml`.
