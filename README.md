# Alpha Research

Local-first research toolkit for crypto perpetual futures alpha hypotheses.

The project is intentionally file-based: public exchange data is downloaded into
`data/raw`, normalized into Parquet under `data/normalized`, transformed into
research tables under `data/curated`, and tracked with JSON manifests.

## Setup

```bash
uv sync --extra dev --extra notebook
uv run alpha --help
```

If `uv` is not installed yet, install it first, then run the commands above.

## No-download dry run

The fetch command supports planning without downloading:

```bash
alpha data fetch --config configs/data/binance_core.yaml --dry-run
```

When ready to download, run the same command without `--dry-run`.

## First target

The first vertical slice is Hypothesis 1:

> Does spot signed trade imbalance at time `t` predict perp returns over short
> horizons after timestamp alignment, latency, spread, and fees?
