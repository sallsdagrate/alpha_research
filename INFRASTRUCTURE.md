# Alpha Research Infrastructure Plan

Status: proposed  
Scope: crypto perpetual futures alpha research from [PLAN.md](PLAN.md)  
Objective: keep infrastructure boring and Python-native so the work stays focused on alpha research.

## 1. Core decision

Use a local-first Python package over Parquet files and JSON manifests.

Start with:

- Python 3.12+
- `uv` with a committed lockfile
- Polars for core pipelines
- pandas only where stats libraries or notebooks require it
- Parquet with Zstandard compression
- YAML configs
- JSON manifests
- Typer CLI: `alpha ...`
- pytest, Hypothesis, Ruff, Pyright
- executed notebooks or Quarto reports

Do not start with DuckDB, databases, warehouses, Airflow, Kafka, MLflow, feature stores, dashboards, Kubernetes, or live trading services.

If Python + Parquet becomes painful, add tooling only after there is measured friction.

## 2. System shape

```text
public archives/APIs
        |
        v
raw files + source manifests
        |
        v
canonical Parquet events
        |
        v
curated research tables
        |
        v
features + labels
        |
        v
experiments + reports
```

Out of scope for now: live feeds, order routing, portfolio operations, and deployment.

## 3. Required guarantees

Even with simple infrastructure, enforce:

- **Point-in-time correctness**: features never use data unavailable at decision time.
- **Reproducibility**: experiments record config, code version, environment, and input fingerprints.
- **Idempotence**: reruns do not duplicate or mutate valid outputs unless `--force` is used.
- **Auditability**: every derived file links to parent files and transform versions.
- **Fast local iteration**: same code runs on one day or a larger date range.

## 4. Repository layout

```text
.
├── PLAN.md
├── INFRASTRUCTURE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── data/
│   ├── experiments/
│   └── costs/
├── src/alpha_research/
│   ├── cli.py
│   ├── config.py
│   ├── paths.py
│   ├── manifests.py
│   ├── provenance.py
│   ├── data/
│   │   ├── contracts.py
│   │   ├── download.py
│   │   ├── normalize.py
│   │   ├── quality.py
│   │   └── align.py
│   ├── features/
│   ├── labels/
│   ├── validation/
│   ├── backtest/
│   └── reporting/
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   └── fixtures/
├── notebooks/
├── reports/
├── data/                       # ignored
│   ├── raw/
│   ├── normalized/
│   ├── curated/
│   └── features/
└── artifacts/                  # ignored
    └── runs/
```

When the skeleton is added, move existing pairs-trading work under `archive/` so it does not mix with the perp package.

## 5. Data layers

### Raw

Exact downloaded files or API responses. Append-only. Each file gets:

```text
<file>.manifest.json
```

Manifest fields:

- source URL/request
- retrieval time
- byte count
- SHA-256
- exchange, market, symbol, date range
- downloader version

### Normalized

Canonical Parquet events with consistent names, types, timestamp fields, side conventions, and schema versions. Invalid records go to quarantine files with reason codes.

### Curated

Research-ready tables:

- one-second trade-flow bars
- spot/perp aligned panels
- funding and basis panels
- later: order-book-derived panels

### Features

Materialize only expensive or reused features. Compute cheap experiment-specific features inside the run.

## 6. Partitioning

Use Hive-style paths:

```text
data/normalized/trades/
  exchange=binance/market=perp/symbol=BTCUSDT/date=2026-01-01/part-000.parquet
```

Partition by event type, exchange, market, symbol, and UTC date. Do not partition by hour initially. Sort within partitions by `event_ts`, then source sequence where available.

## 7. Time contract

Every event table has:

| Field | Meaning |
|---|---|
| `event_ts` | Exchange/source event time |
| `receive_ts` | Collector receipt time; nullable for archives |
| `available_ts` | Earliest defensible time the strategy could observe the event |
| `ingested_at` | Local ingestion time; provenance only |
| `source_sequence` | Exchange sequence/update identifier where supplied |

For historical archives, use:

```text
available_ts = event_ts + configured_latency
```

Record that assumption in the experiment config. Never align on `ingested_at`.

Cross-source joins must be backward as-of joins on `available_ts`, with grouping keys and max staleness. Features may not look forward. Labels may.

## 8. Initial tables

Build only what the next hypotheses need:

- `trades`
- `quotes_l1`
- `klines`
- `funding`
- `mark_index`
- `open_interest`
- `instrument_metadata`

Defer `book_updates_l2` and order-book reconstruction until the order-flow hypothesis.

## 9. Manifests instead of a database

The source of truth is file manifests.

Dataset manifest:

```json
{
  "dataset_id": "normalized.trades.binance.perp.BTCUSDT.2026-01-01",
  "schema_version": "trades.v1",
  "path": "data/normalized/trades/exchange=binance/market=perp/symbol=BTCUSDT/date=2026-01-01/",
  "parents": ["raw.binance.perp.trades.BTCUSDT.2026-01-01"],
  "row_count": 123456,
  "sha256": "...",
  "transform": "normalize_trades@v1",
  "config_hash": "...",
  "created_at": "..."
}
```

Run manifest:

```json
{
  "run_id": "...",
  "git_commit": "...",
  "resolved_config_hash": "...",
  "inputs": ["curated.spot_perp_1s.binance.BTCUSDT.2026-01"],
  "lockfile_hash": "..."
}
```

Discovery is a Python scan over manifest files:

```bash
alpha manifest list --kind dataset
alpha manifest list --kind run
alpha manifest show <id>
alpha manifest check <id>
```

If scanning becomes slow, generate a cache file. Do not add a persistent query engine until this is actually a bottleneck.

## 10. CLI contract

```bash
alpha data fetch --config configs/data/binance_core.yaml
alpha data normalize --dataset trades --date 2026-01-01
alpha data validate --dataset trades --date 2026-01-01
alpha data build --dataset spot_perp_1s --from 2026-01-01 --to 2026-01-31
alpha experiment run --config configs/experiments/h1_spot_lead_baseline.yaml
alpha report build --run-id <id>
```

Every command:

1. loads typed config;
2. fingerprints inputs;
3. skips valid existing outputs unless `--force`;
4. writes to a temp path;
5. validates;
6. atomically publishes output and manifest;
7. logs row counts, paths, failures, and duration.

Keep orchestration as normal Python functions. Add a DAG runner only if manual ordering becomes painful.

## 11. Experiment contract

Each experiment is a committed YAML file containing:

- hypothesis and plain-language claim
- universe, venues, markets, dates
- input dataset IDs/fingerprints
- feature and label versions
- horizons and split definition
- latency, fees, spread, slippage, funding assumptions
- model and hyperparameters, if any
- random seeds
- metrics and robustness scenarios

Run output:

```text
artifacts/runs/<run-id>/
├── resolved_config.yaml
├── manifest.json
├── data_quality.json
├── metrics.json
├── predictions.parquet
├── positions.parquet
├── trades.parquet
├── figures/
└── report.html
```

The run directory must be self-contained. Notebooks can explain results, but core logic must live in package code.

## 12. Research safety rails

Implement reusable primitives for:

- point-in-time joins
- feature availability declarations
- label builders separate from features
- chronological/walk-forward splits
- purge/embargo for overlapping labels
- execution after `decision_ts + latency`
- bid/ask-aware fills
- spread, fee, slippage, impact, and funding costs
- gross and net metrics

Property tests should prove:

- as-of joins never select `available_ts > decision_ts`;
- future rows can change labels but not earlier features;
- increasing non-negative costs cannot improve PnL.

## 13. Testing

Use:

- unit tests for formulas, joins, labels, costs, and metrics;
- contract tests for schemas and exchange adapters;
- one tiny raw-to-report integration test;
- invariant tests for causality and accounting.

CI runs lint, type checks, unit tests, contract tests, and the tiny integration test. Full historical builds are local.

## 14. Versioning

Git stores code, configs, schemas, docs, fixtures, and small manifest examples.

Git ignores raw data, large Parquet files, generated reports/figures, notebook outputs, and run artifacts.

Dataset identity is a hash of source checksums, schema version, transform version, resolved config, and parent IDs.

Add object storage or DVC/lakeFS only if data sharing, disk limits, mutable sources, or remote compute require it.

## 15. Delivery sequence

### Phase 0 — foundation

Package skeleton, locked environment, CLI, typed config, path handling, manifest utilities, logging, tests, and one synthetic end-to-end fixture.

Exit: `uv run pytest` and `uv run alpha --help` work.

### Phase 1 — Hypothesis 1 data path

Binance spot/perp trades for BTCUSDT and ETHUSDT. Add raw manifests, canonical trades, validation, one-second aggregation, and point-in-time spot/perp alignment.

Exit: one command builds a reproducible spot/perp research table for a date range.

### Phase 2 — first research loop

Forward returns, IC/rank-IC, quantile returns, decay curves, chronological splits, baseline cost model, run manifests, and report template.

Exit: first spot-lead report has gross/net results, latency sensitivity, and data-quality caveats.

### Phase 3 — funding/basis

Funding, mark/index, open interest, basis panels, and funding-aware PnL.

### Phase 4 — order flow

L2 snapshots/deltas, sequence checks, deterministic reconstruction, microprice/depth features, and latency-aware execution tests.

### Phase 5 — second venue

Add another venue behind the same schemas and quantify clock/availability limitations.

## 16. First research slice

Start with:

> Does spot signed trade imbalance at time `t` predict perp returns over 1s, 5s, 30s, and 60s after latency, spread, fees, and out-of-sample validation?

Use trades aggregated to one-second intervals. Do not start with L2 order books.

## 17. Explicit deferrals

- no DuckDB or database layer;
- no live/paper trading;
- no generic backtesting framework before first signal tests;
- no feature store;
- no distributed compute;
- no dashboard;
- no ML experiment platform;
- no automatic raw-data deletion;
- no nanosecond claims from millisecond/archive timestamps.
