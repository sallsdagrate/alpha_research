# Alpha Research

Minimal local workflow for downloading public Binance trade data and exploring it
in notebooks.

There is intentionally no normalization or dataset-building pipeline. The CLI
stores the original daily ZIP archives under `data/raw`; notebook code loads
those files directly with Polars or pandas.

## Setup

```bash
uv sync --extra dev --extra notebook
```

## Download data

Edit `configs/data/binance_core.yaml`, preview the request list, then download:

```bash
uv run alpha data download \
  --config configs/data/binance_core.yaml \
  --dry-run

uv run alpha data download \
  --config configs/data/binance_core.yaml
```

Existing archives are reused. Add `--force` only when you intentionally want to
download them again.

## Load data in a notebook

Polars:

```python
from alpha_research.data import load_trades

spot = load_trades(
    market="spot",
    symbol="BTCUSDT",
    start_date="2024-01-01",
    end_date="2024-01-03",
)
```

pandas:

```python
perp = load_trades(
    market="perp",
    symbol="BTCUSDT",
    start_date="2024-01-01",
    end_date="2024-01-03",
    engine="pandas",
)
```

The loader only handles Binance's inconsistent CSV headers and adds
`exchange`, `market`, `symbol`, and `source_date`. Timestamp conversion,
cleaning, aggregation, alignment, features, and labels belong in the notebook.

To list the matching archives without loading them:

```python
from alpha_research.data import find_trade_archives

find_trade_archives(market="spot", symbol="BTCUSDT")
```
