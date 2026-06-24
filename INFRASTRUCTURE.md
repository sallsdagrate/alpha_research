# Data workflow

The infrastructure is deliberately small:

```text
Binance public archive
        |
        v
data/raw/**/*.zip
        |
        v
load_trades(...)
        |
        v
research notebook
```

## Responsibilities

The CLI:

- constructs Binance public archive URLs;
- downloads daily trade ZIP files;
- skips files that already exist unless `--force` is supplied.

The loader:

- finds local archives by market, symbol, and date;
- reads the different Binance spot and futures CSV formats;
- returns one Polars or pandas DataFrame.

The notebook owns everything else:

- timestamp conversion;
- filtering and cleaning;
- resampling and time grids;
- spot/perp alignment;
- feature engineering;
- labels, validation, plots, and backtests.

## Stored data

Only original downloaded archives are considered infrastructure-managed data:

```text
data/raw/
  exchange=binance/
    market=spot|perp/
      dataset=trades/
        symbol=BTCUSDT/
          date=YYYY-MM-DD/
            BTCUSDT-trades-YYYY-MM-DD.zip
```

Any Parquet, CSV, or notebook-derived dataset is optional research output and
should be created explicitly by the researcher.
