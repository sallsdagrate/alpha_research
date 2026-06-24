"""Public archive download planning and execution."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from alpha_research.config import DataConfig
from alpha_research.paths import DataPaths


@dataclass(frozen=True)
class DownloadRequest:
    exchange: str
    market: str
    dataset: str
    symbol: str
    day: date
    url: str
    destination: Path

    @property
    def id(self) -> str:
        return (
            f"raw.{self.exchange}.{self.market}.{self.dataset}."
            f"{self.symbol}.{self.day.isoformat()}"
        )


def iter_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def plan_downloads(config: DataConfig) -> list[DownloadRequest]:
    paths = DataPaths(config.data_root)
    requests: list[DownloadRequest] = []
    for market in config.markets:
        for symbol in config.symbols:
            for day in iter_days(config.start_date, config.end_date):
                url = binance_archive_url(market, "trades", symbol, day)
                destination = paths.raw_file(
                    exchange="binance",
                    market=market,
                    dataset="trades",
                    symbol=symbol,
                    day=day,
                    suffix=".zip",
                )
                requests.append(
                    DownloadRequest(
                        exchange="binance",
                        market=market,
                        dataset="trades",
                        symbol=symbol,
                        day=day,
                        url=url,
                        destination=destination,
                    )
                )
    return requests


def binance_archive_url(market: str, dataset: str, symbol: str, day: date) -> str:
    market_path = {
        "spot": "spot",
        "perp": "futures/um",
    }.get(market)
    if market_path is None:
        raise ValueError(f"Unsupported Binance market: {market}")

    return (
        "https://data.binance.vision/data/"
        f"{market_path}/daily/{dataset}/{symbol}/"
        f"{symbol}-{dataset}-{day.isoformat()}.zip"
    )


def download_request(request: DownloadRequest, *, overwrite: bool = False) -> Path:
    request.destination.parent.mkdir(parents=True, exist_ok=True)
    if request.destination.exists() and not overwrite:
        return request.destination

    with tempfile.NamedTemporaryFile(
        dir=request.destination.parent, delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        with (
            urllib.request.urlopen(request.url, timeout=60) as response,
            tmp_path.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        tmp_path.replace(request.destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return request.destination
