"""Command-line interface for local alpha research workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from alpha_research.config import DataConfig
from alpha_research.data.download import download_request, plan_downloads
from alpha_research.exceptions import AlphaResearchError, MissingDependencyError
from alpha_research.logging import configure_logging
from alpha_research.paths import DataPaths


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=getattr(args, "verbose", False))

    try:
        result = args.func(args)
    except MissingDependencyError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 2
    except AlphaResearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0 if result is None else int(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpha")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(required=True)

    data = subparsers.add_parser("data", help="Download raw market data")
    data_sub = data.add_subparsers(required=True)

    download = data_sub.add_parser("download", help="Download Binance trade ZIPs")
    download.add_argument("--config", required=True, type=Path)
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--limit", type=int, default=None)
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=cmd_data_download)

    return parser


def cmd_data_download(args: argparse.Namespace) -> None:
    config = DataConfig.from_file(args.config)
    requests = plan_downloads(config)
    if args.limit is not None:
        requests = requests[: args.limit]

    print_json(
        {
            "action": "plan" if args.dry_run else "download",
            "count": len(requests),
            "requests": [
                {
                    "id": request.id,
                    "url": request.url,
                    "destination": str(request.destination),
                }
                for request in requests
            ],
        }
    )

    if args.dry_run:
        return

    paths = DataPaths(config.data_root)
    paths.ensure()
    for request in requests:
        output = download_request(
            request,
            overwrite=bool(args.force or config.overwrite),
        )
        print(f"downloaded {output}")


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
