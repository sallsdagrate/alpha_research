"""Command-line interface for local alpha research workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alpha_research.config import DataConfig, ExperimentConfig
from alpha_research.data.align import build_spot_perp_1s
from alpha_research.data.download import download_request, plan_downloads
from alpha_research.data.normalize import normalize_binance_trades
from alpha_research.data.quality import validate_parquet_dataset
from alpha_research.exceptions import AlphaResearchError, MissingDependencyError
from alpha_research.experiments import create_run
from alpha_research.logging import configure_logging
from alpha_research.manifests import iter_manifests, read_manifest
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

    data = subparsers.add_parser("data", help="Data ingestion and transformation")
    data_sub = data.add_subparsers(required=True)

    fetch = data_sub.add_parser("fetch", help="Download or plan raw data")
    fetch.add_argument("--config", required=True, type=Path)
    fetch.add_argument("--dry-run", action="store_true")
    fetch.add_argument("--limit", type=int, default=None)
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=cmd_data_fetch)

    normalize = data_sub.add_parser("normalize", help="Normalize raw files")
    normalize.add_argument("--config", required=True, type=Path)
    normalize.add_argument("--market", choices=["spot", "perp"], required=True)
    normalize.add_argument("--symbol", required=True)
    normalize.add_argument("--date", required=True)
    normalize.add_argument("--force", action="store_true")
    normalize.set_defaults(func=cmd_data_normalize)

    validate = data_sub.add_parser("validate", help="Validate canonical data")
    validate.add_argument("--dataset", default="trades")
    validate.add_argument("--path", required=True, type=Path)
    validate.set_defaults(func=cmd_data_validate)

    build = data_sub.add_parser("build", help="Build curated research tables")
    build.add_argument("--dataset", required=True, choices=["spot_perp_1s"])
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--symbol", required=True)
    build.add_argument("--force", action="store_true")
    build.set_defaults(func=cmd_data_build)

    manifest = subparsers.add_parser("manifest", help="Inspect JSON manifests")
    manifest_sub = manifest.add_subparsers(required=True)

    manifest_list = manifest_sub.add_parser("list", help="List manifests")
    manifest_list.add_argument("--root", type=Path, default=Path("."))
    manifest_list.add_argument("--kind", choices=["source", "dataset", "run"])
    manifest_list.set_defaults(func=cmd_manifest_list)

    manifest_show = manifest_sub.add_parser("show", help="Show a manifest by path")
    manifest_show.add_argument("path", type=Path)
    manifest_show.set_defaults(func=cmd_manifest_show)

    experiment = subparsers.add_parser("experiment", help="Experiment workflows")
    experiment_sub = experiment.add_subparsers(required=True)

    run = experiment_sub.add_parser("run", help="Initialize/run an experiment")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--initialize-only", action="store_true")
    run.set_defaults(func=cmd_experiment_run)

    report = subparsers.add_parser("report", help="Report workflows")
    report_sub = report.add_subparsers(required=True)

    report_build = report_sub.add_parser("build", help="Confirm report artifact exists")
    report_build.add_argument("--run-dir", required=True, type=Path)
    report_build.set_defaults(func=cmd_report_build)

    return parser


def cmd_data_fetch(args: argparse.Namespace) -> None:
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


def cmd_data_normalize(args: argparse.Namespace) -> None:
    config = DataConfig.from_file(args.config)
    paths = DataPaths(config.data_root)
    raw_path = paths.raw_file(
        exchange=config.exchange,
        market=args.market,
        dataset="trades",
        symbol=args.symbol,
        day=_date_from_arg(args.date),
        suffix=".zip",
    )
    result = normalize_binance_trades(
        raw_path=raw_path,
        data_root=config.data_root,
        exchange=config.exchange,
        market=args.market,
        symbol=args.symbol,
        latency_ms=config.latency_ms,
        force=args.force,
    )
    print_json(asdict(result))


def cmd_data_validate(args: argparse.Namespace) -> int:
    result = validate_parquet_dataset(args.dataset, args.path)
    print_json(asdict(result))
    return 0 if result.passed else 1


def cmd_data_build(args: argparse.Namespace) -> None:
    config = DataConfig.from_file(args.config)
    if args.dataset != "spot_perp_1s":
        raise ValueError(f"Unsupported curated dataset: {args.dataset}")
    result = build_spot_perp_1s(
        data_root=config.data_root,
        exchange=config.exchange,
        symbol=args.symbol,
        start_date=config.start_date,
        end_date=config.end_date,
        force=args.force,
    )
    print_json(asdict(result))


def cmd_manifest_list(args: argparse.Namespace) -> None:
    manifests = iter_manifests(args.root, kind=args.kind)
    rows: list[dict[str, Any]] = []
    for path in manifests:
        data = read_manifest(path)
        rows.append(
            {
                "path": str(path),
                "kind": data.get("kind"),
                "id": data.get("id"),
                "created_at": data.get("created_at"),
                "fingerprint": data.get("fingerprint"),
            }
        )
    print_json(rows)


def cmd_manifest_show(args: argparse.Namespace) -> None:
    print_json(read_manifest(args.path))


def cmd_experiment_run(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_file(args.config)
    run_dir = create_run(config)
    if not args.initialize_only:
        print(
            "experiment execution is scaffolded; implement signal evaluation after "
            "curated data exists"
        )
    print_json({"run_dir": str(run_dir)})


def cmd_report_build(args: argparse.Namespace) -> int:
    report = args.run_dir / "report.html"
    if not report.exists():
        print(f"missing report: {report}", file=sys.stderr)
        return 1
    print_json({"report": str(report)})
    return 0


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _date_from_arg(value: str):
    from datetime import date

    return date.fromisoformat(value)
