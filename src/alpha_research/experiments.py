"""Experiment run scaffolding."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from alpha_research.config import ExperimentConfig
from alpha_research.manifests import Manifest, write_manifest
from alpha_research.provenance import stable_json_hash
from alpha_research.reporting import write_basic_html_report


def create_run(config: ExperimentConfig, *, artifacts_root: Path = Path("artifacts")) -> Path:
    config_hash = stable_json_hash(config.raw or asdict(config))
    commit = _git_commit()
    dirty = _git_dirty()
    run_id = f"{config.hypothesis_id}.{config_hash[:12]}.{commit[:8]}"
    if dirty:
        run_id += ".dirty"

    run_dir = artifacts_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        json.dumps(config.raw or asdict(config), indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    metrics = {
        "status": "initialized",
        "hypothesis_id": config.hypothesis_id,
        "claim": config.claim,
        "symbols": list(config.symbols),
        "horizons_seconds": list(config.horizons_seconds),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "data_quality.json").write_text("{}\n", encoding="utf-8")
    write_basic_html_report(run_dir=run_dir, title=config.hypothesis_id, metrics=metrics)

    manifest = Manifest.create(
        kind="run",
        id=run_id,
        path=run_dir,
        payload={
            "git_commit": commit,
            "dirty_worktree": dirty,
            "resolved_config_hash": config_hash,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    write_manifest(manifest, run_dir / "manifest.json")
    return run_dir


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return True
    return bool(status.strip())
