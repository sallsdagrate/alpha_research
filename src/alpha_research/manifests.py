"""JSON manifest read/write/discovery utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from alpha_research.provenance import stable_json_hash

ManifestKind = Literal["source", "dataset", "run"]


@dataclass(frozen=True)
class Manifest:
    """Small JSON record describing an input, derived dataset, or run."""

    kind: ManifestKind
    id: str
    path: str
    created_at: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: ManifestKind,
        id: str,
        path: Path,
        payload: dict[str, Any] | None = None,
    ) -> Manifest:
        return cls(
            kind=kind,
            id=id,
            path=str(path),
            created_at=datetime.now(UTC).isoformat(),
            payload=payload or {},
        )

    @property
    def fingerprint(self) -> str:
        return stable_json_hash(asdict(self))


def write_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    data["fingerprint"] = manifest.fingerprint
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_manifests(root: Path, *, kind: str | None = None) -> list[Path]:
    if not root.exists():
        return []
    paths = (
        sorted(root.rglob("*.manifest.json"))
        + sorted(root.rglob("_manifest.json"))
        + sorted(root.rglob("manifest.json"))
    )
    if kind is None:
        return paths

    selected: list[Path] = []
    for path in paths:
        try:
            data = read_manifest(path)
        except json.JSONDecodeError:
            continue
        if data.get("kind") == kind:
            selected.append(path)
    return selected
