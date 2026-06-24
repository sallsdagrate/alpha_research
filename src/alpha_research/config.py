"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from alpha_research.exceptions import ConfigurationError


def load_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON mapping from disk.

    YAML support uses PyYAML when installed. A deliberately small fallback parser
    handles the simple project configs committed in this repository so planning
    commands still work before dependencies are installed.
    """

    if not path.exists():
        raise ConfigurationError(f"Config file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        value = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            value = _parse_simple_yaml(text)
        else:
            value = yaml.safe_load(text)

    if not isinstance(value, dict):
        raise ConfigurationError(f"Config must be a mapping: {path}")
    return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small subset of YAML used by repository configs."""

    result: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if line.startswith("  - "):
            if current_key is None:
                raise ConfigurationError("List item without a preceding key")
            current_value = result.setdefault(current_key, [])
            if not isinstance(current_value, list):
                raise ConfigurationError(f"Key is not a list: {current_key}")
            current_value.append(_coerce_scalar(line[4:].strip()))
            continue

        if line.startswith(" "):
            raise ConfigurationError(
                "Fallback YAML parser supports only top-level scalars/lists"
            )

        if ":" not in line:
            raise ConfigurationError(f"Invalid config line: {raw_line}")

        key, raw_value = line.split(":", 1)
        current_key = key.strip()
        value = raw_value.strip()
        result[current_key] = [] if value == "" else _coerce_scalar(value)

    return result


def _coerce_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


@dataclass(frozen=True)
class DataConfig:
    """Binance archive download configuration."""

    data_root: Path
    symbols: tuple[str, ...]
    markets: tuple[str, ...]
    start_date: date
    end_date: date
    overwrite: bool = False

    @classmethod
    def from_file(cls, path: Path) -> DataConfig:
        raw = load_mapping(path)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> DataConfig:
        required = [
            "data_root",
            "symbols",
            "markets",
            "start_date",
            "end_date",
        ]
        missing = [key for key in required if key not in raw]
        if missing:
            raise ConfigurationError(f"Missing required data config keys: {missing}")

        start = date.fromisoformat(str(raw["start_date"]))
        end = date.fromisoformat(str(raw["end_date"]))
        if end < start:
            raise ConfigurationError("end_date must be on or after start_date")
        markets = tuple(str(item) for item in raw["markets"])
        unsupported = sorted(set(markets) - {"spot", "perp"})
        if unsupported:
            raise ConfigurationError(f"Unsupported markets: {unsupported}")

        return cls(
            data_root=Path(str(raw["data_root"])),
            symbols=tuple(str(item) for item in raw["symbols"]),
            markets=markets,
            start_date=start,
            end_date=end,
            overwrite=bool(raw.get("overwrite", False)),
        )
