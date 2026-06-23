"""Research report helpers."""

from __future__ import annotations

import json
from pathlib import Path


def write_basic_html_report(*, run_dir: Path, title: str, metrics: dict[str, object]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_html = "".join(
        f"<li><code>{key}</code>: {value}</li>" for key, value in sorted(metrics.items())
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>"
        f"<h1>{title}</h1><ul>{metrics_html}</ul>"
        "<h2>Raw metrics</h2>"
        f"<pre>{json.dumps(metrics, indent=2, sort_keys=True)}</pre>"
        "</body></html>"
    )
    path = run_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
