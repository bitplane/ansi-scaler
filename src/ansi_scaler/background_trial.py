from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from ansi_scaler.config import BackgroundSettings, RunConfig, load_yaml
from ansi_scaler.manifests import read_jsonl, resolve_path, write_jsonl
from ansi_scaler.stages.background import run_background


def _spread(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit >= len(records):
        return records
    if limit == 1:
        return records[:1]
    return [records[round(index * (len(records) - 1) / (limit - 1))] for index in range(limit)]


def _report(
    destination: Path,
    config: RunConfig,
    sources: list[dict[str, Any]],
    results: dict[str, dict[str, dict[str, Any]]],
) -> None:
    columns = "".join(f"<th>{html.escape(name)}</th>" for name in results)
    rows = []
    for source in sources:
        title = source.get("concept_name") or source.get("prompt") or source["id"][:12]
        original = resolve_path(source["artifact"], config.data_dir).resolve().as_uri()
        cells = []
        for records in results.values():
            record = records.get(source["id"])
            if record:
                uri = resolve_path(record["artifact"], config.data_dir).resolve().as_uri()
                cells.append(f'<td><img src="{uri}" loading="lazy"></td>')
            else:
                cells.append('<td class="failed">failed</td>')
        rows.append(
            f'<tr><th colspan="{len(results) + 1}">{html.escape(str(title))}</th></tr>'
            f'<tr><td><img src="{original}" loading="lazy"></td>{"".join(cells)}</tr>'
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Background provider comparison</title>
<style>
body{{font:16px system-ui;background:#0c1118;color:#dce6f2;margin:24px}}table{{width:100%;border-spacing:12px}}
th{{text-align:left;color:#62d8e4;padding-top:20px}}td{{width:{100 / (len(results) + 1):.2f}%;height:360px;
background-color:#bcc3ca;background-image:linear-gradient(45deg,#aab2ba 25%,transparent 25%),
linear-gradient(-45deg,#aab2ba 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#aab2ba 75%),
linear-gradient(-45deg,transparent 75%,#aab2ba 75%);background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0}}
img{{display:block;width:100%;height:360px;object-fit:contain}}.failed{{color:#ff7185;text-align:center}}
</style></head><body><h1>Background provider comparison</h1><table>
<thead><tr><th>Original</th>{columns}</tr></thead><tbody>{"".join(rows)}</tbody></table></body></html>""",
        encoding="utf-8",
    )


def run_background_trial(config: RunConfig, trial_path: Path, *, force: bool = False) -> Path:
    trial = load_yaml(trial_path)
    limit = int(trial.get("limit", 12))
    sources = _spread(list(read_jsonl(config.manifest_dir / "rasters.jsonl")), limit)
    if not sources:
        raise ValueError(f"No generated rasters found in {config.manifest_dir}")

    root = config.data_dir / "runs" / "trials" / "background-providers"
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in trial["candidates"]:
        name = str(candidate["name"])
        candidate_config = config.model_copy(deep=True)
        candidate_config.name = f"trials/background-providers/{name}"
        candidate_config.limit = None
        candidate_config.background = BackgroundSettings.model_validate(candidate["background"])
        candidate_config.manifest_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(candidate_config.manifest_dir / "rasters.jsonl", sources)
        run_background(candidate_config, force=force, retry_errors=True)
        results[name] = {
            record["parent_id"]: record for record in read_jsonl(candidate_config.manifest_dir / "backgrounds.jsonl")
        }

    report = root / "comparison.html"
    _report(report, config, sources, results)
    return report
