"""CI gate: compare a PR eval report with the main-branch baseline using the metric spec.

    python -m lab9.gate --pr reports/eval_workflow.json --baseline reports/baseline_v1.json

exit 0 = merge, exit 1 = blocked. Safety metrics: max 0, no averaging away.
Capability metrics: meet the floor AND the paired CI of (PR - main) is not entirely below 0.
Thresholds may only get stricter on the same eval-set version (ratchet).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from lab8.console import utf8_stdio
from lab9 import stats
from lab9.eval import SPEC, load_spec, per_item


def _finite(value: Any) -> bool:
    return not isinstance(value, Real) or math.isfinite(float(value))


def _metric_value(row: Mapping[str, Any], metric_id: str) -> tuple[bool, Any]:
    row_metrics = row.get("metrics")
    if isinstance(row_metrics, Mapping) and metric_id in row_metrics:
        return True, row_metrics[metric_id]
    return metric_id in row, row.get(metric_id)


def _tags(row: Mapping[str, Any]) -> frozenset[Any]:
    tags = row.get("slices") or []
    if isinstance(tags, (list, tuple, set, frozenset)):
        return frozenset(tags)
    return frozenset({tags})


def _manifest_item_count(manifest: Mapping[str, Any]) -> int | None:
    items = manifest.get("items")
    if isinstance(items, int):
        return items
    if isinstance(items, (list, tuple)):
        return len(items)
    item_count = manifest.get("item_count")
    return item_count if isinstance(item_count, int) else None


def _ratchet_problems(spec: dict, base_spec: dict) -> list[str]:
    problems: list[str] = []
    base_metrics = {m["id"]: m for m in base_spec.get("metrics", []) if "id" in m}
    current_metrics = {m["id"]: m for m in spec.get("metrics", []) if "id" in m}

    for metric_id, old in base_metrics.items():
        current = current_metrics.get(metric_id)
        if current is None:
            problems.append(f"metric {metric_id} disappeared")
            continue
        if old.get("blocking") and not current.get("blocking"):
            problems.append(f"metric {metric_id} relaxed from blocking to non-blocking")

        old_threshold = old.get("threshold") or {}
        threshold = current.get("threshold") or {}
        for key in ("max", "max_increase"):
            if key in old_threshold:
                if key not in threshold:
                    problems.append(f"metric {metric_id} threshold {key} removed")
                elif threshold[key] > old_threshold[key]:
                    problems.append(
                        f"metric {metric_id} threshold {key} loosened "
                        f"from {old_threshold[key]} to {threshold[key]}"
                    )
        if "min" in old_threshold:
            if "min" not in threshold:
                problems.append(f"metric {metric_id} threshold min removed")
            elif threshold["min"] < old_threshold["min"]:
                problems.append(
                    f"metric {metric_id} threshold min loosened "
                    f"from {old_threshold['min']} to {threshold['min']}"
                )
        if old_threshold.get("no_significant_drop") and not threshold.get("no_significant_drop"):
            problems.append(f"metric {metric_id} dropped no_significant_drop requirement")

    base_slices = {s["name"]: s for s in base_spec.get("slices", []) if "name" in s}
    current_slices = {s["name"]: s for s in spec.get("slices", []) if "name" in s}
    for name, old in base_slices.items():
        current = current_slices.get(name)
        if current is None:
            problems.append(f"slice {name} disappeared")
            continue
        if old.get("blocking") and not current.get("blocking"):
            problems.append(f"slice {name} relaxed from blocking to non-blocking")
        if old.get("metric") != current.get("metric"):
            problems.append(f"slice {name} metric changed from {old.get('metric')} to {current.get('metric')}")
        if "min" in old:
            if "min" not in current:
                problems.append(f"slice {name} min threshold removed")
            elif current["min"] < old["min"]:
                problems.append(
                    f"slice {name} min threshold loosened from {old['min']} to {current['min']}"
                )
        if old.get("no_significant_drop") and not current.get("no_significant_drop"):
            problems.append(f"slice {name} dropped no_significant_drop requirement")
    return problems


def integrity_check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> list[str]:
    """Return problems that make a PR report unsafe to compare with its baseline."""
    problems: list[str] = []
    metrics = pr.get("metrics") or {}
    for metric_id, value in metrics.items():
        if isinstance(value, Real) and not _finite(value):
            problems.append(f"metric {metric_id} is NaN or inf")

    manifest = pr.get("manifest") or {}
    item_count = _manifest_item_count(manifest)
    seeds = manifest.get("seeds", pr.get("seeds"))
    rows = pr.get("rows") or []
    if item_count is None or not isinstance(seeds, (list, tuple)):
        problems.append("manifest is missing item count or seed list")
    else:
        expected_rows = item_count * len(seeds)
        if len(rows) != expected_rows:
            problems.append(f"row count mismatch: got {len(rows)}, expected {expected_rows}")

    def row_pairs(report: dict) -> set[tuple[Any, Any]]:
        pairs = set()
        for row in report.get("rows") or []:
            if not isinstance(row, Mapping):
                problems.append("report contains a malformed row")
                continue
            pairs.add((row.get("item_id"), row.get("seed")))
        return pairs

    pr_pairs = row_pairs(pr)
    base_pairs = row_pairs(base)
    if len(pr_pairs) != len(rows):
        problems.append("duplicate (item_id, seed) rows")
    if pr_pairs != base_pairs:
        problems.append("(item_id, seed) set differs from baseline (cherry-picked seeds/items)")

    if pr.get("evaluator_failures"):
        problems.append(f"evaluator_failures non-empty: {pr['evaluator_failures']}")
    graded_ids = [metric["id"] for metric in spec.get("metrics", []) if "id" in metric]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for metric_id in graded_ids:
            present, value = _metric_value(row, metric_id)
            if present and value is None:
                problems.append(
                    f"row {row.get('item_id')} seed {row.get('seed')} has None for metric {metric_id}"
                )
            elif present and isinstance(value, Real) and not _finite(value):
                problems.append(
                    f"row {row.get('item_id')} seed {row.get('seed')} metric {metric_id} is NaN or inf"
                )

    if base_spec is not None:
        base_version = base_spec.get("eval_set_version")
        current_version = spec.get("eval_set_version")
        if base_version is None or current_version is None or base_version == current_version:
            problems.extend(_ratchet_problems(spec, base_spec))

    base_tags = {}
    for row in base.get("rows") or []:
        if isinstance(row, Mapping):
            base_tags[(row.get("item_id"), row.get("seed"))] = _tags(row)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = (row.get("item_id"), row.get("seed"))
        if key in base_tags and _tags(row) != base_tags[key]:
            problems.append(f"item {key[0]} seed {key[1]} slice tags differ from baseline")
            break
    return problems


def _per_item_values(report: dict, metric: str, slice_name: str | None = None) -> dict[str, float]:
    safe_report = {**report, "rows": report.get("rows") or []}
    values = per_item(safe_report, metric, slice_name=slice_name)
    if values or not metric.startswith("pass_hat_"):
        return values

    try:
        k = int(metric.removeprefix("pass_hat_"))
    except ValueError:
        return values

    trials: dict[str, list[bool]] = defaultdict(list)
    for row in report.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        if slice_name is not None and slice_name not in _tags(row):
            continue
        if row.get("pass_rate") is not None:
            trials[str(row["item_id"])].append(bool(row["pass_rate"]))
    return {
        item_id: stats.pass_hat_k(len(results), sum(results), k)
        for item_id, results in trials.items()
        if len(results) >= k
    }


def slice_check(pr: dict, base: dict, spec: dict) -> tuple[list[str], list[str]]:
    """Evaluate every configured slice against its floor and paired regression guard."""
    blocked: list[str] = []
    lines: list[str] = []
    bootstrap = spec.get("bootstrap", {})
    for slice_spec in spec.get("slices", []):
        name = slice_spec["name"]
        metric = slice_spec["metric"]
        minimum = slice_spec.get("min")
        blocking = slice_spec.get("blocking", False)
        slice_id = f"slice:{name}/{metric}"
        values = _per_item_values(pr, metric, slice_name=name)
        baseline_values = _per_item_values(base, metric, slice_name=name)
        notes: list[str] = []
        verdict = "ok"

        if not values:
            verdict = "missing"
            notes.append("no PR items in slice")
            value_text = "missing"
        else:
            mean_value = sum(values.values()) / len(values)
            value_text = f"{mean_value:.4f}"
            if not math.isfinite(mean_value):
                verdict = "fail"
                notes.append("non-finite slice metric")
            elif minimum is not None and mean_value < minimum:
                verdict = "fail"

        if slice_spec.get("no_significant_drop"):
            keys = sorted(set(values) & set(baseline_values))
            if not keys:
                verdict = "missing"
                notes.append("no paired baseline items")
            else:
                delta, low, high = stats.paired_bootstrap(
                    [values[key] for key in keys],
                    [baseline_values[key] for key in keys],
                    n_resamples=bootstrap.get("n_resamples", 1000),
                    rng=bootstrap.get("rng", 2027),
                    confidence=bootstrap.get("confidence", 0.95),
                )
                notes.append(f"Δ {delta:+.3f} CI [{low:+.3f}, {high:+.3f}]")
                if high < 0:
                    verdict = "fail"

        if verdict != "ok" and blocking:
            blocked.append(slice_id)
        label = "BLOCK" if verdict != "ok" and blocking else ("warn" if verdict != "ok" else "ok")
        lines.append(
            f"  {slice_id:<30} {value_text:<8} min: {minimum!s:<10} "
            f"{' '.join(notes):<38} {label}"
        )
    return blocked, lines


def check(pr: dict, base: dict, spec: dict, base_spec: dict | None = None) -> tuple[list[str], list[str]]:
    pr_sha = pr.get("manifest", {}).get("eval_set_sha")
    base_sha = base.get("manifest", {}).get("eval_set_sha")
    if not pr_sha or not base_sha or pr_sha != base_sha:
        return ["eval_set_sha"], ["  EVAL SET differs from the baseline's: re-run the baseline on the same eval set"]
    problems = integrity_check(pr, base, spec, base_spec)
    if problems:
        return ["integrity"], [f"  INTEGRITY {problem}" for problem in problems]

    blocked: list[str] = []
    lines: list[str] = []
    bootstrap = spec.get("bootstrap", {})
    for metric_spec in spec.get("metrics", []):
        metric_id = metric_spec["id"]
        threshold = metric_spec.get("threshold") or {}
        value = (pr.get("metrics") or {}).get(metric_id)
        verdict, note = "ok", ""
        if value is None:
            verdict = "missing"
        elif "max" in threshold and value > threshold["max"]:
            verdict = "fail"
        elif "min" in threshold and value < threshold["min"]:
            verdict = "fail"
        elif "max_increase" in threshold:
            base_value = (base.get("metrics") or {}).get(metric_id)
            if base_value is not None and value > base_value * (1 + threshold["max_increase"]):
                verdict = "fail"

        if threshold.get("no_significant_drop") and value is not None:
            pr_values = _per_item_values(pr, metric_id)
            base_values = _per_item_values(base, metric_id)
            keys = sorted(set(pr_values) & set(base_values))
            if not keys:
                verdict = "missing"
                note = "no paired baseline items"
            else:
                delta, low, high = stats.paired_bootstrap(
                    [pr_values[key] for key in keys],
                    [base_values[key] for key in keys],
                    n_resamples=bootstrap.get("n_resamples", 1000),
                    rng=bootstrap.get("rng", 2027),
                    confidence=bootstrap.get("confidence", 0.95),
                )
                note = f"Δ {delta:+.3f} CI [{low:+.3f}, {high:+.3f}]"
                if high < 0:
                    verdict = "fail"

        blocking = metric_spec.get("blocking", False)
        if verdict != "ok" and blocking:
            blocked.append(metric_id)
        label = "ok" if verdict == "ok" else (
            "MISSING" if verdict == "missing" else ("BLOCK" if blocking else "warn")
        )
        lines.append(
            f"  {metric_id:<22} {value!s:<8} {json.dumps(threshold):<40} {note:<28} {label}"
        )

    if spec.get("slices"):
        slice_blocked, slice_lines = slice_check(pr, base, spec)
        blocked.extend(slice_blocked)
        lines.extend(slice_lines)
    return blocked, lines


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    parser = argparse.ArgumentParser(prog="lab9.gate")
    parser.add_argument("--pr", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--spec", default=str(SPEC))
    parser.add_argument("--base-spec", help="metric spec on main, for the ratchet check (M7)")
    args = parser.parse_args(argv)
    pr = json.loads(Path(args.pr).read_text(encoding="utf-8"))
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    base_spec = load_spec(Path(args.base_spec)) if args.base_spec else None
    blocked, lines = check(pr, base, load_spec(Path(args.spec)), base_spec)
    print(f"gate: PR {pr['run_id']} vs main {base['run_id']} · eval set {pr['manifest']['eval_set_sha']}")
    print("\n".join(lines))
    if blocked:
        print(f"exit 1 — blocked: {', '.join(blocked)}")
        return 1
    print("exit 0 — merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
