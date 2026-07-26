"""Aggregate controlled DecoArt runs into rebuttal-ready statistics."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from .state_metrics import DecoArtMetricConfig, analyze_path
except ImportError:
    from state_metrics import DecoArtMetricConfig, analyze_path


METRIC_KEYS = ("pv_rate", "align", "normal", "penetration", "j_cost")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def _mean_std(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "mean": _mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _seed_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {key: 0.0 for key in METRIC_KEYS}
    return {
        "pv_rate": _mean(float(record["summary"]["pv_valid"]) for record in records),
        "align": _mean(float(record["summary"]["align"]) for record in records),
        "normal": _mean(float(record["summary"]["normal"]) for record in records),
        "penetration": _mean(float(record["summary"]["penetration"]) for record in records),
        "j_cost": _mean(float(record["summary"]["j_cost"]) for record in records),
    }


def _load_guidance_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else []


def _guidance_seed_summary(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    active = sum(int(item.get("active_count", 0)) for item in logs)
    if active == 0:
        return {
            "active_count": 0,
            "j_pre": 0.0,
            "j_post": 0.0,
            "reduction_percent": 0.0,
            "accepted_rate": 0.0,
            "route_counts": {},
        }
    j_pre = sum(float(item.get("j_pre", 0.0)) * int(item.get("active_count", 0)) for item in logs) / active
    j_post = sum(float(item.get("j_post", 0.0)) * int(item.get("active_count", 0)) for item in logs) / active
    requested = sum(int(item.get("requested_steps", 0)) for item in logs)
    accepted = sum(int(item.get("accepted_steps", 0)) for item in logs)
    route_counts: Counter[str] = Counter()
    for item in logs:
        route_counts.update(item.get("route_counts", {}))
    return {
        "active_count": active,
        "j_pre": j_pre,
        "j_post": j_post,
        "reduction_percent": 100.0 * (j_pre - j_post) / max(abs(j_pre), 1.0e-8),
        "accepted_rate": accepted / max(1, requested),
        "route_counts": dict(route_counts),
    }


def _condition_summary(condition_dir: Path, cfg: DecoArtMetricConfig) -> Dict[str, Any]:
    by_seed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    guidance_by_seed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    failures = []
    output_files = sorted(condition_dir.glob("*/*/output.dat"))
    for path in output_files:
        seed = path.parent.name
        try:
            by_seed[seed].append(analyze_path(path, cfg))
        except Exception as exc:
            failures.append({"source": path.as_posix(), "error": str(exc)})
        guidance_by_seed[seed].extend(
            _load_guidance_log(path.parent / "structured_guidance_log.json")
        )

    per_seed = {seed: _seed_metrics(records) for seed, records in sorted(by_seed.items())}
    aggregate = {
        key: _mean_std(metrics[key] for metrics in per_seed.values())
        for key in METRIC_KEYS
    }
    guidance_per_seed = {
        seed: _guidance_seed_summary(logs)
        for seed, logs in sorted(guidance_by_seed.items())
    }
    guidance = {
        key: _mean_std(item[key] for item in guidance_per_seed.values())
        for key in ("j_pre", "j_post", "reduction_percent", "accepted_rate")
    }
    route_counts: Counter[str] = Counter()
    for item in guidance_per_seed.values():
        route_counts.update(item["route_counts"])
    guidance["route_counts"] = dict(route_counts)

    return {
        "condition": condition_dir.name,
        "object_runs": len(output_files),
        "seed_count": len(per_seed),
        "metrics": aggregate,
        "guidance": guidance,
        "per_seed": per_seed,
        "guidance_per_seed": guidance_per_seed,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="elog/final_output",
        help="Root containing one directory per controlled condition.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional condition-directory names; defaults to every subdirectory.",
    )
    parser.add_argument(
        "--output",
        default="experiments/decoart/outputs/rebuttal_summary.json",
    )
    args = parser.parse_args()

    root = Path(args.root)
    condition_dirs = (
        [root / name for name in args.conditions]
        if args.conditions
        else sorted(path for path in root.iterdir() if path.is_dir())
    )
    metric_cfg = DecoArtMetricConfig()
    result = {
        "root": root.as_posix(),
        "conditions": [
            _condition_summary(condition_dir, metric_cfg)
            for condition_dir in condition_dirs
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output.as_posix())


if __name__ == "__main__":
    main()
