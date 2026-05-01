"""Build DecoArt physical/routing metadata from JSON or generated output files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from state_metrics import DecoArtMetricConfig, analyze_path


def _iter_input_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(input_path.as_posix())
    files = sorted(input_path.rglob(pattern))
    return [path for path in files if path.name != "meta.json"]


def _mean(records: Iterable[Dict[str, Any]], key: str) -> float:
    values = [float(record["summary"][key]) for record in records]
    return sum(values) / max(1, len(values))


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"count": 0}

    groups = {
        "d_le_2": [r for r in records if r["summary"]["max_depth"] <= 2],
        "d_3_4": [r for r in records if 3 <= r["summary"]["max_depth"] <= 4],
        "d_ge_5": [r for r in records if r["summary"]["max_depth"] >= 5],
    }
    route_counts: Dict[str, int] = {"structure": 0, "physical": 0, "detail": 0}
    for record in records:
        for item in record["routing"]:
            route_counts[item["group"]] = route_counts.get(item["group"], 0) + 1

    return {
        "count": len(records),
        "pv_rate": sum(1 for r in records if r["summary"]["pv_valid"]) / len(records),
        "align": _mean(records, "align"),
        "normal": _mean(records, "normal"),
        "penetration": _mean(records, "penetration"),
        "j_cost": _mean(records, "j_cost"),
        "route_counts": route_counts,
        "depth_groups": {
            name: {
                "count": len(group_records),
                "pv_rate": (
                    sum(1 for r in group_records if r["summary"]["pv_valid"]) / len(group_records)
                    if group_records else 0.0
                ),
                "align": _mean(group_records, "align") if group_records else 0.0,
                "normal": _mean(group_records, "normal") if group_records else 0.0,
                "penetration": _mean(group_records, "penetration") if group_records else 0.0,
            }
            for name, group_records in groups.items()
        },
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_part_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "source",
        "dfn",
        "parent_dfn",
        "depth",
        "group",
        "align",
        "normal",
        "penetration_positive",
        "j_cost",
        "structure_score",
        "physical_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            route_by_dfn = {route["dfn"]: route for route in record["routing"]}
            for metric in record["parts"]:
                route = route_by_dfn.get(metric["dfn"], {})
                writer.writerow({
                    "source": record["source"],
                    "dfn": metric["dfn"],
                    "parent_dfn": metric["parent_dfn"],
                    "depth": route.get("depth", 0),
                    "group": route.get("group", "detail"),
                    "align": metric["align"],
                    "normal": metric["normal"],
                    "penetration_positive": metric["penetration_positive"],
                    "j_cost": metric["j_cost"],
                    "structure_score": route.get("structure_score", 0.0),
                    "physical_score": route.get("physical_score", 0.0),
                })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DecoArt physical/routing metadata.")
    parser.add_argument("--input", required=True, help="A JSON/output.dat file or a directory.")
    parser.add_argument("--pattern", default="*.json", help="Pattern used when --input is a directory.")
    parser.add_argument("--output-dir", default="experiments/decoart/outputs", help="Directory for metrics.")
    parser.add_argument("--bbox-format", default="center_size", choices=["center_size", "min_max"])
    parser.add_argument("--samples-per-face", type=int, default=9)
    parser.add_argument("--dmax", type=float, default=0.10)
    parser.add_argument("--alpha-normal", type=float, default=0.50)
    parser.add_argument("--beta-penetration", type=float, default=1.00)
    parser.add_argument("--tau-pen", type=float, default=0.02)
    parser.add_argument("--theta-tan", type=float, default=0.10)
    parser.add_argument("--theta-norm", type=float, default=0.25)
    parser.add_argument("--theta-pen", type=float, default=0.02)
    parser.add_argument("--no-nearby-support", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DecoArtMetricConfig(
        bbox_format=args.bbox_format,
        samples_per_face=args.samples_per_face,
        dmax=args.dmax,
        alpha_normal=args.alpha_normal,
        beta_penetration=args.beta_penetration,
        tau_pen=args.tau_pen,
        theta_tan=args.theta_tan,
        theta_norm=args.theta_norm,
        theta_pen=args.theta_pen,
        include_nearby_support=not args.no_nearby_support,
    )

    files = _iter_input_files(Path(args.input), args.pattern)
    records = []
    failures = []
    for path in files:
        try:
            records.append(analyze_path(path, cfg))
        except Exception as exc:
            failures.append({"source": path.as_posix(), "error": str(exc)})

    _write_jsonl(output_dir / "decoart_state_metrics.jsonl", records)
    _write_part_csv(output_dir / "decoart_part_metrics.csv", records)
    (output_dir / "decoart_summary.json").write_text(
        json.dumps({
            "input": Path(args.input).as_posix(),
            "pattern": args.pattern,
            "config": cfg.to_dict(),
            "aggregate": _aggregate(records),
            "failures": failures,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Processed {len(records)} files, failed {len(failures)}.")
    print(f"Wrote {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
