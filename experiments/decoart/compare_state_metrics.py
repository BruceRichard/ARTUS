"""Compare two DecoArt metric JSONL files, e.g. no-guidance vs guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {
            "count": 0,
            "pv_rate": 0.0,
            "align": 0.0,
            "normal": 0.0,
            "penetration": 0.0,
            "j_cost": 0.0,
        }
    return {
        "count": float(len(records)),
        "pv_rate": sum(1 for r in records if r["summary"]["pv_valid"]) / len(records),
        "align": sum(float(r["summary"]["align"]) for r in records) / len(records),
        "normal": sum(float(r["summary"]["normal"]) for r in records) / len(records),
        "penetration": sum(float(r["summary"]["penetration"]) for r in records) / len(records),
        "j_cost": sum(float(r["summary"]["j_cost"]) for r in records) / len(records),
    }


def _delta(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
    return {
        key: float(after[key] - before[key])
        for key in ["pv_rate", "align", "normal", "penetration", "j_cost"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DecoArt state metric JSONL files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline = _aggregate(_read_jsonl(Path(args.baseline)))
    ours = _aggregate(_read_jsonl(Path(args.ours)))
    result = {
        "baseline": baseline,
        "ours": ours,
        "delta_ours_minus_baseline": _delta(ours, baseline),
    }

    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
