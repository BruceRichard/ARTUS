"""Create controlled DecoArt inference configs for rebuttal experiments.

All generated configs share the same object-selection and per-sample seeds.
Only the requested routing or perturbation factor is changed.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _variant(base: Dict[str, Any], name: str, **overrides: Any) -> Dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("generation", {})["experiment_name"] = name
    guidance = config.setdefault("structured_state_guidance", {})
    guidance["enabled"] = True
    guidance["routing_mode"] = "correct"
    guidance["state_noise_std"] = 0.0
    guidance["routing_corruption_ratio"] = 0.0
    guidance["support_corruption_ratio"] = 0.0
    guidance.update(overrides)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="configs/3_TF-Diff/text-eval.yaml",
        help="Base evaluation YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/3_TF-Diff/rebuttal",
        help="Directory for generated controlled configs.",
    )
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.base).read_text())
    variants = {
        "routing_correct": _variant(base, "routing_correct"),
        "routing_random": _variant(base, "routing_random", routing_mode="random"),
        "routing_uniform": _variant(base, "routing_uniform", routing_mode="uniform"),
    }
    for level in (0.01, 0.03, 0.05):
        tag = str(level).replace(".", "p")
        variants[f"state_noise_{tag}"] = _variant(
            base,
            f"state_noise_{tag}",
            state_noise_std=level,
        )
    for ratio in (0.10, 0.20, 0.30):
        tag = str(ratio).replace(".", "p")
        variants[f"routing_corrupt_{tag}"] = _variant(
            base,
            f"routing_corrupt_{tag}",
            routing_corruption_ratio=ratio,
        )
        variants[f"support_corrupt_{tag}"] = _variant(
            base,
            f"support_corrupt_{tag}",
            support_corruption_ratio=ratio,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, config in variants.items():
        path = output_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(path.as_posix())


if __name__ == "__main__":
    main()
