"""Evaluate decoded-mesh contact errors and correlate them with state cost.

Generated ``output.dat`` files contain both the predicted structured state and
the decoded canonical part meshes. This script fits every decoded mesh into its
predicted box, evaluates contact/support consistency on mesh surface samples,
and reports object-level Spearman correlations with the structured-state cost.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from .state_metrics import (
        EPS,
        DecoArtMetricConfig,
        _aabb_gap,
        _clearance_margin,
        _motion_direction,
        _part_from_raw,
        _tree_info,
        analyze_parts,
    )
except ImportError:
    from state_metrics import (
        EPS,
        DecoArtMetricConfig,
        _aabb_gap,
        _clearance_margin,
        _motion_direction,
        _part_from_raw,
        _tree_info,
        analyze_parts,
    )


def _mesh_arrays(mesh: Any) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(mesh, "geometry") and not hasattr(mesh, "vertices"):
        geometries = list(mesh.geometry.values())
        vertices = []
        faces = []
        offset = 0
        for geometry in geometries:
            v = np.asarray(geometry.vertices, dtype=np.float64)
            f = np.asarray(geometry.faces, dtype=np.int64)
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        return np.concatenate(vertices), np.concatenate(faces)
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def _fit_vertices(vertices: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    source_min = vertices.min(axis=0)
    source_max = vertices.max(axis=0)
    source_size = np.maximum(source_max - source_min, 1.0e-6)
    target_min = center - size * 0.5
    return target_min + size * ((vertices - source_min) / source_size)


def _sample_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=-1)
    valid = double_area > EPS
    if not np.any(valid):
        raise ValueError("Decoded mesh has no non-degenerate faces")
    probabilities = np.where(valid, double_area, 0.0)
    probabilities /= probabilities.sum()
    face_index = rng.choice(len(faces), size=count, replace=True, p=probabilities)
    selected = triangles[face_index]
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    points = (
        (1.0 - r1)[:, None] * selected[:, 0]
        + (r1 * (1.0 - r2))[:, None] * selected[:, 1]
        + (r1 * r2)[:, None] * selected[:, 2]
    )
    normals = cross[face_index]
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), EPS)
    return points, normals


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: Iterable[float], right: Iterable[float]) -> float:
    left = np.asarray(list(left), dtype=np.float64)
    right = np.asarray(list(right), dtype=np.float64)
    if len(left) < 2 or np.std(left) < EPS or np.std(right) < EPS:
        return 0.0
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _load_raw(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as fp:
        data = pickle.load(fp)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("output.dat must contain a list of generated parts")
    return data


def analyze_decoded_mesh(
    path: Path,
    cfg: DecoArtMetricConfig,
    *,
    surface_samples: int = 1024,
    contact_samples: int = 128,
    seed: int = 2026,
) -> Dict[str, Any]:
    raw = _load_raw(path)
    parts = [_part_from_raw(part, idx, cfg) for idx, part in enumerate(raw)]
    parents, depth, _ = _tree_info(parts)
    rng = np.random.default_rng(seed)

    sampled = []
    for raw_part, part in zip(raw, parts):
        if "mesh" not in raw_part:
            raise ValueError("Generated part does not contain a decoded mesh")
        vertices, faces = _mesh_arrays(raw_part["mesh"])
        vertices = _fit_vertices(vertices, part["center"], part["size"])
        sampled.append(_sample_mesh(vertices, faces, surface_samples, rng))

    metrics = []
    for idx, part in enumerate(parts):
        parent_idx = parents[idx]
        if parent_idx is None:
            continue
        motion = _motion_direction(part, parts[parent_idx])
        points, normals = sampled[idx]
        projection = points @ motion
        projection = (projection - projection.min()) / max(float(np.ptp(projection)), EPS)
        facing = normals @ motion
        contact_score = projection + 0.5 * facing
        selected = np.argpartition(contact_score, -min(contact_samples, len(points)))[
            -min(contact_samples, len(points)):
        ]
        q = points[selected]
        m = normals[selected]

        support_indices = [parent_idx]
        for support_idx, support_part in enumerate(parts):
            if support_idx in {idx, parent_idx}:
                continue
            if _aabb_gap(part, support_part) <= cfg.neighbor_margin:
                support_indices.append(support_idx)
        support_points = np.concatenate([sampled[support_idx][0] for support_idx in support_indices])
        support_normals = np.concatenate([sampled[support_idx][1] for support_idx in support_indices])

        distance = np.linalg.norm(q[:, None, :] - support_points[None, :, :], axis=-1)
        normal_pair = 1.0 + np.einsum("qd,sd->qs", m, support_normals)
        pair_score = distance / max(cfg.dmax, EPS) + cfg.alpha_normal * normal_pair
        nearest = pair_score.argmin(axis=1)
        s = support_points[nearest]
        n = support_normals[nearest]
        diff = q - s
        normal_projection = np.sum(diff * n, axis=-1, keepdims=True) * n
        tangent = np.linalg.norm(diff - normal_projection, axis=-1)
        normal_error = 1.0 + np.sum(n * m, axis=-1)
        penetration = np.maximum(
            -np.sum(n * diff, axis=-1) - _clearance_margin(part, cfg),
            0.0,
        )
        metrics.append({
            "part_index": idx,
            "align": float(tangent.mean()),
            "normal": float(normal_error.mean()),
            "penetration": float(penetration.mean()),
        })

    align = float(np.mean([item["align"] for item in metrics])) if metrics else 0.0
    normal = float(np.mean([item["normal"] for item in metrics])) if metrics else 0.0
    penetration = float(np.mean([item["penetration"] for item in metrics])) if metrics else 0.0
    state = analyze_parts(parts, cfg)
    aggregate_error = (
        align / max(cfg.theta_tan, EPS)
        + normal / max(cfg.theta_norm, EPS)
        + penetration / max(cfg.theta_pen, EPS)
    )
    return {
        "source": path.as_posix(),
        "state_j_cost": float(state["summary"]["j_cost"]),
        "state_pv_valid": bool(state["summary"]["pv_valid"]),
        "mesh_align": align,
        "mesh_normal": normal,
        "mesh_penetration": penetration,
        "mesh_aggregate_error": aggregate_error,
        "parts": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Condition directory containing output.dat files.")
    parser.add_argument("--pattern", default="*/*/output.dat")
    parser.add_argument("--surface-samples", type=int, default=1024)
    parser.add_argument("--contact-samples", type=int, default=128)
    parser.add_argument(
        "--output",
        default="experiments/decoart/outputs/mesh_state_correlation.json",
    )
    parser.add_argument(
        "--dynamic-results",
        default=None,
        help="Optional JSON list containing source and success fields.",
    )
    args = parser.parse_args()

    cfg = DecoArtMetricConfig()
    records = []
    failures = []
    for index, path in enumerate(sorted(Path(args.input).glob(args.pattern))):
        try:
            records.append(
                analyze_decoded_mesh(
                    path,
                    cfg,
                    surface_samples=args.surface_samples,
                    contact_samples=args.contact_samples,
                    seed=2026 + index,
                )
            )
        except Exception as exc:
            failures.append({"source": path.as_posix(), "error": str(exc)})

    summary: Dict[str, Any] = {
        "count": len(records),
        "spearman_state_vs_mesh_aggregate": _spearman(
            (item["state_j_cost"] for item in records),
            (item["mesh_aggregate_error"] for item in records),
        ),
        "spearman_state_vs_mesh_penetration": _spearman(
            (item["state_j_cost"] for item in records),
            (item["mesh_penetration"] for item in records),
        ),
    }
    if args.dynamic_results:
        dynamic = json.loads(Path(args.dynamic_results).read_text())
        success_by_source = {item["source"]: bool(item["success"]) for item in dynamic}
        successful = [
            item["state_j_cost"] for item in records
            if success_by_source.get(item["source"]) is True
        ]
        failed = [
            item["state_j_cost"] for item in records
            if success_by_source.get(item["source"]) is False
        ]
        summary["dynamic_success_state_cost"] = float(np.mean(successful)) if successful else None
        summary["dynamic_failure_state_cost"] = float(np.mean(failed)) if failed else None

    result = {"summary": summary, "records": records, "failures": failures}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output.as_posix())


if __name__ == "__main__":
    main()
