"""State-derived physical metrics and representation routing for DecoArt.

The functions in this file operate on ArtFormer/DecoArt metadata, not on
decoded meshes. They implement the experiment-side quantities described in the
paper:

- tangential alignment, normal consistency, and non-penetration;
- the part-wise physical validity cost J_i;
- representation routing groups for structure/physics/detail tokens.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


EPS = 1.0e-8


@dataclass
class DecoArtMetricConfig:
    bbox_format: str = "center_size"
    samples_per_face: int = 9
    dmax: float = 0.10
    alpha_normal: float = 0.50
    beta_penetration: float = 1.00
    tau_pen: float = 0.02
    base_clearance: float = 0.00
    limit_clearance_scale: float = 0.02
    neighbor_margin: float = 0.03
    include_nearby_support: bool = True
    theta_tan: float = 0.10
    theta_norm: float = 0.25
    theta_pen: float = 0.02
    lambda_depth: float = 0.55
    lambda_subtree: float = 0.45
    lambda_delta: float = 0.55
    lambda_omega: float = 0.45
    tau_structure: float = 0.50
    tau_physical: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _arr(value: Any, fallback: Optional[Iterable[float]] = None) -> np.ndarray:
    if value is None:
        if fallback is None:
            raise ValueError("Missing required numeric value")
        value = fallback
    return np.asarray(value, dtype=np.float64)


def _unit(value: Any, fallback: Optional[Iterable[float]] = None) -> np.ndarray:
    vector = _arr(value, fallback)
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        if fallback is not None:
            vector = _arr(fallback)
            norm = float(np.linalg.norm(vector))
        if norm < EPS:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vector / norm


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _bbox_from_part(part: Dict[str, Any], bbox_format: str) -> Tuple[np.ndarray, np.ndarray]:
    if "bbox_center" in part and ("bbox_l" in part or "bbox_size" in part):
        center = _arr(part["bbox_center"])
        size = _arr(part.get("bbox_l", part.get("bbox_size")))
        return center, np.maximum(np.abs(size), EPS)

    if "token" in part and len(part["token"]) >= 6:
        token = _arr(part["token"])
        return token[3:6], np.maximum(np.abs(token[0:3]), EPS)

    if "bbx" not in part:
        raise ValueError("Part must contain either bbx, bbox_center/bbox_l, or token")

    bbx = part["bbx"]
    first = _arr(bbx[0])
    second = _arr(bbx[1])
    if bbox_format == "min_max":
        center = (first + second) * 0.5
        size = second - first
    elif bbox_format == "center_size":
        center = first
        size = second
    else:
        raise ValueError("bbox_format must be 'center_size' or 'min_max'")
    return center, np.maximum(np.abs(size), EPS)


def _part_from_raw(part: Dict[str, Any], index: int, cfg: DecoArtMetricConfig) -> Dict[str, Any]:
    center, size = _bbox_from_part(part, cfg.bbox_format)
    limit = _arr(part.get("limit", [0.0, 0.0, 0.0, 0.0]))
    axis = part.get("joint_data_direction", part.get("joint_axis", None))
    origin = part.get("joint_data_origin", part.get("joint_origin", center))

    return {
        "index": index,
        "name": part.get("name", str(index)),
        "dfn": int(part.get("dfn", index + 1)),
        "dfn_fa": int(part.get("dfn_fa", part.get("fa", part.get("parent", 0)))),
        "center": center,
        "size": size,
        "joint_origin": _arr(origin),
        "joint_axis": _unit(axis, fallback=center),
        "limit": limit,
    }


def _load_json_parts(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data, {}
    if "shape_info" in data:
        return data["shape_info"], data.get("meta", {})
    if "part" in data:
        return data["part"], data.get("meta", {})
    if "data" in data and isinstance(data["data"], list):
        return data["data"], data.get("meta", {})
    raise ValueError(f"Cannot find parts in {path.as_posix()}")


def load_parts(path: Path, cfg: DecoArtMetricConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load a transformer JSON, preprocessed metadata JSON, or generated output.dat."""
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle", ".dat"}:
        with path.open("rb") as fp:
            data = pickle.load(fp)
        if isinstance(data, dict) and "data" in data:
            raw_parts = data["data"]
            meta = data.get("meta", {})
        else:
            raw_parts = data
            meta = {}
    else:
        raw_parts, meta = _load_json_parts(path)
    parts = [_part_from_raw(part, idx, cfg) for idx, part in enumerate(raw_parts)]
    return parts, meta


def _tree_info(parts: List[Dict[str, Any]]) -> Tuple[List[Optional[int]], List[int], List[int]]:
    id_to_idx = {part["dfn"]: idx for idx, part in enumerate(parts)}
    parents: List[Optional[int]] = []
    children = [[] for _ in parts]
    for idx, part in enumerate(parts):
        parent_idx = id_to_idx.get(part["dfn_fa"])
        if parent_idx == idx:
            parent_idx = None
        parents.append(parent_idx)
        if parent_idx is not None:
            children[parent_idx].append(idx)

    depth = [0 for _ in parts]

    def assign_depth(idx: int, current_depth: int) -> None:
        depth[idx] = current_depth
        for child in children[idx]:
            assign_depth(child, current_depth + 1)

    roots = [idx for idx, parent in enumerate(parents) if parent is None]
    for root in roots:
        assign_depth(root, 0)

    subtree = [1 for _ in parts]

    def count_subtree(idx: int) -> int:
        total = 1
        for child in children[idx]:
            total += count_subtree(child)
        subtree[idx] = total
        return total

    for root in roots:
        count_subtree(root)

    return parents, depth, subtree


def _aabb_gap(part_a: Dict[str, Any], part_b: Dict[str, Any]) -> float:
    half_a = part_a["size"] * 0.5
    half_b = part_b["size"] * 0.5
    gap = np.abs(part_a["center"] - part_b["center"]) - (half_a + half_b)
    return float(np.linalg.norm(np.maximum(gap, 0.0)))


def _box_faces(center: np.ndarray, size: np.ndarray) -> List[Dict[str, Any]]:
    half = size * 0.5
    axes = np.eye(3, dtype=np.float64)
    faces = []
    specs = [
        (axes[0], axes[1], axes[2], half[0], half[1], half[2]),
        (-axes[0], axes[1], axes[2], half[0], half[1], half[2]),
        (axes[1], axes[0], axes[2], half[1], half[0], half[2]),
        (-axes[1], axes[0], axes[2], half[1], half[0], half[2]),
        (axes[2], axes[0], axes[1], half[2], half[0], half[1]),
        (-axes[2], axes[0], axes[1], half[2], half[0], half[1]),
    ]
    for normal, axis_u, axis_v, offset, extent_u, extent_v in specs:
        faces.append({
            "center": center + normal * offset,
            "normal": normal,
            "axis_u": axis_u,
            "axis_v": axis_v,
            "extent_u": float(extent_u),
            "extent_v": float(extent_v),
            "area": float(max(2.0 * extent_u, EPS) * max(2.0 * extent_v, EPS)),
        })
    return faces


def _sample_face(face: Dict[str, Any], samples_per_face: int) -> Tuple[np.ndarray, np.ndarray]:
    side = max(1, int(math.ceil(math.sqrt(samples_per_face))))
    if side == 1:
        uv = [(0.0, 0.0)]
    else:
        coords = np.linspace(-0.75, 0.75, side)
        uv = [(float(u), float(v)) for u in coords for v in coords]
    uv = uv[:samples_per_face]

    points = []
    reliability = []
    for u, v in uv:
        point = (
            face["center"]
            + u * face["extent_u"] * face["axis_u"]
            + v * face["extent_v"] * face["axis_v"]
        )
        edge_margin = 1.0 - max(abs(u), abs(v))
        points.append(point)
        reliability.append(0.25 + 0.75 * max(0.0, edge_margin))
    return np.asarray(points, dtype=np.float64), np.asarray(reliability, dtype=np.float64)


def _motion_direction(part: Dict[str, Any], parent: Optional[Dict[str, Any]]) -> np.ndarray:
    axis = _unit(part["joint_axis"])
    slide_span = abs(float(part["limit"][1] - part["limit"][0]))
    rotate_span = abs(float(part["limit"][3] - part["limit"][2]))
    if slide_span >= rotate_span and slide_span > EPS:
        return axis

    radial = part["center"] - part["joint_origin"]
    radial = radial - float(np.dot(radial, axis)) * axis
    if np.linalg.norm(radial) > EPS:
        return _unit(radial)

    if parent is not None:
        fallback = part["center"] - parent["center"]
        if np.linalg.norm(fallback) > EPS:
            return _unit(fallback)
    return axis


def _clearance_margin(part: Dict[str, Any], cfg: DecoArtMetricConfig) -> float:
    slide_span = abs(float(part["limit"][1] - part["limit"][0]))
    rotate_span = abs(float(part["limit"][3] - part["limit"][2]))
    box_radius = float(np.linalg.norm(part["size"]) * 0.5)
    motion_range = max(slide_span, rotate_span * box_radius)
    return float(cfg.base_clearance + cfg.limit_clearance_scale * motion_range)


def _support_candidates(
    parts: List[Dict[str, Any]],
    part_idx: int,
    parent_idx: Optional[int],
    cfg: DecoArtMetricConfig,
) -> List[int]:
    candidates = []
    if parent_idx is not None:
        candidates.append(parent_idx)
    if not cfg.include_nearby_support:
        return candidates

    for idx, candidate in enumerate(parts):
        if idx == part_idx or idx == parent_idx:
            continue
        if _aabb_gap(parts[part_idx], candidate) <= cfg.neighbor_margin:
            candidates.append(idx)
    return candidates


def _physical_terms_for_part(
    parts: List[Dict[str, Any]],
    part_idx: int,
    parent_idx: Optional[int],
    cfg: DecoArtMetricConfig,
) -> Dict[str, Any]:
    part = parts[part_idx]
    if parent_idx is None:
        return {
            "part_index": part_idx,
            "dfn": part["dfn"],
            "parent_dfn": part["dfn_fa"],
            "is_root": True,
            "align": 0.0,
            "normal": 0.0,
            "penetration": 0.0,
            "penetration_positive": 0.0,
            "j_cost": 0.0,
            "support_part_dfn": None,
        }

    parent = parts[parent_idx]
    motion_direction = _motion_direction(part, parent)
    contact_faces = _box_faces(part["center"], part["size"])
    contact_face = max(contact_faces, key=lambda f: float(np.dot(f["normal"], motion_direction)))
    contact_points, _ = _sample_face(contact_face, cfg.samples_per_face)
    contact_normals = np.repeat(contact_face["normal"][None, :], contact_points.shape[0], axis=0)

    support_target = -contact_face["normal"]
    best_support = None
    for candidate_idx in _support_candidates(parts, part_idx, parent_idx, cfg):
        for face in _box_faces(parts[candidate_idx]["center"], parts[candidate_idx]["size"]):
            dist = float(np.linalg.norm(face["center"] - contact_face["center"]))
            score = float(np.dot(face["normal"], support_target)) + math.exp(-dist / max(cfg.dmax, EPS))
            if best_support is None or score > best_support[0]:
                best_support = (score, candidate_idx, face)

    if best_support is None:
        return {
            "part_index": part_idx,
            "dfn": part["dfn"],
            "parent_dfn": part["dfn_fa"],
            "is_root": False,
            "align": 0.0,
            "normal": 0.0,
            "penetration": 0.0,
            "penetration_positive": 0.0,
            "j_cost": 0.0,
            "support_part_dfn": None,
        }

    _, support_idx, support_face = best_support
    support_points, reliability = _sample_face(support_face, cfg.samples_per_face)
    support_normals = np.repeat(support_face["normal"][None, :], support_points.shape[0], axis=0)

    q = contact_points[:, None, :]
    m = contact_normals[:, None, :]
    s = support_points[None, :, :]
    n = support_normals[None, :, :]
    rho = reliability[None, :]

    diff = q - s
    normal_projection = np.sum(diff * n, axis=-1, keepdims=True) * n
    tangent = np.linalg.norm(diff - normal_projection, axis=-1)
    normal_error = 1.0 + np.sum(n * m, axis=-1)

    delta = _clearance_margin(part, cfg)
    penetration = -np.sum(n * diff, axis=-1) - delta
    penetration_positive = np.maximum(penetration, 0.0)

    contact_reward = rho * np.exp(-(tangent + cfg.alpha_normal * normal_error) / max(cfg.dmax, EPS))
    penetration_penalty = _softplus(penetration / max(cfg.tau_pen, EPS))
    j_cost = -float(np.mean(contact_reward)) + cfg.beta_penetration * float(np.mean(penetration_penalty))

    return {
        "part_index": part_idx,
        "dfn": part["dfn"],
        "parent_dfn": part["dfn_fa"],
        "is_root": False,
        "align": float(np.mean(tangent)),
        "normal": float(np.mean(normal_error)),
        "penetration": float(np.mean(penetration)),
        "penetration_positive": float(np.mean(penetration_positive)),
        "j_cost": j_cost,
        "clearance": delta,
        "support_part_dfn": parts[support_idx]["dfn"],
    }


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < EPS:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def _route_representations(
    parts: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    depth: List[int],
    subtree: List[int],
    cfg: DecoArtMetricConfig,
) -> List[Dict[str, Any]]:
    n_parts = max(1, len(parts))
    max_depth = max(max(depth), 1) if depth else 1

    align = _normalize_scores(np.asarray([m["align"] for m in metrics], dtype=np.float64))
    pen = _normalize_scores(np.asarray([m["penetration_positive"] for m in metrics], dtype=np.float64))

    routes = []
    for idx, part in enumerate(parts):
        structure_score = (
            cfg.lambda_depth * (1.0 - float(depth[idx]) / float(max_depth))
            + cfg.lambda_subtree * (float(subtree[idx]) / float(n_parts))
        )
        physical_score = cfg.lambda_delta * float(align[idx]) + cfg.lambda_omega * float(pen[idx])

        if structure_score >= cfg.tau_structure and structure_score >= physical_score:
            group = "structure"
        elif physical_score >= cfg.tau_physical and physical_score > structure_score:
            group = "physical"
        else:
            group = "detail"

        routes.append({
            "part_index": idx,
            "dfn": part["dfn"],
            "depth": int(depth[idx]),
            "subtree_size": int(subtree[idx]),
            "structure_score": float(structure_score),
            "physical_score": float(physical_score),
            "group": group,
        })
    return routes


def summarize_object(metrics: List[Dict[str, Any]], depth: List[int], cfg: DecoArtMetricConfig) -> Dict[str, Any]:
    valid_metrics = [m for m in metrics if not m.get("is_root", False)]
    if not valid_metrics:
        valid_metrics = metrics

    align = float(np.mean([m["align"] for m in valid_metrics])) if valid_metrics else 0.0
    normal = float(np.mean([m["normal"] for m in valid_metrics])) if valid_metrics else 0.0
    penetration = float(np.mean([m["penetration_positive"] for m in valid_metrics])) if valid_metrics else 0.0
    j_cost = float(np.mean([m["j_cost"] for m in valid_metrics])) if valid_metrics else 0.0
    valid = bool(align <= cfg.theta_tan and normal <= cfg.theta_norm and penetration <= cfg.theta_pen)

    return {
        "part_count": len(metrics),
        "max_depth": int(max(depth) if depth else 0),
        "align": align,
        "normal": normal,
        "penetration": penetration,
        "j_cost": j_cost,
        "pv_valid": valid,
    }


def analyze_parts(parts: List[Dict[str, Any]], cfg: DecoArtMetricConfig) -> Dict[str, Any]:
    parents, depth, subtree = _tree_info(parts)
    metrics = [
        _physical_terms_for_part(parts, idx, parents[idx], cfg)
        for idx in range(len(parts))
    ]
    routes = _route_representations(parts, metrics, depth, subtree, cfg)
    summary = summarize_object(metrics, depth, cfg)
    return {
        "summary": summary,
        "parts": metrics,
        "routing": routes,
        "config": cfg.to_dict(),
    }


def analyze_path(path: Path, cfg: DecoArtMetricConfig) -> Dict[str, Any]:
    parts, meta = load_parts(path, cfg)
    result = analyze_parts(parts, cfg)
    result["source"] = Path(path).as_posix()
    result["meta"] = meta
    return result
