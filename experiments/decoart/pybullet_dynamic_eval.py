"""PyBullet closed--open--closed validation for generated DecoArt objects."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh


def _load_parts(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as fp:
        data = pickle.load(fp)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("output.dat must contain a list of generated parts")
    return sorted(data, key=lambda item: int(item["dfn"]))


def _as_mesh(mesh: Any) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        return trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh.copy()


def _fit_mesh(part: Dict[str, Any], frame_origin: np.ndarray) -> trimesh.Trimesh:
    mesh = _as_mesh(part["mesh"])
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    source_min = vertices.min(axis=0)
    source_max = vertices.max(axis=0)
    source_size = np.maximum(source_max - source_min, 1.0e-6)
    center = np.asarray(part["bbx"][0], dtype=np.float64)
    size = np.maximum(np.abs(np.asarray(part["bbx"][1], dtype=np.float64)), 1.0e-4)
    target_min = center - size * 0.5
    mesh.vertices = target_min + size * ((vertices - source_min) / source_size) - frame_origin
    return mesh


def _joint_spec(part: Dict[str, Any]) -> Tuple[str, float, float]:
    limit = np.asarray(part.get("limit", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    axis = np.asarray(part.get("joint_data_direction", [0.0, 0.0, 0.0]), dtype=np.float64)
    slide_span = abs(float(limit[1] - limit[0]))
    rotate_span = abs(float(limit[3] - limit[2]))
    if np.linalg.norm(axis) < 1.0e-6 or max(slide_span, rotate_span) < 1.0e-6:
        return "fixed", 0.0, 0.0
    if slide_span >= rotate_span:
        return "prismatic", float(min(limit[0], limit[1])), float(max(limit[0], limit[1]))
    return "revolute", float(min(limit[2], limit[3])), float(max(limit[2], limit[3]))


def _xyz(vector: np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in vector)


def _build_urdf(parts: List[Dict[str, Any]], directory: Path) -> Tuple[Path, Dict[int, Dict[str, Any]]]:
    frame_origins: Dict[int, np.ndarray] = {}
    for part in parts:
        dfn = int(part["dfn"])
        if int(part["dfn_fa"]) == 0:
            frame_origins[dfn] = np.zeros(3, dtype=np.float64)
        else:
            frame_origins[dfn] = np.asarray(part["joint_data_origin"], dtype=np.float64)

    robot = ET.Element("robot", name="decoart_generated")
    joint_meta: Dict[int, Dict[str, Any]] = {}
    for part in parts:
        dfn = int(part["dfn"])
        link = ET.SubElement(robot, "link", name=f"part_{dfn}")
        mesh = _fit_mesh(part, frame_origins[dfn])
        mesh_path = directory / f"part_{dfn}.obj"
        mesh.export(mesh_path)

        center = np.asarray(part["bbx"][0], dtype=np.float64) - frame_origins[dfn]
        size = np.maximum(np.abs(np.asarray(part["bbx"][1], dtype=np.float64)), 1.0e-3)
        mass = max(0.01, float(np.prod(size)))
        inertia = mass / 12.0 * np.array(
            [size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2]
        )
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=_xyz(center), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{inertia[0]:.9g}",
            ixy="0",
            ixz="0",
            iyy=f"{inertia[1]:.9g}",
            iyz="0",
            izz=f"{inertia[2]:.9g}",
        )
        for tag in ("visual", "collision"):
            element = ET.SubElement(link, tag)
            ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "mesh", filename=mesh_path.name)

    for part in parts:
        child_dfn = int(part["dfn"])
        parent_dfn = int(part["dfn_fa"])
        if parent_dfn == 0:
            continue
        joint_type, lower, upper = _joint_spec(part)
        joint = ET.SubElement(robot, "joint", name=f"joint_{child_dfn}", type=joint_type)
        ET.SubElement(joint, "parent", link=f"part_{parent_dfn}")
        ET.SubElement(joint, "child", link=f"part_{child_dfn}")
        relative_origin = frame_origins[child_dfn] - frame_origins[parent_dfn]
        ET.SubElement(joint, "origin", xyz=_xyz(relative_origin), rpy="0 0 0")
        axis = np.asarray(part.get("joint_data_direction", [1.0, 0.0, 0.0]), dtype=np.float64)
        axis_norm = np.linalg.norm(axis)
        axis = axis / axis_norm if axis_norm > 1.0e-8 else np.array([1.0, 0.0, 0.0])
        ET.SubElement(joint, "axis", xyz=_xyz(axis))
        if joint_type != "fixed":
            ET.SubElement(
                joint,
                "limit",
                lower=f"{lower:.9g}",
                upper=f"{upper:.9g}",
                effort="100",
                velocity="2",
            )
            joint_meta[child_dfn] = {
                "type": joint_type,
                "lower": lower,
                "upper": upper,
            }

    roots = [int(part["dfn"]) for part in parts if int(part["dfn_fa"]) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}")
    urdf_path = directory / "object.urdf"
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path, joint_meta


def evaluate_object(
    path: Path,
    *,
    steps_per_leg: int = 240,
    settle_steps: int = 120,
    floor_min_height: float = -0.015,
    revolute_tolerance_deg: float = 5.0,
    prismatic_tolerance: float = 0.03,
    gui: bool = False,
) -> Dict[str, Any]:
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:
        raise RuntimeError(
            "PyBullet is required. Install the project environment or `pip install pybullet==3.2.6`."
        ) from exc

    parts = _load_parts(path)
    with tempfile.TemporaryDirectory(prefix="decoart-pybullet-") as temp:
        directory = Path(temp)
        urdf_path, joint_meta_by_dfn = _build_urdf(parts, directory)
        client = p.connect(p.GUI if gui else p.DIRECT)
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
            p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
            p.setTimeStep(1.0 / 240.0, physicsClientId=client)
            p.loadURDF("plane.urdf", physicsClientId=client)
            flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            body = p.loadURDF(
                urdf_path.as_posix(),
                useFixedBase=False,
                flags=flags,
                physicsClientId=client,
            )

            joint_indices = []
            joint_meta = []
            for joint_idx in range(p.getNumJoints(body, physicsClientId=client)):
                info = p.getJointInfo(body, joint_idx, physicsClientId=client)
                joint_name = info[1].decode("utf-8")
                if not joint_name.startswith("joint_"):
                    continue
                child_dfn = int(joint_name.split("_")[-1])
                meta = joint_meta_by_dfn.get(child_dfn)
                if meta is not None:
                    joint_indices.append(joint_idx)
                    joint_meta.append(meta)
                    p.resetJointState(body, joint_idx, meta["lower"], physicsClientId=client)

            failure_reasons = set()
            max_tracking_error = 0.0
            min_height = float("inf")
            self_contact_count = 0

            def validate(targets: List[float], check_tracking: bool) -> None:
                nonlocal max_tracking_error, min_height, self_contact_count
                for joint_idx, target in zip(joint_indices, targets):
                    p.setJointMotorControl2(
                        body,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target,
                        force=100.0,
                        positionGain=0.3,
                        velocityGain=1.0,
                        physicsClientId=client,
                    )
                p.stepSimulation(physicsClientId=client)
                aabbs = [
                    p.getAABB(body, link_idx, physicsClientId=client)
                    for link_idx in range(-1, p.getNumJoints(body, physicsClientId=client))
                ]
                min_height = min(min_height, min(float(aabb[0][2]) for aabb in aabbs))
                if min_height < floor_min_height:
                    failure_reasons.add("floor_instability")
                contacts = p.getContactPoints(bodyA=body, bodyB=body, physicsClientId=client)
                invalid_contacts = [
                    contact for contact in contacts
                    if contact[3] != contact[4] and float(contact[8]) < -1.0e-4
                ]
                if invalid_contacts:
                    self_contact_count += len(invalid_contacts)
                    failure_reasons.add("self_contact")
                if check_tracking:
                    for joint_idx, meta, target in zip(joint_indices, joint_meta, targets):
                        actual = float(p.getJointState(body, joint_idx, physicsClientId=client)[0])
                        error = abs(actual - target)
                        max_tracking_error = max(max_tracking_error, error)
                        tolerance = (
                            math.radians(revolute_tolerance_deg)
                            if meta["type"] == "revolute"
                            else prismatic_tolerance
                        )
                        if error > tolerance:
                            failure_reasons.add("tracking")

            lower = [meta["lower"] for meta in joint_meta]
            upper = [meta["upper"] for meta in joint_meta]
            for _ in range(settle_steps):
                validate(lower, check_tracking=False)
            for start, end in ((lower, upper), (upper, lower)):
                for step in range(steps_per_leg):
                    ratio = 0.5 - 0.5 * math.cos(math.pi * (step + 1) / steps_per_leg)
                    targets = [
                        float(left + ratio * (right - left))
                        for left, right in zip(start, end)
                    ]
                    validate(targets, check_tracking=True)
            for _ in range(settle_steps):
                validate(lower, check_tracking=True)

            return {
                "source": path.as_posix(),
                "success": not failure_reasons,
                "failure_reasons": sorted(failure_reasons),
                "joint_count": len(joint_indices),
                "max_tracking_error": max_tracking_error,
                "min_aabb_height": min_height,
                "self_contact_count": self_contact_count,
            }
        finally:
            p.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--pattern", default="*/*/output.dat")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps-per-leg", type=int, default=240)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    records = []
    failures = []
    for path in sorted(Path(args.input).glob(args.pattern)):
        try:
            records.append(
                evaluate_object(
                    path,
                    steps_per_leg=args.steps_per_leg,
                    settle_steps=args.settle_steps,
                    gui=args.gui,
                )
            )
        except Exception as exc:
            failures.append({"source": path.as_posix(), "error": str(exc)})
    dyn_sr = (
        100.0 * sum(bool(item["success"]) for item in records) / len(records)
        if records else 0.0
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dyn_sr": dyn_sr,
                "count": len(records),
                "records": records,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output.as_posix())


if __name__ == "__main__":
    main()
