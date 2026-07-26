"""Differentiable structured-state guidance used by DecoArt inference.

This module implements the inference-time counterpart of Eqs. (11)--(13):

* a differentiable box/joint/range physical evaluator;
* deterministic structure/physics/detail routing;
* a token-space validity direction defined as ``-d J_i / d e_i``;
* group-weighted, bounded representation intervention.

The implementation intentionally has no dependency on Lightning or the model
package so that the evaluator and routing controls can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F


EPS = 1.0e-8
GROUP_NAMES = ("structure", "physical", "detail")


@dataclass
class StructuredGuidanceResult:
    hidden: torch.Tensor
    decoded: Dict[str, Any]
    log: Dict[str, Any]


def _cfg(config: Mapping[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def _unit(vector: torch.Tensor) -> torch.Tensor:
    fallback = torch.zeros_like(vector)
    fallback[..., 0] = 1.0
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    return torch.where(norm > EPS, vector / norm.clamp_min(EPS), fallback)


def _split_state(state: torch.Tensor) -> Dict[str, torch.Tensor]:
    if state.shape[-1] < 16:
        raise ValueError("A structured part state must contain at least 16 values")
    return {
        "size": state[..., 0:3].abs().clamp_min(1.0e-3),
        "center": state[..., 3:6],
        "origin": state[..., 6:9],
        "axis": _unit(state[..., 9:12]),
        "limit": state[..., 12:16],
    }


def _face_basis(state: torch.Tensor) -> Dict[str, torch.Tensor]:
    fields = _split_state(state)
    device, dtype = state.device, state.dtype
    normals = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )
    axis_u = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    axis_v = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    half = fields["size"] * 0.5
    centers = fields["center"].unsqueeze(-2) + normals * half.unsqueeze(-2)
    extent_u = torch.stack(
        (half[..., 1], half[..., 1], half[..., 0], half[..., 0], half[..., 0], half[..., 0]),
        dim=-1,
    )
    extent_v = torch.stack(
        (half[..., 2], half[..., 2], half[..., 2], half[..., 2], half[..., 1], half[..., 1]),
        dim=-1,
    )
    return {
        "centers": centers,
        "normals": normals,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "extent_u": extent_u,
        "extent_v": extent_v,
    }


def _straight_through_choice(
    scores: torch.Tensor,
    temperature: float,
    hard_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Use a hard face in the forward pass and a soft selector in backward."""
    soft = torch.softmax(scores / max(float(temperature), 1.0e-4), dim=-1)
    if hard_index is None:
        hard_index = scores.detach().argmax(dim=-1)
    hard = F.one_hot(hard_index, num_classes=scores.shape[-1]).to(scores.dtype)
    return hard + soft - soft.detach()


def _select(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...f,...fd->...d", weights, values)


def _select_scalar(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(weights * values, dim=-1)


def _face_samples(
    face_center: torch.Tensor,
    face_normal: torch.Tensor,
    axis_u: torch.Tensor,
    axis_v: torch.Tensor,
    extent_u: torch.Tensor,
    extent_v: torch.Tensor,
    samples_per_face: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    side = max(1, int(round(max(1, samples_per_face) ** 0.5)))
    while side * side < samples_per_face:
        side += 1
    if side == 1:
        coords = torch.zeros(1, device=face_center.device, dtype=face_center.dtype)
    else:
        coords = torch.linspace(-0.75, 0.75, side, device=face_center.device, dtype=face_center.dtype)
    uv = torch.cartesian_prod(coords, coords)[:samples_per_face]
    if uv.ndim == 1:
        uv = uv.unsqueeze(0)
    points = (
        face_center.unsqueeze(-2)
        + uv[:, 0:1] * extent_u[..., None, None] * axis_u.unsqueeze(-2)
        + uv[:, 1:2] * extent_v[..., None, None] * axis_v.unsqueeze(-2)
    )
    reliability = 0.25 + 0.75 * (1.0 - uv.abs().amax(dim=-1)).clamp_min(0.0)
    return points, reliability


def _add_state_noise(
    state: torch.Tensor,
    noise_std: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if noise_std <= 0.0:
        return state
    fields = _split_state(state)
    noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
    result = state.clone()
    result[..., 0:3] = state[..., 0:3] + noise_std * fields["size"] * noise[..., 0:3]
    result[..., 3:6] = state[..., 3:6] + noise_std * fields["size"] * noise[..., 3:6]
    result[..., 6:9] = state[..., 6:9] + noise_std * fields["size"] * noise[..., 6:9]
    result[..., 9:12] = _unit(state[..., 9:12] + noise_std * noise[..., 9:12])
    spans = torch.stack(
        (
            (fields["limit"][..., 1] - fields["limit"][..., 0]).abs(),
            (fields["limit"][..., 3] - fields["limit"][..., 2]).abs(),
        ),
        dim=-1,
    ).amax(dim=-1, keepdim=True).clamp_min(0.1)
    result[..., 12:16] = state[..., 12:16] + noise_std * spans * noise[..., 12:16]
    return result


def physical_terms_for_candidates(
    candidate_states: torch.Tensor,
    existing_states: torch.Tensor,
    candidate_parent_indices: torch.Tensor,
    config: Mapping[str, Any],
    *,
    seed: int = 2026,
) -> Dict[str, torch.Tensor]:
    """Compute differentiable physical terms for newly predicted parts.

    ``existing_states`` includes the start token at index zero. The root part,
    whose parent is the start token, is excluded from physical correction.
    Parent faces are always available as supports; nearby existing parts within
    ``neighbor_margin`` are included as additional support candidates.
    """
    if candidate_states.ndim != 2 or existing_states.ndim != 2:
        raise ValueError("candidate_states and existing_states must be rank-2 tensors")
    count = candidate_states.shape[0]
    if count == 0:
        empty = candidate_states.new_zeros((0,))
        return {key: empty for key in ("align", "normal", "penetration", "j_cost", "clearance")}

    generator = torch.Generator(device=candidate_states.device)
    generator.manual_seed(int(seed))
    state_noise_std = float(_cfg(config, "state_noise_std", 0.0))
    candidate_eval = _add_state_noise(candidate_states, state_noise_std, generator)
    existing_eval = _add_state_noise(existing_states.detach(), state_noise_std, generator)

    dmax = float(_cfg(config, "dmax", 0.10))
    alpha = float(_cfg(config, "alpha_normal", 0.50))
    beta = float(_cfg(config, "beta_penetration", 1.00))
    tau_pen = float(_cfg(config, "tau_pen", 0.02))
    base_clearance = float(_cfg(config, "base_clearance", 0.0))
    clearance_scale = float(_cfg(config, "limit_clearance_scale", 0.02))
    neighbor_margin = float(_cfg(config, "neighbor_margin", 0.03))
    samples_per_face = int(_cfg(config, "samples_per_face", 9))
    face_temperature = float(_cfg(config, "face_temperature", 0.10))
    support_corruption = float(_cfg(config, "support_corruption_ratio", 0.0))

    results: Dict[str, List[torch.Tensor]] = {
        "align": [],
        "normal": [],
        "penetration": [],
        "j_cost": [],
        "clearance": [],
    }

    existing_fields = _split_state(existing_eval)
    for idx in range(count):
        state = candidate_eval[idx]
        parent_idx = int(candidate_parent_indices[idx].item())
        root = parent_idx <= 0
        if root:
            zero = state.sum() * 0.0
            for key in results:
                results[key].append(zero)
            continue

        part = _split_state(state)
        axis = part["axis"]
        slide_span = (part["limit"][1] - part["limit"][0]).abs()
        rotate_span = (part["limit"][3] - part["limit"][2]).abs()
        radial = part["center"] - part["origin"]
        radial = radial - torch.dot(radial, axis) * axis
        radial = _unit(radial)
        is_prismatic = (slide_span >= rotate_span) & (slide_span > EPS)
        motion = torch.where(is_prismatic, axis, radial)

        child_faces = _face_basis(state)
        contact_scores = torch.mv(child_faces["normals"], motion)
        contact_weights = _straight_through_choice(contact_scores, face_temperature)
        contact_center = _select(child_faces["centers"], contact_weights)
        contact_normal = _unit(_select(child_faces["normals"], contact_weights))
        contact_u = _select(child_faces["axis_u"], contact_weights)
        contact_v = _select(child_faces["axis_v"], contact_weights)
        contact_extent_u = _select_scalar(child_faces["extent_u"], contact_weights)
        contact_extent_v = _select_scalar(child_faces["extent_v"], contact_weights)
        contact_points, _ = _face_samples(
            contact_center,
            contact_normal,
            contact_u,
            contact_v,
            contact_extent_u,
            contact_extent_v,
            samples_per_face,
        )

        support_indices = [parent_idx]
        candidate_center = part["center"].detach()
        candidate_half = part["size"].detach() * 0.5
        for support_idx in range(1, existing_eval.shape[0]):
            if support_idx == parent_idx:
                continue
            support_half = existing_fields["size"][support_idx] * 0.5
            gap = (
                (candidate_center - existing_fields["center"][support_idx]).abs()
                - (candidate_half + support_half)
            ).clamp_min(0.0)
            if float(torch.linalg.norm(gap).item()) <= neighbor_margin:
                support_indices.append(support_idx)

        support_state = existing_eval[support_indices]
        support_faces = _face_basis(support_state)
        support_centers = support_faces["centers"].reshape(-1, 3)
        support_normals = support_faces["normals"].repeat(len(support_indices), 1)
        support_u = support_faces["axis_u"].repeat(len(support_indices), 1)
        support_v = support_faces["axis_v"].repeat(len(support_indices), 1)
        support_extent_u = support_faces["extent_u"].reshape(-1)
        support_extent_v = support_faces["extent_v"].reshape(-1)

        support_scores = (
            torch.mv(support_normals, -contact_normal)
            + torch.exp(-torch.linalg.norm(support_centers - contact_center, dim=-1) / max(dmax, EPS))
        )
        corrupt = bool(torch.rand((), device=state.device, generator=generator).item() < support_corruption)
        hard_support_idx = None
        if corrupt and support_scores.numel() > 1:
            best = int(support_scores.detach().argmax().item())
            offset = int(torch.randint(1, support_scores.numel(), (), device=state.device, generator=generator).item())
            hard_support_idx = torch.tensor((best + offset) % support_scores.numel(), device=state.device)
        support_weights = _straight_through_choice(
            support_scores,
            face_temperature,
            hard_index=hard_support_idx,
        )
        support_center = _select(support_centers, support_weights)
        support_normal = _unit(_select(support_normals, support_weights))
        support_axis_u = _select(support_u, support_weights)
        support_axis_v = _select(support_v, support_weights)
        support_eu = _select_scalar(support_extent_u, support_weights)
        support_ev = _select_scalar(support_extent_v, support_weights)
        support_points, reliability = _face_samples(
            support_center,
            support_normal,
            support_axis_u,
            support_axis_v,
            support_eu,
            support_ev,
            samples_per_face,
        )

        q = contact_points[:, None, :]
        s = support_points[None, :, :]
        diff = q - s
        n = support_normal.view(1, 1, 3)
        m = contact_normal.view(1, 1, 3)
        normal_projection = torch.sum(diff * n, dim=-1, keepdim=True) * n
        tangent = torch.linalg.norm(diff - normal_projection, dim=-1)
        normal_error = 1.0 + torch.sum(n * m, dim=-1)

        box_radius = torch.linalg.norm(part["size"]) * 0.5
        motion_range = torch.maximum(slide_span, rotate_span * box_radius)
        clearance = base_clearance + clearance_scale * motion_range
        penetration_raw = -torch.sum(n * diff, dim=-1) - clearance
        penetration = penetration_raw.clamp_min(0.0)

        rho = reliability.view(1, -1)
        contact_reward = rho * torch.exp(
            -(tangent + alpha * normal_error) / max(dmax, EPS)
        )
        penetration_penalty = F.softplus(penetration_raw / max(tau_pen, EPS))
        j_cost = -contact_reward.mean() + beta * penetration_penalty.mean()

        results["align"].append(tangent.mean())
        results["normal"].append(normal_error.mean())
        results["penetration"].append(penetration.mean())
        results["j_cost"].append(j_cost)
        results["clearance"].append(clearance)

    return {key: torch.stack(value) for key, value in results.items()}


def tree_scores_for_candidates(
    existing_parent_indices: torch.Tensor,
    candidate_parent_indices: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    parents = [int(value) for value in existing_parent_indices.detach().cpu().tolist()]
    parents.extend(int(value) for value in candidate_parent_indices.detach().cpu().tolist())
    count = len(parents)
    depth = [-1] + [0] * max(0, count - 1)
    for idx in range(1, count):
        parent = parents[idx]
        depth[idx] = 0 if parent <= 0 else depth[parent] + 1
    subtree = [1] * count
    for idx in range(count - 1, 0, -1):
        parent = parents[idx]
        if parent >= 0 and parent != idx:
            subtree[parent] += subtree[idx]
    real_count = max(1, count - 1)
    max_depth = max(1, max(depth[1:], default=0))
    start = len(existing_parent_indices)
    lambda_depth = float(_cfg(config, "lambda_depth", 0.55))
    lambda_subtree = float(_cfg(config, "lambda_subtree", 0.45))
    scores = [
        lambda_depth * (1.0 - float(depth[idx]) / float(max_depth))
        + lambda_subtree * (float(subtree[idx]) / float(real_count))
        for idx in range(start, count)
    ]
    return candidate_parent_indices.new_tensor(scores, dtype=torch.float32)


def route_candidates(
    structure_scores: torch.Tensor,
    terms: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    seed: int = 2026,
) -> torch.Tensor:
    theta_tan = max(float(_cfg(config, "theta_tan", 0.10)), EPS)
    theta_pen = max(float(_cfg(config, "theta_pen", 0.02)), EPS)
    lambda_delta = float(_cfg(config, "lambda_delta", 0.55))
    lambda_omega = float(_cfg(config, "lambda_omega", 0.45))
    physical_scores = (
        lambda_delta * (terms["align"] / theta_tan).clamp(0.0, 1.0)
        + lambda_omega * (terms["penetration"] / theta_pen).clamp(0.0, 1.0)
    )
    tau_structure = float(_cfg(config, "tau_structure", 0.50))
    tau_physical = float(_cfg(config, "tau_physical", 0.35))
    routes = torch.full_like(structure_scores, 2, dtype=torch.long)
    structure_mask = (structure_scores >= tau_structure) & (structure_scores >= physical_scores)
    physical_mask = (physical_scores >= tau_physical) & (physical_scores > structure_scores)
    routes[structure_mask] = 0
    routes[physical_mask] = 1

    mode = str(_cfg(config, "routing_mode", "correct")).lower()
    generator = torch.Generator(device=routes.device)
    generator.manual_seed(int(seed))
    if mode == "random" and routes.numel() > 1:
        routes = routes[torch.randperm(routes.numel(), device=routes.device, generator=generator)]
    elif mode not in {"correct", "uniform"}:
        raise ValueError("routing_mode must be one of: correct, random, uniform")

    corruption = float(_cfg(config, "routing_corruption_ratio", 0.0))
    corrupt_count = min(routes.numel(), max(0, int(round(routes.numel() * corruption))))
    if corrupt_count > 0:
        selected = torch.randperm(routes.numel(), device=routes.device, generator=generator)[:corrupt_count]
        offset = torch.randint(1, 3, (corrupt_count,), device=routes.device, generator=generator)
        routes = routes.clone()
        routes[selected] = (routes[selected] + offset) % 3
    return routes


def _route_weights(routes: torch.Tensor, config: Mapping[str, Any], dtype: torch.dtype) -> torch.Tensor:
    if str(_cfg(config, "routing_mode", "correct")).lower() == "uniform":
        return torch.full(
            routes.shape,
            float(_cfg(config, "uniform_weight", 1.0)),
            device=routes.device,
            dtype=dtype,
        )
    weights_cfg = _cfg(
        config,
        "group_weights",
        {"structure": 0.50, "physical": 1.00, "detail": 0.25},
    )
    weights = torch.tensor(
        [
            float(weights_cfg.get("structure", 0.50)),
            float(weights_cfg.get("physical", 1.00)),
            float(weights_cfg.get("detail", 0.25)),
        ],
        device=routes.device,
        dtype=dtype,
    )
    return weights[routes]


def guide_part_representations(
    hidden: torch.Tensor,
    decode_hidden: Callable[[torch.Tensor], Dict[str, Any]],
    existing_states: torch.Tensor,
    existing_parent_indices: torch.Tensor,
    candidate_parent_indices: torch.Tensor,
    config: Mapping[str, Any],
    *,
    seed: int = 2026,
) -> StructuredGuidanceResult:
    """Apply Eq. (13) to newly predicted part representations.

    The code-grounded validity direction is

    ``v_i_phys = -grad_{e_i} J_i(D_state(e_i))``.

    A small backtracking loop implements the step-dependent ``eta_t`` and keeps
    accepted interventions non-increasing in the evaluated physical cost.
    """
    if hidden.shape[0] == 0:
        decoded = decode_hidden(hidden)
        return StructuredGuidanceResult(hidden, decoded, {"active_count": 0})

    steps = max(1, int(_cfg(config, "steps", 1)))
    base_eta = float(_cfg(config, "eta", 0.05))
    max_backtracks = max(0, int(_cfg(config, "max_backtracks", 5)))
    min_eta = float(_cfg(config, "min_eta", 1.0e-4))
    current = hidden.detach()
    route_counts = {name: 0 for name in GROUP_NAMES}
    first_cost: Optional[torch.Tensor] = None
    accepted_steps = 0
    last_eta = 0.0
    final_routes = torch.full((hidden.shape[0],), 2, device=hidden.device, dtype=torch.long)

    for iteration in range(steps):
        h = current.detach().requires_grad_(True)
        decoded = decode_hidden(h)
        states = decoded["articulated_info"]
        terms = physical_terms_for_candidates(
            states,
            existing_states,
            candidate_parent_indices,
            config,
            seed=seed + iteration,
        )
        active = candidate_parent_indices > 0
        if first_cost is None:
            first_cost = terms["j_cost"].detach()
        if not bool(active.any()):
            current = h.detach()
            break

        structure_scores = tree_scores_for_candidates(
            existing_parent_indices,
            candidate_parent_indices,
            config,
        ).to(device=h.device, dtype=h.dtype)
        routes = route_candidates(structure_scores, terms, config, seed=seed + iteration)
        final_routes = routes.detach()
        route_weight = _route_weights(routes, config, h.dtype)
        severity = torch.sigmoid(terms["j_cost"].detach())

        objective = terms["j_cost"][active].sum()
        grad = torch.autograd.grad(objective, h, retain_graph=False, create_graph=False)[0]
        direction = -grad
        direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(EPS)
        direction = direction * active.to(h.dtype).unsqueeze(-1)

        pre_mean = terms["j_cost"][active].detach().mean()
        eta = base_eta
        accepted = False
        for _ in range(max_backtracks + 1):
            scale = eta * route_weight * severity
            proposal = (h + scale.unsqueeze(-1) * direction).detach()
            with torch.no_grad():
                proposal_states = decode_hidden(proposal)["articulated_info"]
                proposal_terms = physical_terms_for_candidates(
                    proposal_states,
                    existing_states,
                    candidate_parent_indices,
                    config,
                    seed=seed + iteration,
                )
                post_mean = proposal_terms["j_cost"][active].mean()
            if bool(post_mean <= pre_mean + 1.0e-7):
                current = proposal
                accepted = True
                accepted_steps += 1
                last_eta = eta
                break
            eta *= 0.5
            if eta < min_eta:
                break
        if not accepted:
            current = h.detach()

    final_decoded = decode_hidden(current)
    with torch.no_grad():
        final_terms = physical_terms_for_candidates(
            final_decoded["articulated_info"],
            existing_states,
            candidate_parent_indices,
            config,
            seed=seed,
        )
    active = candidate_parent_indices > 0
    for group_idx, name in enumerate(GROUP_NAMES):
        route_counts[name] = int(((final_routes == group_idx) & active).sum().item())
    if bool(active.any()):
        pre_value = float(first_cost[active].mean().item()) if first_cost is not None else 0.0
        post_value = float(final_terms["j_cost"][active].mean().item())
        reduction = 100.0 * (pre_value - post_value) / max(abs(pre_value), EPS)
    else:
        pre_value = post_value = reduction = 0.0
    log = {
        "active_count": int(active.sum().item()),
        "j_pre": pre_value,
        "j_post": post_value,
        "reduction_percent": reduction,
        "accepted_steps": accepted_steps,
        "requested_steps": steps,
        "eta_final": last_eta,
        "routing_mode": str(_cfg(config, "routing_mode", "correct")),
        "route_counts": route_counts,
        "state_noise_std": float(_cfg(config, "state_noise_std", 0.0)),
        "routing_corruption_ratio": float(_cfg(config, "routing_corruption_ratio", 0.0)),
        "support_corruption_ratio": float(_cfg(config, "support_corruption_ratio", 0.0)),
    }
    return StructuredGuidanceResult(current, final_decoded, log)
