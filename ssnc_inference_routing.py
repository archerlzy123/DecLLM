"""Inference-time routing for unseen SSNC scale-mask rows.

Training stores one binary mask row for every exact ``(n_jobs, n_machines)``
group. Rows never observed during training remain all-one. This module can
replace only the requested unseen row with a deterministic vote from nearby
seen SSNC domains. Existing seen rows and the default all-one behavior are
left unchanged.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch

from ssnc_mask import static_structure_vector


MASK_TABLE_NAMES: Tuple[str, ...] = (
    "scale_param_mask_gnn_encoder",
    "scale_param_mask_state_projection",
    "scale_param_mask_lora",
    "scale_param_mask_candidate_head",
    "scale_param_mask_scale_adapter",
)


def _unpack_structures(ssnc_state: Mapping[str, object]) -> Dict[int, torch.Tensor]:
    packed = ssnc_state.get("structure_ema", {}) if isinstance(ssnc_state, Mapping) else {}
    result: Dict[int, torch.Tensor] = {}
    for key, value in (packed or {}).items():
        if torch.is_tensor(value) and value.numel() >= 5:
            result[int(key)] = value.detach().cpu().float().reshape(-1)[:5]
    return result


def _gaussian_topk(
    matrix: torch.Tensor,
    target: torch.Tensor,
    neighbor_k: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected row indices, normalized weights, and all distances."""
    std = matrix.std(dim=0, unbiased=False)
    standardized = (matrix - matrix.mean(dim=0)) / torch.where(
        std > eps, std, torch.ones_like(std)
    )
    target_standardized = (target - matrix.mean(dim=0)) / torch.where(
        std > eps, std, torch.ones_like(std)
    )
    distances = torch.linalg.vector_norm(
        standardized - target_standardized.unsqueeze(0), dim=1
    )
    k = min(max(1, int(neighbor_k)), int(matrix.shape[0]))
    chosen = torch.topk(distances, k=k, largest=False).indices
    finite = distances[chosen]
    bandwidth = max(float(finite[-1].item()), float(eps))
    raw = torch.exp(-(finite.pow(2)) / (2.0 * bandwidth * bandwidth))
    weights = raw / raw.sum().clamp_min(eps)
    return chosen, weights, distances


def infer_target_structure_and_neighbors(
    ssnc_state: Mapping[str, object],
    target_n_jobs: int,
    target_n_machines: int,
    max_jobs: int,
    max_machines: int,
    neighbor_k: int = 3,
    eps: float = 1e-8,
):
    """Infer a scale-level target descriptor and its seen SSNC neighbors.

    The first three descriptor components are known exactly from problem size.
    The last two training-time dynamic averages (legal ratio and frontier
    conflict) are estimated from the nearest seen domains in static size space.
    Final neighbors are then selected in the same standardized five-dimensional
    structure space used by SSNC training.
    """
    structures = _unpack_structures(ssnc_state)
    if not structures:
        raise ValueError("Checkpoint does not contain SSNC structure_ema domains")

    groups = sorted(structures)
    matrix = torch.stack([structures[group] for group in groups])
    target_group = int(target_n_jobs) * (int(max_machines) + 1) + int(target_n_machines)
    target = static_structure_vector(target_group, int(max_jobs), int(max_machines))

    # Estimate scale-level dynamic coordinates using only known static geometry.
    static_chosen, static_weights, _ = _gaussian_topk(
        matrix[:, :3], target[:3], neighbor_k=neighbor_k, eps=eps
    )
    target[3:] = torch.sum(
        matrix[static_chosen, 3:] * static_weights.unsqueeze(1), dim=0
    )

    chosen, weights, distances = _gaussian_topk(
        matrix, target, neighbor_k=neighbor_k, eps=eps
    )
    neighbors = [
        {
            "group_id": int(groups[int(index)]),
            "n_jobs": int(groups[int(index)]) // (int(max_machines) + 1),
            "n_machines": int(groups[int(index)]) % (int(max_machines) + 1),
            "weight": float(weight),
            "distance": float(distances[int(index)]),
        }
        for index, weight in zip(chosen, weights)
    ]
    return target, neighbors


def weighted_vote_mask(
    rows: torch.Tensor,
    weights: torch.Tensor,
    rule: str = "density_preserving",
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate binary neighbor rows and return ``(binary_mask, vote_scores)``."""
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError(f"Expected non-empty [neighbors, channels] rows, got {rows.shape}")
    weights = weights.detach().cpu().float().reshape(-1)
    if weights.numel() != rows.shape[0]:
        raise ValueError("Neighbor weight count does not match mask rows")
    weights = weights / weights.sum().clamp_min(1e-8)
    binary_rows = (rows.detach().cpu().float() > 0.5).float()
    scores = torch.sum(binary_rows * weights.unsqueeze(1), dim=0)

    normalized_rule = str(rule or "density_preserving").strip().lower()
    if normalized_rule == "threshold":
        mask = (scores >= float(threshold)).float()
    elif normalized_rule == "density_preserving":
        expected_open = torch.sum(binary_rows.sum(dim=1) * weights)
        open_count = int(round(float(expected_open.item())))
        open_count = max(1, min(open_count, int(scores.numel())))
        # Stable sorting makes ties deterministic by preserving channel order.
        order = torch.argsort(scores, descending=True, stable=True)
        mask = torch.zeros_like(scores)
        mask[order[:open_count]] = 1.0
    else:
        raise ValueError(f"Unknown vote rule: {rule}")
    return mask, scores


def apply_unseen_scale_neighbor_routing(
    model,
    ssnc_state: Mapping[str, object],
    target_n_jobs: int,
    target_n_machines: int,
    neighbor_k: int = 3,
    mode: str = "neighbor_vote",
    vote_rule: str = "density_preserving",
    vote_threshold: float = 0.5,
):
    """Route the requested unseen exact-scale row on a loaded DT model."""
    max_jobs = int(model.n_jobs)
    max_machines = int(model.n_machines)
    target_group = int(target_n_jobs) * (max_machines + 1) + int(target_n_machines)
    structures = _unpack_structures(ssnc_state)
    report = {
        "mode": str(mode),
        "target": {
            "group_id": target_group,
            "n_jobs": int(target_n_jobs),
            "n_machines": int(target_n_machines),
        },
        "neighbor_k": int(neighbor_k),
        "vote_rule": str(vote_rule),
        "vote_threshold": float(vote_threshold),
        "seen_groups": sorted(int(group) for group in structures),
        "masks": {},
    }
    if target_group in structures:
        report["status"] = "seen_scale_exact_row_unchanged"
        return report

    target_structure, neighbors = infer_target_structure_and_neighbors(
        ssnc_state=ssnc_state,
        target_n_jobs=target_n_jobs,
        target_n_machines=target_n_machines,
        max_jobs=max_jobs,
        max_machines=max_machines,
        neighbor_k=neighbor_k,
    )
    report["target_structure"] = [float(value) for value in target_structure]
    report["neighbors"] = neighbors
    neighbor_groups = [int(item["group_id"]) for item in neighbors]
    neighbor_weights = torch.tensor(
        [float(item["weight"]) for item in neighbors], dtype=torch.float32
    )

    normalized_mode = str(mode or "neighbor_vote").strip().lower()
    if normalized_mode not in {"nearest", "neighbor_vote"}:
        raise ValueError(f"Unsupported unseen routing mode: {mode}")

    changed = 0
    for table_name in MASK_TABLE_NAMES:
        table = getattr(model, table_name, None)
        if not torch.is_tensor(table) or table.ndim != 2:
            report["masks"][table_name] = {"status": "not_available"}
            continue
        if target_group < 0 or target_group >= int(table.shape[0]):
            raise IndexError(
                f"Target group {target_group} outside {table_name} rows={table.shape[0]}"
            )
        source_rows = table.detach().cpu()[neighbor_groups]
        before = table.detach().cpu()[target_group].clone()
        if normalized_mode == "nearest":
            routed = (source_rows[0] > 0.5).float()
            scores = routed.clone()
        else:
            routed, scores = weighted_vote_mask(
                source_rows,
                neighbor_weights,
                rule=vote_rule,
                threshold=vote_threshold,
            )
        with torch.no_grad():
            table[target_group].copy_(
                routed.to(device=table.device, dtype=table.dtype)
            )
        changed += 1
        report["masks"][table_name] = {
            "status": "routed",
            "channels": int(table.shape[1]),
            "before_density": float((before > 0.5).float().mean()),
            "after_density": float(routed.mean()),
            "vote_score_mean": float(scores.mean()),
            "vote_score_min": float(scores.min()),
            "vote_score_max": float(scores.max()),
            "neighbor_densities": [
                float((row > 0.5).float().mean()) for row in source_rows
            ],
        }
    report["status"] = "routed"
    report["routed_mask_tables"] = changed
    report["notes"] = [
        "Only the requested unseen exact-scale mask row is modified.",
        "The LoRA table is routed for consistency; current code uses it for gradient gating, not forward-time rank masking.",
        "Scale embeddings and scale-adapter alpha rows are intentionally unchanged in this mask-only ablation.",
    ]
    return report
