"""Scale/structure-neighborhood conflict-aware mask utilities.

This module is intentionally independent from the model implementation.  It
changes only how a scale-specific binary mask ranks its channels; the five
mask locations and the model forward pass remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

import torch


TensorMap = Mapping[int, torch.Tensor]


def decode_scale_group(group: int, max_jobs: int, max_machines: int) -> Tuple[int, int]:
    """Decode the exact-scale group id used by DecisionTransformer."""
    group = int(group)
    stride = int(max_machines) + 1
    return group // stride, group % stride


def static_structure_vector(
    group: int,
    max_jobs: int,
    max_machines: int,
    legal_action_ratio: float = 0.0,
    machine_conflict_intensity: float = 0.0,
) -> torch.Tensor:
    """Return [n, m, work, legal ratio, frontier conflict] in normalized units."""
    n_jobs, n_machines = decode_scale_group(group, max_jobs, max_machines)
    jobs_den = float(max(1, max_jobs))
    machines_den = float(max(1, max_machines))
    return torch.tensor(
        [
            n_jobs / jobs_den,
            n_machines / machines_den,
            (n_jobs * n_machines) / (jobs_den * machines_den),
            float(legal_action_ratio),
            float(machine_conflict_intensity),
        ],
        dtype=torch.float32,
    )


def extract_probe_structure_features(
    batch: Mapping[str, object],
    group: int,
    max_jobs: int,
    max_machines: int,
) -> torch.Tensor:
    """Estimate dynamic structure features from a tensorized probe batch.

    The conflict value is the fraction of legal frontier operations competing
    with another legal frontier operation for the same next machine.
    """
    base = static_structure_vector(group, max_jobs, max_machines)
    states = batch.get("states")
    valid_mask = batch.get("mask")
    if not isinstance(states, Mapping) or not torch.is_tensor(valid_mask):
        return base
    legal = states.get("legal_action_mask")
    next_op = states.get("job_next_op")
    machine_sequence = states.get("machine_sequence")
    if not all(torch.is_tensor(x) for x in (legal, next_op, machine_sequence)):
        return base

    n_jobs, n_machines = decode_scale_group(group, max_jobs, max_machines)
    if n_jobs <= 0 or n_machines <= 0:
        return base

    legal = legal.detach().to(device="cpu")
    next_op = next_op.detach().to(device="cpu")
    machine_sequence = machine_sequence.detach().to(device="cpu")
    valid_mask = valid_mask.detach().to(device="cpu").bool()
    if legal.ndim < 3 or next_op.ndim < 3 or machine_sequence.ndim < 3:
        return base

    # Flatten only batch/time axes. The last axis retains padded job/op layout.
    legal = legal.reshape(-1, legal.shape[-1])
    next_op = next_op.reshape(-1, next_op.shape[-1])
    machine_sequence = machine_sequence.reshape(-1, machine_sequence.shape[-1])
    valid_mask = valid_mask.reshape(-1)
    usable = min(legal.shape[0], next_op.shape[0], machine_sequence.shape[0], valid_mask.shape[0])
    legal_ratios = []
    conflicts = []
    padded_ops = int(max_jobs) * int(max_machines)
    for idx in range(usable):
        if not bool(valid_mask[idx]):
            continue
        legal_row = legal[idx, :n_jobs] > 0.5
        legal_count = int(legal_row.sum().item())
        legal_ratios.append(legal_count / float(max(1, n_jobs)))
        if legal_count <= 1 or machine_sequence.shape[-1] < padded_ops:
            conflicts.append(0.0)
            continue
        ops = next_op[idx, :n_jobs].to(dtype=torch.long)
        valid_jobs = legal_row & (ops >= 0) & (ops < n_machines)
        job_ids = torch.arange(n_jobs, dtype=torch.long)[valid_jobs]
        if job_ids.numel() <= 1:
            conflicts.append(0.0)
            continue
        seq = machine_sequence[idx, :padded_ops].reshape(max_jobs, max_machines)
        machines = seq[job_ids, ops[valid_jobs]].to(dtype=torch.long)
        machines = machines[(machines >= 0) & (machines < n_machines)]
        if machines.numel() <= 1:
            conflicts.append(0.0)
            continue
        counts = torch.bincount(machines, minlength=n_machines)
        competing = counts[counts > 1].sum().item()
        conflicts.append(float(competing) / float(machines.numel()))

    if legal_ratios:
        base[3] = float(sum(legal_ratios) / len(legal_ratios))
    if conflicts:
        base[4] = float(sum(conflicts) / len(conflicts))
    return base


def neighbor_weights(structures: TensorMap, neighbor_k: int = 3, eps: float = 1e-8):
    """Build adaptive top-k Gaussian neighborhoods in standardized structure space."""
    groups = sorted(int(g) for g in structures)
    if len(groups) < 2:
        return {g: {} for g in groups}
    matrix = torch.stack([structures[g].detach().cpu().float() for g in groups])
    std = matrix.std(dim=0, unbiased=False)
    matrix = (matrix - matrix.mean(dim=0)) / torch.where(std > eps, std, torch.ones_like(std))
    result: Dict[int, Dict[int, float]] = {}
    k = min(max(1, int(neighbor_k)), len(groups) - 1)
    for i, group in enumerate(groups):
        distances = torch.linalg.vector_norm(matrix - matrix[i], dim=1)
        distances[i] = float("inf")
        chosen = torch.topk(distances, k=k, largest=False).indices
        finite = distances[chosen]
        bandwidth = max(float(finite[-1].item()), eps)
        raw = torch.exp(-(finite.pow(2)) / (2.0 * bandwidth * bandwidth))
        raw = raw / raw.sum().clamp_min(eps)
        result[group] = {groups[int(j)]: float(w) for j, w in zip(chosen, raw)}
    return result


@dataclass
class SSNCMaskController:
    """EMA state and score computation for SSNC binary-mask updates."""

    neighbor_k: int = 3
    structure_ema_decay: float = 0.9
    grad_norm_ema_decay: float = 0.9
    fisher_ema_decay: float = 0.9
    score_ema_decay: float = 0.9
    eps: float = 1e-8
    structure_ema: Dict[int, torch.Tensor] = field(default_factory=dict)
    grad_rms_ema: Dict[Tuple[str, int], torch.Tensor] = field(default_factory=dict)
    fisher_ema: Dict[Tuple[str, int], torch.Tensor] = field(default_factory=dict)
    score_ema: Dict[Tuple[str, int], torch.Tensor] = field(default_factory=dict)

    def state_dict(self):
        def pack(mapping):
            packed = {}
            for key, value in mapping.items():
                name = str(key) if not isinstance(key, tuple) else f"{key[0]}::{int(key[1])}"
                packed[name] = value.detach().cpu()
            return packed

        return {
            "version": 1,
            "structure_ema": pack(self.structure_ema),
            "grad_rms_ema": pack(self.grad_rms_ema),
            "fisher_ema": pack(self.fisher_ema),
            "score_ema": pack(self.score_ema),
        }

    def load_state_dict(self, state):
        if not isinstance(state, Mapping):
            return

        def unpack(source, tuple_keys=False):
            result = {}
            for key, value in (source or {}).items():
                if not torch.is_tensor(value):
                    continue
                if tuple_keys:
                    name, group = str(key).rsplit("::", 1)
                    parsed_key = (name, int(group))
                else:
                    parsed_key = int(key)
                result[parsed_key] = value.detach().cpu().float()
            return result

        self.structure_ema = unpack(state.get("structure_ema", {}), tuple_keys=False)
        self.grad_rms_ema = unpack(state.get("grad_rms_ema", {}), tuple_keys=True)
        self.fisher_ema = unpack(state.get("fisher_ema", {}), tuple_keys=True)
        self.score_ema = unpack(state.get("score_ema", {}), tuple_keys=True)

    @staticmethod
    def _ema(old: torch.Tensor | None, new: torch.Tensor, decay: float) -> torch.Tensor:
        new = new.detach().cpu().float()
        if old is None:
            return new.clone()
        return float(decay) * old + (1.0 - float(decay)) * new

    def update_structures(self, observations: TensorMap) -> Dict[int, torch.Tensor]:
        for group, vector in observations.items():
            group = int(group)
            self.structure_ema[group] = self._ema(
                self.structure_ema.get(group), vector, self.structure_ema_decay
            )
        return dict(self.structure_ema)

    def normalized_gradients(self, mask_name: str, gradients: TensorMap) -> Dict[int, torch.Tensor]:
        normalized = {}
        for group, grad in gradients.items():
            group = int(group)
            grad = grad.detach().cpu().float()
            rms = grad.pow(2).mean().add(self.eps).sqrt()
            key = (str(mask_name), group)
            self.grad_rms_ema[key] = self._ema(
                self.grad_rms_ema.get(key), rms.reshape(1), self.grad_norm_ema_decay
            )
            normalized[group] = grad / self.grad_rms_ema[key].clamp_min(self.eps)
        return normalized

    def compute_scores(self, mask_name: str, gradients: TensorMap, gamma: float):
        normalized = self.normalized_gradients(mask_name, gradients)
        structures = {g: self.structure_ema[g] for g in normalized if g in self.structure_ema}
        weights = neighbor_weights(structures, self.neighbor_k, self.eps)
        scores: Dict[int, torch.Tensor] = {}
        diagnostics = {"neighbors": weights, "groups": {}}
        for group, grad in normalized.items():
            group_weights = weights.get(group, {})
            usable = [(other, weight) for other, weight in group_weights.items() if other in normalized]
            if not usable:
                # Defensive fallback for incomplete probe batches; never includes self when peers exist.
                peers = [other for other in normalized if other != group]
                usable = [(other, 1.0 / len(peers)) for other in peers] if peers else [(group, 1.0)]
            total_weight = sum(weight for _, weight in usable)
            reference = sum(
                normalized[other] * (weight / max(total_weight, self.eps)) for other, weight in usable
            )
            agreement = grad * reference
            key = (str(mask_name), group)
            fisher_now = grad.pow(2)
            fisher = self._ema(self.fisher_ema.get(key), fisher_now, self.fisher_ema_decay)
            self.fisher_ema[key] = fisher
            fisher_scaled = fisher / fisher.mean().clamp_min(self.eps)
            raw_score = float(gamma) * agreement + fisher_scaled
            score = self._ema(self.score_ema.get(key), raw_score, self.score_ema_decay)
            self.score_ema[key] = score
            scores[group] = score
            cosine = torch.nn.functional.cosine_similarity(
                grad.reshape(1, -1), reference.reshape(1, -1), dim=1, eps=self.eps
            ).item()
            diagnostics["groups"][group] = {
                "gradient_rms_ema": float(self.grad_rms_ema[key].item()),
                "reference_cosine": float(cosine),
                "score_mean": float(score.mean().item()),
                "score_std": float(score.std(unbiased=False).item()),
            }
        return scores, diagnostics
