"""Canonical graph-state tensorization for mixed-size JSSP models.

All training datasets, training-time rollouts, and external inference should
use this module.  Keeping the transformation in one place prevents a raw
``(n_jobs, n_machines)`` matrix from being flattened before its rows are
padded to the model capacity.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import torch


LEGACY_TRAIN_COMPATIBLE = "legacy_train_compatible"
CANONICAL_V2 = "canonical_v2"
VALID_TENSORIZATION_MODES = (LEGACY_TRAIN_COMPATIBLE, CANONICAL_V2)

GRAPH_STATE_KEYS = (
    "job_next_op",
    "job_completion_mask",
    "remaining_processing_times",
    "normalized_machine_availability",
    "machine_busy_mask",
    "overall_info",
    "machine_sequence",
    "scheduled_machine_adj",
    "legal_action_mask",
    "problem_n_jobs",
    "problem_n_machines",
)

MATRIX_KEYS = {
    "job_completion_mask",
    "remaining_processing_times",
    "machine_sequence",
}
JOB_VECTOR_KEYS = {"job_next_op", "legal_action_mask"}
MACHINE_VECTOR_KEYS = {
    "normalized_machine_availability",
    "machine_busy_mask",
}

_ACTIVE_MODE = LEGACY_TRAIN_COMPATIBLE


def set_state_tensorization_mode(mode: str) -> None:
    """Set the process-wide mode used by rollout helpers."""

    global _ACTIVE_MODE
    _ACTIVE_MODE = validate_state_tensorization_mode(mode)


def get_state_tensorization_mode() -> str:
    return _ACTIVE_MODE


def validate_state_tensorization_mode(mode: str) -> str:
    mode = str(mode or LEGACY_TRAIN_COMPATIBLE).strip().lower()
    if mode not in VALID_TENSORIZATION_MODES:
        raise ValueError(
            f"Unknown state tensorization mode {mode!r}; "
            f"expected one of {VALID_TENSORIZATION_MODES}."
        )
    return mode


def _positive_int(value: object, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _to_float_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _pad_or_trim_vector(value: object, target: int) -> torch.Tensor:
    flat = _to_float_tensor(value).reshape(-1)
    out = torch.zeros((target,), dtype=torch.float32)
    count = min(target, int(flat.numel()))
    if count:
        out[:count] = flat[:count]
    return out


def _matrix_from_value(
    value: object,
    actual_jobs: int,
    actual_machines: int,
    max_jobs: int,
    max_machines: int,
) -> torch.Tensor:
    tensor = _to_float_tensor(value)
    if tensor.ndim == 2:
        if tuple(tensor.shape) == (max_jobs, max_machines):
            return tensor
        if tuple(tensor.shape) == (actual_jobs, actual_machines):
            return tensor
    if tensor.numel() == max_jobs * max_machines:
        return tensor.reshape(max_jobs, max_machines)
    if tensor.numel() == actual_jobs * actual_machines:
        return tensor.reshape(actual_jobs, actual_machines)
    raise ValueError(
        "Matrix state has incompatible shape "
        f"{tuple(tensor.shape)} for actual={actual_jobs}x{actual_machines}, "
        f"capacity={max_jobs}x{max_machines}."
    )


def _pad_matrix_rows(
    value: object,
    actual_jobs: int,
    actual_machines: int,
    max_jobs: int,
    max_machines: int,
) -> torch.Tensor:
    matrix = _matrix_from_value(
        value, actual_jobs, actual_machines, max_jobs, max_machines
    )
    if tuple(matrix.shape) == (max_jobs, max_machines):
        return matrix.contiguous()
    out = torch.zeros((max_jobs, max_machines), dtype=torch.float32)
    jobs = min(actual_jobs, max_jobs, int(matrix.shape[0]))
    machines = min(actual_machines, max_machines, int(matrix.shape[1]))
    if jobs and machines:
        out[:jobs, :machines] = matrix[:jobs, :machines]
    return out


def _adjacency_from_value(
    value: object,
    actual_nodes: int,
    max_nodes: int,
) -> torch.Tensor:
    tensor = _to_float_tensor(value)
    if tensor.ndim == 2:
        if tuple(tensor.shape) == (max_nodes, max_nodes):
            return tensor
        if tuple(tensor.shape) == (actual_nodes, actual_nodes):
            return tensor
    if tensor.numel() == max_nodes * max_nodes:
        return tensor.reshape(max_nodes, max_nodes)
    if tensor.numel() == actual_nodes * actual_nodes:
        return tensor.reshape(actual_nodes, actual_nodes)
    raise ValueError(
        "scheduled_machine_adj has incompatible shape "
        f"{tuple(tensor.shape)} for actual_nodes={actual_nodes}, "
        f"capacity_nodes={max_nodes}."
    )


def _pad_adjacency(
    value: object,
    actual_jobs: int,
    actual_machines: int,
    max_jobs: int,
    max_machines: int,
    mode: str,
) -> torch.Tensor:
    actual_nodes = actual_jobs * actual_machines
    max_nodes = max_jobs * max_machines
    adjacency = _adjacency_from_value(value, actual_nodes, max_nodes)
    if tuple(adjacency.shape) == (max_nodes, max_nodes):
        return adjacency.contiguous()

    out = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)
    if mode == LEGACY_TRAIN_COMPATIBLE:
        # Reproduce the layout used by the existing checkpoints' training
        # dataset: the old adjacency was copied into the top-left block.
        out[:actual_nodes, :actual_nodes] = adjacency
        return out

    # Correct mapping from old node id j * actual_machines + op to the model
    # capacity layout j * max_machines + op.
    job_ids = torch.arange(actual_jobs, dtype=torch.long).repeat_interleave(
        actual_machines
    )
    op_ids = torch.arange(actual_machines, dtype=torch.long).repeat(actual_jobs)
    remapped = job_ids * max_machines + op_ids
    out[remapped[:, None], remapped[None, :]] = adjacency
    return out


def _zero_for_key(key: str, max_jobs: int, max_machines: int) -> torch.Tensor:
    if key in JOB_VECTOR_KEYS:
        return torch.zeros((max_jobs,), dtype=torch.float32)
    if key in MACHINE_VECTOR_KEYS:
        return torch.zeros((max_machines,), dtype=torch.float32)
    if key in MATRIX_KEYS:
        return torch.zeros((max_jobs * max_machines,), dtype=torch.float32)
    if key == "scheduled_machine_adj":
        max_nodes = max_jobs * max_machines
        return torch.zeros((max_nodes * max_nodes,), dtype=torch.float32)
    if key == "overall_info":
        return torch.zeros((3,), dtype=torch.float32)
    return torch.zeros((1,), dtype=torch.float32)


def tensorize_graph_state(
    state: Mapping[str, object],
    *,
    actual_n_jobs: int,
    actual_n_machines: int,
    max_jobs: int,
    max_machines: int,
    keys: Optional[Iterable[str]] = None,
    mode: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Tensorize one raw environment state into model-capacity flat features.

    Ordinary operation matrices are always padded row-by-row before flattening.
    ``mode`` only controls how the legacy scheduled-machine adjacency is laid
    out, allowing old checkpoints to be evaluated on their training layout.
    """

    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a mapping state, got {type(state)!r}")
    actual_n_jobs = _positive_int(actual_n_jobs, "actual_n_jobs")
    actual_n_machines = _positive_int(actual_n_machines, "actual_n_machines")
    max_jobs = _positive_int(max_jobs, "max_jobs")
    max_machines = _positive_int(max_machines, "max_machines")
    if actual_n_jobs > max_jobs or actual_n_machines > max_machines:
        raise ValueError(
            f"Problem {actual_n_jobs}x{actual_n_machines} exceeds model "
            f"capacity {max_jobs}x{max_machines}."
        )
    mode = validate_state_tensorization_mode(mode or get_state_tensorization_mode())
    selected_keys = tuple(keys) if keys is not None else GRAPH_STATE_KEYS

    result: dict[str, torch.Tensor] = {}
    for key in selected_keys:
        if key == "problem_n_jobs":
            result[key] = torch.tensor([float(actual_n_jobs)], dtype=torch.float32)
            continue
        if key == "problem_n_machines":
            result[key] = torch.tensor([float(actual_n_machines)], dtype=torch.float32)
            continue
        if key not in state:
            result[key] = _zero_for_key(key, max_jobs, max_machines)
            continue

        value = state[key]
        if key in JOB_VECTOR_KEYS:
            result[key] = _pad_or_trim_vector(value, max_jobs)
        elif key in MACHINE_VECTOR_KEYS:
            result[key] = _pad_or_trim_vector(value, max_machines)
        elif key == "overall_info":
            result[key] = _pad_or_trim_vector(value, 3)
        elif key in MATRIX_KEYS:
            result[key] = _pad_matrix_rows(
                value,
                actual_n_jobs,
                actual_n_machines,
                max_jobs,
                max_machines,
            ).reshape(-1)
        elif key == "scheduled_machine_adj":
            result[key] = _pad_adjacency(
                value,
                actual_n_jobs,
                actual_n_machines,
                max_jobs,
                max_machines,
                mode,
            ).reshape(-1)
        else:
            result[key] = _to_float_tensor(value).reshape(-1)
    return result


def tensorize_graph_state_history(
    states: Iterable[Mapping[str, object]],
    *,
    actual_n_jobs: int,
    actual_n_machines: int,
    max_jobs: int,
    max_machines: int,
    keys: Optional[Iterable[str]] = None,
    mode: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Tensorize and stack a sequence as ``(T, feature_dim)`` tensors."""

    states = list(states)
    if not states:
        raise ValueError("Cannot tensorize an empty state history.")
    selected_keys = tuple(keys) if keys is not None else GRAPH_STATE_KEYS
    rows = [
        tensorize_graph_state(
            state,
            actual_n_jobs=actual_n_jobs,
            actual_n_machines=actual_n_machines,
            max_jobs=max_jobs,
            max_machines=max_machines,
            keys=selected_keys,
            mode=mode,
        )
        for state in states
    ]
    return {
        key: torch.stack([row[key] for row in rows], dim=0)
        for key in selected_keys
    }
