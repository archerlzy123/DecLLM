"""Verify dataset, training-rollout, and external-inference state parity."""

import torch

from dt_data_processing import JSSPDecisionTransformerDataset
from dt_infer import _obs_to_state_dict
from dt_train import process_state_for_batch
from state_tensorization import CANONICAL_V2, GRAPH_STATE_KEYS, tensorize_graph_state
from tests.test_state_tensorization_canonical_v2 import MAX_JOBS, MAX_MACHINES, SCALES, make_state


for n_jobs, n_machines in SCALES:
    state, _ = make_state(n_jobs, n_machines)
    expected = tensorize_graph_state(
        state,
        actual_n_jobs=n_jobs,
        actual_n_machines=n_machines,
        max_jobs=MAX_JOBS,
        max_machines=MAX_MACHINES,
        keys=GRAPH_STATE_KEYS,
        mode=CANONICAL_V2,
    )

    trajectory = {
        "trajectory_length": 1,
        "states": [state],
        "actions": torch.tensor([0], dtype=torch.long),
        "returns_to_go": torch.tensor([0.0]),
        "timesteps": torch.tensor([0], dtype=torch.long),
        "mask": torch.tensor([1.0]),
    }
    dataset = JSSPDecisionTransformerDataset(
        [trajectory],
        max_length=1,
        max_jobs=MAX_JOBS,
        max_machines=MAX_MACHINES,
        state_tensorization_mode=CANONICAL_V2,
    )
    dataset_state = dataset[0]["states"]
    rollout_state = process_state_for_batch(
        [state],
        device=torch.device("cpu"),
        max_jobs=MAX_JOBS,
        max_machines=MAX_MACHINES,
        state_tensorization_mode=CANONICAL_V2,
    )
    inference_state = _obs_to_state_dict(
        state,
        GRAPH_STATE_KEYS,
        problem_n_jobs=n_jobs,
        problem_n_machines=n_machines,
        model_max_jobs=MAX_JOBS,
        model_max_machines=MAX_MACHINES,
        state_tensorization_mode=CANONICAL_V2,
    )

    for key in GRAPH_STATE_KEYS:
        assert torch.equal(dataset_state[key][0], expected[key]), (n_jobs, n_machines, key, "dataset")
        assert torch.equal(rollout_state[key][0, 0], expected[key]), (n_jobs, n_machines, key, "rollout")
        assert torch.equal(inference_state[key], expected[key]), (n_jobs, n_machines, key, "inference")

print("CANONICAL_V2_ENTRYPOINT_PARITY_OK scales=" + ",".join(f"{n}x{m}" for n, m in SCALES))
