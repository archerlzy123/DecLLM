import unittest

import torch

from state_tensorization import CANONICAL_V2, GRAPH_STATE_KEYS, tensorize_graph_state


MAX_JOBS = 20
MAX_MACHINES = 20
SCALES = ((10, 5), (10, 10), (15, 10), (20, 15), (20, 20))


def make_state(n_jobs, n_machines):
    matrix = torch.zeros((n_jobs, n_machines), dtype=torch.float32)
    for job in range(n_jobs):
        for op in range(n_machines):
            matrix[job, op] = 1000 * job + op + 1
    sequence = torch.arange(n_machines, dtype=torch.float32).repeat(n_jobs, 1)
    nodes = n_jobs * n_machines
    adjacency = torch.zeros((nodes, nodes), dtype=torch.float32)
    # A scheduled-machine edge inside job 1 exposes the stride mismatch when m < 20.
    old_src = n_machines
    old_dst = n_machines + min(1, n_machines - 1)
    adjacency[old_src, old_dst] = 1.0
    return {
        "job_next_op": torch.zeros(n_jobs),
        "job_completion_mask": matrix.clone(),
        "remaining_processing_times": matrix.clone(),
        "normalized_machine_availability": torch.arange(n_machines, dtype=torch.float32),
        "machine_busy_mask": torch.zeros(n_machines),
        "overall_info": torch.tensor([1.0, 2.0, 3.0]),
        "machine_sequence": sequence,
        "scheduled_machine_adj": adjacency,
        "legal_action_mask": torch.ones(n_jobs),
    }, matrix


class CanonicalTensorizationTests(unittest.TestCase):
    def test_row_padding_and_adjacency_remap_for_representative_scales(self):
        for n_jobs, n_machines in SCALES:
            with self.subTest(scale=f"{n_jobs}x{n_machines}"):
                state, sentinel = make_state(n_jobs, n_machines)
                result = tensorize_graph_state(
                    state,
                    actual_n_jobs=n_jobs,
                    actual_n_machines=n_machines,
                    max_jobs=MAX_JOBS,
                    max_machines=MAX_MACHINES,
                    keys=GRAPH_STATE_KEYS,
                    mode=CANONICAL_V2,
                )
                padded = result["remaining_processing_times"].reshape(MAX_JOBS, MAX_MACHINES)
                self.assertTrue(torch.equal(padded[:n_jobs, :n_machines], sentinel))
                self.assertEqual(int(torch.count_nonzero(padded[:n_jobs, n_machines:])), 0)
                self.assertEqual(int(torch.count_nonzero(padded[n_jobs:, :])), 0)

                adjacency = result["scheduled_machine_adj"].reshape(
                    MAX_JOBS * MAX_MACHINES, MAX_JOBS * MAX_MACHINES
                )
                new_src = MAX_MACHINES
                new_dst = MAX_MACHINES + min(1, n_machines - 1)
                self.assertEqual(float(adjacency[new_src, new_dst]), 1.0)
                self.assertEqual(int(torch.count_nonzero(adjacency)), 1)
                if n_machines < MAX_MACHINES:
                    self.assertEqual(float(adjacency[n_machines, n_machines + 1]), 0.0)

                self.assertEqual(float(result["problem_n_jobs"].item()), float(n_jobs))
                self.assertEqual(float(result["problem_n_machines"].item()), float(n_machines))


if __name__ == "__main__":
    unittest.main()
