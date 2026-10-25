import unittest

import torch

from ssnc_mask import (
    SSNCMaskController,
    extract_probe_structure_features,
    neighbor_weights,
    static_structure_vector,
)


def gid(n, m, max_m=20):
    return n * (max_m + 1) + m


class SSNCMaskTests(unittest.TestCase):
    def test_neighborhood_excludes_self_and_normalizes(self):
        groups = [gid(10, 5), gid(10, 10), gid(15, 10), gid(20, 20)]
        structures = {g: static_structure_vector(g, 20, 20, 0.5, 0.3) for g in groups}
        weights = neighbor_weights(structures, neighbor_k=2)
        for group in groups:
            self.assertNotIn(group, weights[group])
            self.assertEqual(len(weights[group]), 2)
            self.assertAlmostEqual(sum(weights[group].values()), 1.0, places=6)
        self.assertIn(gid(10, 10), weights[gid(10, 5)])
        self.assertNotIn(gid(20, 20), weights[gid(10, 5)])

    def test_rms_normalization_removes_scale_amplitude(self):
        ctl = SSNCMaskController(grad_norm_ema_decay=0.0)
        gradients = {gid(10, 5): torch.tensor([1.0, -2.0]), gid(15, 10): torch.tensor([100.0, -200.0])}
        normed = ctl.normalized_gradients("gnn", gradients)
        self.assertTrue(torch.allclose(normed[gid(10, 5)], normed[gid(15, 10)], atol=1e-6))

    def test_dynamic_conflict_feature(self):
        group = gid(3, 3, max_m=3)
        legal = torch.tensor([[[1, 1, 1]]], dtype=torch.float32)
        next_op = torch.tensor([[[0, 0, 0]]], dtype=torch.float32)
        # All three frontier operations require machine 0.
        seq = torch.tensor([[[0, 1, 2, 0, 2, 1, 0, 1, 2]]], dtype=torch.float32)
        batch = {"states": {"legal_action_mask": legal, "job_next_op": next_op, "machine_sequence": seq}, "mask": torch.ones(1, 1)}
        vector = extract_probe_structure_features(batch, group, 3, 3)
        self.assertAlmostEqual(float(vector[3]), 1.0, places=6)
        self.assertAlmostEqual(float(vector[4]), 1.0, places=6)

    def test_scores_are_finite_and_shape_preserving(self):
        groups = [gid(10, 5), gid(10, 10), gid(15, 10)]
        ctl = SSNCMaskController(neighbor_k=2, score_ema_decay=0.5)
        ctl.update_structures({g: static_structure_vector(g, 20, 20, 0.5, 0.2) for g in groups})
        gradients = {g: torch.linspace(-1.0, 1.0, 16) * (idx + 1) for idx, g in enumerate(groups)}
        scores, diag = ctl.compute_scores("candidate", gradients, gamma=10.0)
        self.assertEqual(set(scores), set(groups))
        self.assertTrue(all(score.shape == (16,) and torch.isfinite(score).all() for score in scores.values()))
        self.assertEqual(set(diag["groups"]), set(groups))
        restored = SSNCMaskController()
        restored.load_state_dict(ctl.state_dict())
        self.assertEqual(set(restored.structure_ema), set(groups))
        self.assertEqual(set(restored.score_ema), set(ctl.score_ema))


if __name__ == "__main__":
    unittest.main()
