from types import SimpleNamespace

import torch

from ssnc_inference_routing import (
    MASK_TABLE_NAMES,
    apply_unseen_scale_neighbor_routing,
    infer_target_structure_and_neighbors,
    weighted_vote_mask,
)


def _group(n_jobs, n_machines):
    return n_jobs * 21 + n_machines


def _dummy_model():
    model = SimpleNamespace(n_jobs=20, n_machines=20)
    widths = [8, 12, 4, 12, 8]
    for name, width in zip(MASK_TABLE_NAMES, widths):
        table = torch.ones(441, width)
        table[_group(10, 5), width // 4 :] = 0
        table[_group(10, 10), width // 2 :] = 0
        table[_group(15, 10), (3 * width) // 4 :] = 0
        setattr(model, name, table)
    return model


def _dummy_state():
    return {
        "structure_ema": {
            str(_group(10, 5)): torch.tensor([0.50, 0.25, 0.125, 0.25, 0.38]),
            str(_group(10, 10)): torch.tensor([0.50, 0.50, 0.250, 0.22, 0.19]),
            str(_group(15, 10)): torch.tensor([0.75, 0.50, 0.375, 0.17, 0.32]),
        }
    }


def test_weighted_vote_density_preserving_is_binary_and_deterministic():
    rows = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.float32)
    weights = torch.tensor([0.75, 0.25])
    first, scores = weighted_vote_mask(rows, weights, rule="density_preserving")
    second, _ = weighted_vote_mask(rows, weights, rule="density_preserving")
    assert torch.equal(first, second)
    assert set(first.tolist()) <= {0.0, 1.0}
    assert int(first.sum()) == 2
    assert torch.allclose(scores, torch.tensor([1.0, 0.75, 0.25, 0.0]))


def test_unseen_row_routes_without_touching_seen_rows():
    model = _dummy_model()
    state = _dummy_state()
    unseen_group = _group(12, 10)
    originals = {
        name: getattr(model, name).clone()
        for name in MASK_TABLE_NAMES
    }
    report = apply_unseen_scale_neighbor_routing(
        model,
        state,
        target_n_jobs=12,
        target_n_machines=10,
        neighbor_k=3,
        mode="neighbor_vote",
    )
    assert report["status"] == "routed"
    assert report["routed_mask_tables"] == len(MASK_TABLE_NAMES)
    assert abs(sum(item["weight"] for item in report["neighbors"]) - 1.0) < 1e-6
    for name in MASK_TABLE_NAMES:
        after = getattr(model, name)
        assert float(after[unseen_group].mean()) < 1.0
        assert torch.equal(after[_group(10, 5)], originals[name][_group(10, 5)])
        assert torch.equal(after[_group(10, 10)], originals[name][_group(10, 10)])
        assert torch.equal(after[_group(15, 10)], originals[name][_group(15, 10)])
        assert report["masks"][name]["before_density"] == 1.0


def test_seen_scale_keeps_exact_checkpoint_row():
    model = _dummy_model()
    state = _dummy_state()
    original = model.scale_param_mask_candidate_head.clone()
    report = apply_unseen_scale_neighbor_routing(
        model,
        state,
        target_n_jobs=10,
        target_n_machines=10,
        mode="neighbor_vote",
    )
    assert report["status"] == "seen_scale_exact_row_unchanged"
    assert torch.equal(model.scale_param_mask_candidate_head, original)


def test_neighbor_search_returns_requested_count_and_five_features():
    target, neighbors = infer_target_structure_and_neighbors(
        _dummy_state(),
        target_n_jobs=12,
        target_n_machines=10,
        max_jobs=20,
        max_machines=20,
        neighbor_k=2,
    )
    assert target.shape == (5,)
    assert len(neighbors) == 2
    assert all(item["group_id"] in _dummy_state()["structure_ema"] or str(item["group_id"]) in _dummy_state()["structure_ema"] for item in neighbors)
