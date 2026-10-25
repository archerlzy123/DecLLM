import argparse
import copy
import json
from types import SimpleNamespace

import torch

from ssnc_inference_routing import (
    MASK_TABLE_NAMES,
    apply_unseen_scale_neighbor_routing,
)


DEFAULT_SCALES = "8x5,10x8,12x10,15x12,18x15,20x18"


def parse_scales(text):
    result = []
    for item in text.split(","):
        n_jobs, n_machines = item.strip().lower().split("x")
        result.append((int(n_jobs), int(n_machines)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_modules", required=True)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--neighbor_k", type=int, default=3)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    custom = torch.load(args.custom_modules, map_location="cpu")
    state = custom.get("ssnc_mask_state")
    if state is None:
        raise RuntimeError("Checkpoint does not contain ssnc_mask_state")

    base_tables = {}
    for name in MASK_TABLE_NAMES:
        value = custom.get(name)
        if not torch.is_tensor(value):
            raise RuntimeError(f"Checkpoint is missing {name}")
        base_tables[name] = value.detach().cpu().clone()

    reports = []
    for n_jobs, n_machines in parse_scales(args.scales):
        model = SimpleNamespace(n_jobs=20, n_machines=20)
        for name, value in base_tables.items():
            setattr(model, name, value.clone())
        before = {name: getattr(model, name).clone() for name in MASK_TABLE_NAMES}
        report = apply_unseen_scale_neighbor_routing(
            model=model,
            ssnc_state=copy.deepcopy(state),
            target_n_jobs=n_jobs,
            target_n_machines=n_machines,
            neighbor_k=args.neighbor_k,
            mode="neighbor_vote",
            vote_rule="density_preserving",
        )
        target_group = n_jobs * 21 + n_machines
        assert report["status"] == "routed"
        assert abs(sum(item["weight"] for item in report["neighbors"]) - 1.0) < 1e-6
        for name in MASK_TABLE_NAMES:
            after = getattr(model, name)
            assert torch.equal(after[:target_group], before[name][:target_group])
            assert torch.equal(after[target_group + 1 :], before[name][target_group + 1 :])
            assert set(torch.unique(after[target_group]).tolist()) <= {0.0, 1.0}
            assert report["masks"][name]["before_density"] == 1.0
        reports.append(report)

    payload = {
        "checkpoint": args.custom_modules,
        "neighbor_k": args.neighbor_k,
        "scales": reports,
        "status": "all_checks_passed",
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
