#!/usr/bin/env python3
"""Assert and record the checkpoint routing state for the six unseen scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


SCALES = [(8, 5), (10, 8), (12, 10), (15, 12), (18, 15), (20, 18)]
MASK_NAMES = [
    "scale_param_mask_gnn_encoder",
    "scale_param_mask_state_projection",
    "scale_param_mask_lora",
    "scale_param_mask_candidate_head",
    "scale_param_mask_scale_adapter",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if isinstance(state, dict) and "custom_state_dict" in state:
        state = state["custom_state_dict"]

    def locate(suffix):
        for key, value in state.items():
            if key == suffix or key.endswith("." + suffix) or key.endswith(suffix):
                if torch.is_tensor(value):
                    return key, value.detach().float().cpu()
        return None, None

    tables = {}
    keys = {}
    for name in MASK_NAMES:
        key, value = locate(name)
        assert value is not None, f"Missing checkpoint table: {name}"
        assert value.shape[0] == 441, (name, tuple(value.shape))
        keys[name] = key
        tables[name] = value

    report = {
        "status": "PASS",
        "checkpoint": str(Path(args.checkpoint)),
        "group_formula": "group_id = n_jobs * 21 + n_machines",
        "capacity": {"n_jobs": 20, "n_machines": 20},
        "mask_tables": keys,
        "scales": {},
    }
    for n_jobs, n_machines in SCALES:
        scale = f"{n_jobs}x{n_machines}"
        group_id = n_jobs * 21 + n_machines
        scale_report = {"group_id": group_id, "masks": {}}
        for name, table in tables.items():
            row = table[group_id]
            density = float((row > 0.5).float().mean())
            minimum, maximum = float(row.min()), float(row.max())
            assert density == 1.0, f"{scale} {name} is not all-one: density={density}"
            scale_report["masks"][name] = {
                "density": density,
                "min": minimum,
                "max": maximum,
            }
        report["scales"][scale] = scale_report

    for name in ["scale_context_adapter_alpha", "scale_candidate_residual_alpha"]:
        key, table = locate(name)
        if table is not None:
            report[name] = {
                f"{n}x{m}": float(table[n * 21 + m].reshape(-1)[0]) for n, m in SCALES
            }
            report[name + "_checkpoint_key"] = key

    key, embedding = locate("scale_embedding.weight")
    if embedding is not None:
        report["scale_embedding_checkpoint_key"] = key
        report["scale_embedding_rows"] = {
            f"{n}x{m}": {
                "norm": float(embedding[n * 21 + m].norm()),
                "mean": float(embedding[n * 21 + m].mean()),
                "std": float(embedding[n * 21 + m].std()),
            }
            for n, m in SCALES
        }
    key, scale_alpha = locate("scale_alpha")
    if scale_alpha is not None:
        report["scale_alpha_checkpoint_key"] = key
        report["scale_alpha"] = float(scale_alpha.reshape(-1)[0])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("UNSEEN_ROUTING_PREFLIGHT=PASS")
    for scale, info in report["scales"].items():
        densities = [entry["density"] for entry in info["masks"].values()]
        print(f"{scale}: group_id={info['group_id']} all_one={all(x == 1.0 for x in densities)}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
