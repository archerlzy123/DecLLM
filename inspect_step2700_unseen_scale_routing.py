#!/usr/bin/env python3
from pathlib import Path
import torch

CKPT = Path("/path/to/local_workspace/output_full_scale_param_mask_fixed_s010_i50_min2_panel55_bnb_nodynamo_continue_gap600_lr8e6_12ep/best_gap_model_step_2700/dt_custom_modules.pt")
obj = torch.load(CKPT, map_location="cpu", weights_only=False)
state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
if isinstance(state, dict) and "custom_state_dict" in state:
    state = state["custom_state_dict"]

def locate(suffix):
    for key, value in state.items():
        if key == suffix or key.endswith("." + suffix) or key.endswith(suffix):
            if torch.is_tensor(value):
                return key, value.detach().float().cpu()
    return None, None

names = [
    "scale_param_mask_gnn_encoder",
    "scale_param_mask_state_projection",
    "scale_param_mask_lora",
    "scale_param_mask_candidate_head",
    "scale_param_mask_scale_adapter",
]
tables = {}
for name in names:
    key, table = locate(name)
    print(name, "key=", key, "shape=", None if table is None else tuple(table.shape))
    if table is not None:
        tables[name] = table

M = 20
active = []
for n in range(1, M + 1):
    for m in range(1, M + 1):
        idx = n * (M + 1) + m
        densities = {k.replace("scale_param_mask_", ""): float((v[idx] > 0.5).float().mean()) for k, v in tables.items()}
        if any(d < 0.999999 for d in densities.values()):
            active.append((n, m, densities))

print("MASK_ROWS_WITH_ANY_CLOSED_CHANNELS")
for n, m, densities in active:
    print(f"{n}x{m}", " ".join(f"{k}={v:.4f}" for k, v in densities.items()))

for name in ["scale_context_adapter_alpha", "scale_candidate_residual_alpha"]:
    key, table = locate(name)
    print(name, "key=", key, "shape=", None if table is None else tuple(table.shape))
    if table is not None:
        changed = []
        for n in range(1, M + 1):
            for m in range(1, M + 1):
                idx = n * (M + 1) + m
                value = float(table[idx].reshape(-1)[0])
                if abs(value - 0.01) > 1e-6:
                    changed.append((n, m, value))
        print("ALPHA_ROWS_CHANGED_FROM_0.01", name)
        for n, m, value in changed:
            print(f"{n}x{m}={value:.8f}")

probe = [(10, 5), (10, 10), (12, 10), (15, 10), (15, 15), (18, 15), (20, 10), (20, 15), (20, 20)]
print("PROBE_ROWS")
for n, m in probe:
    idx = n * (M + 1) + m
    parts = []
    for k, v in tables.items():
        density = float((v[idx] > 0.5).float().mean())
        parts.append(f"{k.replace('scale_param_mask_', '')}={density:.4f}")
    print(f"{n}x{m} id={idx} " + " ".join(parts))

for name in ["scale_alpha", "scale_embedding.weight"]:
    key, value = locate(name)
    if value is not None:
        print(name, "key=", key, "shape=", tuple(value.shape), "value_or_norm=", value.tolist() if value.numel() <= 8 else float(value.norm()))
