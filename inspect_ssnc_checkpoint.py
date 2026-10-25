"""Small read-only checker for SSNC controller state in a DT checkpoint."""

import argparse

import torch


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
args = parser.parse_args()

payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
state = payload.get("ssnc_mask_state") if isinstance(payload, dict) else None
if not isinstance(state, dict):
    raise SystemExit("SSNC_STATE_MISSING")
print("SSNC_STATE_OK")
print("keys=" + ",".join(sorted(state)))
print(f"structure_groups={len(state.get('structure_ema', {}))}")
print(f"module_scale_scores={len(state.get('score_ema', {}))}")
