#!/usr/bin/env python3
"""Parse dt_train.py run.log rollout panel gaps and save summary plots."""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_rollout_records(log_path):
    best_re = re.compile(r"\[RolloutPanel\] best_target_return=([^,]+), score=([\-0-9.]+)%")
    scale_re = re.compile(r"\[RolloutPanel\]\s+(\d+x\d+): gap=([\-0-9.]+)% \((\d+)/(\d+) valid\)")
    step_re = re.compile(r"Step\s+(\d+)\s+- Evaluation Relative Gap Score: ([\-0-9.]+)%")

    records = []
    current = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = best_re.search(line)
            if m:
                current = {
                    "step": None,
                    "score": float(m.group(2)),
                    "target_return": float(m.group(1)),
                    "per_scale": {},
                }
                continue

            m = scale_re.search(line)
            if m and current is not None:
                scale, gap, valid, total = m.groups()
                current["per_scale"][scale] = {
                    "mean_gap": float(gap),
                    "valid_instances": int(valid),
                    "total_instances": int(total),
                }
                continue

            m = step_re.search(line)
            if m and current is not None:
                current["step"] = int(m.group(1))
                current["score"] = float(m.group(2))
                records.append(current)
                current = None

    return records


def save_plot(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "validation_rollout_gap_history_from_log.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    if not records:
        print(f"No rollout records found. Wrote empty history to {json_path}")
        return

    steps = np.array([int(r["step"]) for r in records], dtype=int)
    scores = np.array([float(r.get("score", np.nan)) for r in records], dtype=float)
    scales = sorted({scale for r in records for scale in (r.get("per_scale", {}) or {}).keys()})

    plt.figure(figsize=(12, 7))
    valid = ~np.isnan(scores)
    if valid.any():
        plt.plot(steps[valid], scores[valid], marker="o", linewidth=2.2, label="Panel Score")

    for scale in scales:
        values = np.array([
            float((r.get("per_scale", {}) or {}).get(scale, {}).get("mean_gap", np.nan))
            for r in records
        ], dtype=float)
        valid = ~np.isnan(values)
        if valid.any():
            plt.plot(steps[valid], values[valid], marker=".", linewidth=1.4, alpha=0.85, label=scale)

    plt.title("Validation Rollout Gap by Evaluation Step")
    plt.xlabel("Global Steps")
    plt.ylabel("Relative Gap (%)")
    plt.grid(True)
    plt.legend()
    plot_path = os.path.join(output_dir, "validation_rollout_gap_by_step_from_log.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    print(f"Saved {json_path}")
    print(f"Saved {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", help="Path to run.log")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to log directory.")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.log_path)) or "."
    save_plot(parse_rollout_records(args.log_path), output_dir)


if __name__ == "__main__":
    main()
