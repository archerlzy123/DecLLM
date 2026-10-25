#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA


SCALES = [
    (10, 5), (10, 10), (10, 15),
    (15, 5), (15, 10), (15, 15), (15, 20),
    (20, 5), (20, 10), (20, 15), (20, 20),
]
LABELS = [f"{j}x{m}" for j, m in SCALES]
MODULE_KEYS = {
    "gnn": "scale_param_mask_gnn_encoder",
    "state": "scale_param_mask_state_projection",
    "lora": "scale_param_mask_lora",
    "candidate": "scale_param_mask_candidate_head",
    "adapter": "scale_param_mask_scale_adapter",
}
COLORS = {
    "10x5": "#172a8a",
    "10x10": "#a018a7",
    "10x15": "#ef7049",
    "15x5": "#1c2c91",
    "15x10": "#a51caf",
    "15x15": "#f0744e",
    "15x20": "#e7ef25",
    "20x5": "#12257e",
    "20x10": "#94129f",
    "20x15": "#e86d47",
    "20x20": "#e3ef22",
}


def parse_steps(text):
    return [int(x) for x in text.split(",") if x.strip()]


def load_custom(path):
    obj = torch.load(path, map_location="cpu")
    return obj.get("state_dict", obj) if isinstance(obj, dict) else obj


def load_masks(ckpt_dir):
    sd = load_custom(Path(ckpt_dir) / "dt_custom_modules.pt")
    out = {}
    for module, key in MODULE_KEYS.items():
        tensor = sd.get(key)
        if not torch.is_tensor(tensor):
            raise KeyError(f"missing tensor {key} in {ckpt_dir}")
        rows = []
        for jobs, machines in SCALES:
            rows.append((tensor[jobs * 21 + machines] <= 0.5).cpu().numpy().astype(np.bool_))
        out[module] = np.stack(rows, axis=0)
    out["overall"] = np.concatenate([out[m] for m in MODULE_KEYS], axis=1)
    return out


def jaccard_matrix(x):
    n = x.shape[0]
    mat = np.eye(n, dtype=np.float64)
    for i, j in itertools.combinations(range(n), 2):
        inter = np.logical_and(x[i], x[j]).sum()
        union = np.logical_or(x[i], x[j]).sum()
        mat[i, j] = mat[j, i] = float(inter / union) if union else 1.0
    return mat


def offdiag_values(mat):
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def scale_distance(i, j):
    a, b = SCALES[i], SCALES[j]
    return math.sqrt(((a[0] - b[0]) / 5.0) ** 2 + ((a[1] - b[1]) / 5.0) ** 2)


def pair_groups():
    near, far, dists = [], [], []
    for i, j in itertools.combinations(range(len(SCALES)), 2):
        d = scale_distance(i, j)
        dists.append((i, j, d))
        if d <= 1.01:
            near.append((i, j))
        if d >= 2.0:
            far.append((i, j))
    return near, far, dists


def mean_pairs(mat, pairs):
    return float(np.mean([mat[i, j] for i, j in pairs])) if pairs else float("nan")


def rank_corr(x, y):
    rx = pd.Series(np.asarray(x)).rank(method="average").to_numpy()
    ry = pd.Series(np.asarray(y)).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def ckpt_for_step(run_dir, step):
    candidates = [
        run_dir / f"best_gap_model_step_{step}",
        run_dir / f"best_model_step_{step}",
    ]
    for path in candidates:
        if (path / "dt_custom_modules.pt").exists():
            return path
    return None


def parse_log(log_path):
    steps = {}
    metrics = {}
    cur = None
    with open(log_path, errors="replace") as f:
        for line in f:
            m = re.search(
                r"Global Step (\d+) - Eval Loss: ([0-9.]+).*AccGain: ([0-9.]+).*NonSingleGain: ([0-9.]+)",
                line,
            )
            if m:
                cur = int(m.group(1))
                metrics[cur] = {
                    "eval_loss": float(m.group(2)),
                    "accgain": float(m.group(3)),
                    "nonsingle_gain": float(m.group(4)),
                }
                steps.setdefault(cur, {})
                continue
            m = re.search(r"\[RolloutPanel\] ([0-9]+x[0-9]+): gap=([0-9.]+)%", line)
            if m and cur is not None:
                steps.setdefault(cur, {})[m.group(1)] = float(m.group(2))
                continue
            m = re.search(r"Step (\d+) - Evaluation Relative Gap Score: ([0-9.]+)%", line)
            if m:
                s = int(m.group(1))
                steps.setdefault(s, {})["score"] = float(m.group(2))
                cur = None
    return steps, metrics


def write_rollout_tables(steps, metrics, out_dir):
    rollout_steps = sorted(s for s, row in steps.items() if "score" in row)
    summary_rows = []
    scale_rows = []
    for step in rollout_steps:
        row = {"step": step, "rollout_score": steps[step]["score"]}
        row.update(metrics.get(step, {}))
        summary_rows.append(row)
        for scale in LABELS:
            if scale in steps[step]:
                scale_rows.append({"step": step, "scale": scale, "gap": steps[step][scale]})

    summary = pd.DataFrame(summary_rows)
    per_scale = pd.DataFrame(scale_rows)
    summary.to_csv(out_dir / "rollout_summary.csv", index=False)
    per_scale.to_csv(out_dir / "rollout_per_scale_long.csv", index=False)
    if not per_scale.empty:
        per_scale.pivot(index="scale", columns="step", values="gap").loc[LABELS].to_csv(
            out_dir / "rollout_per_scale_wide.csv"
        )

    if not summary.empty:
        plt.figure(figsize=(8, 4.5))
        plt.plot(summary["step"], summary["rollout_score"], marker="o", label="rollout gap")
        best_idx = summary["rollout_score"].idxmin()
        plt.scatter([summary.loc[best_idx, "step"]], [summary.loc[best_idx, "rollout_score"]], s=110, color="crimson", zorder=3)
        plt.text(summary.loc[best_idx, "step"], summary.loc[best_idx, "rollout_score"], " best", va="bottom")
        plt.xlabel("step")
        plt.ylabel("gap score (%)")
        plt.title("Fixed Panel Rollout Gap by Step")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "rollout_score_trend.png", dpi=220)
        plt.close()

    if not per_scale.empty:
        plt.figure(figsize=(11, 6))
        for scale in LABELS:
            sub = per_scale[per_scale["scale"] == scale]
            plt.plot(sub["step"], sub["gap"], marker="o", linewidth=1.4, label=scale)
        plt.xlabel("step")
        plt.ylabel("gap (%)")
        plt.title("Per-Scale Rollout Gap by Step")
        plt.grid(alpha=0.2)
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "rollout_per_scale_trend.png", dpi=220)
        plt.close()
    return summary, per_scale


def module_stats(module, step, mat):
    near, far, dists = pair_groups()
    sims = [mat[i, j] for i, j, _ in dists]
    dist_values = [d for _, _, d in dists]
    return {
        "module": module,
        "step": step,
        "offdiag_mean": float(np.mean(offdiag_values(mat))),
        "near_mean": mean_pairs(mat, near),
        "far_mean": mean_pairs(mat, far),
        "near_minus_far": mean_pairs(mat, near) - mean_pairs(mat, far),
        "spearman_distance_similarity": rank_corr(dist_values, sims),
        "max_offdiag": float(np.max(offdiag_values(mat))),
        "min_offdiag": float(np.min(offdiag_values(mat))),
    }


def plot_heatmap_grid(mats_by_step, module, path):
    steps = list(mats_by_step)
    cols = 4
    rows = int(math.ceil(len(steps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.9), squeeze=False)
    vmax = max(0.25, max(float(mat.max()) for mat in mats_by_step.values()))
    for ax, step in zip(axes.flat, steps):
        sns.heatmap(
            mats_by_step[step], ax=ax, xticklabels=LABELS, yticklabels=LABELS,
            vmin=0, vmax=vmax, cmap="YlGnBu", square=True, cbar=False,
        )
        ax.set_title(f"step {step}")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=6)
    for ax in axes.flat[len(steps):]:
        ax.axis("off")
    fig.suptitle(f"{module} Closed-Mask Jaccard Heatmaps")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_pca_trajectory(vectors_by_step, module, path, coords_csv):
    steps = list(vectors_by_step)
    combined = np.concatenate([vectors_by_step[s].astype(np.float64) for s in steps], axis=0)
    coords = PCA(n_components=2, random_state=62).fit_transform(combined)
    coords_by_step = {
        step: coords[i * len(LABELS):(i + 1) * len(LABELS)]
        for i, step in enumerate(steps)
    }
    rows = []
    plt.figure(figsize=(8.8, 6.8))
    for idx, label in enumerate(LABELS):
        xy = np.stack([coords_by_step[s][idx] for s in steps], axis=0)
        plt.plot(xy[:, 0], xy[:, 1], color=COLORS[label], linewidth=1.5, alpha=0.75)
        plt.scatter(xy[:, 0], xy[:, 1], color=COLORS[label], edgecolor="black", linewidth=0.45, s=45)
        plt.text(xy[-1, 0], xy[-1, 1], " " + label, fontsize=8, va="center")
        for step, point in zip(steps, xy):
            rows.append({"module": module, "scale": label, "step": step, "pc1": point[0], "pc2": point[1]})
    plt.title(f"{module} Closed-Mask PCA Trajectory")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    pd.DataFrame(rows).to_csv(coords_csv, index=False)


def analyze_masks(run_dir, requested_steps, out_dir):
    available = {}
    for step in requested_steps:
        ckpt = ckpt_for_step(run_dir, step)
        if ckpt is not None:
            available[step] = load_masks(ckpt)
    with open(out_dir / "mask_checkpoint_steps.json", "w") as f:
        json.dump({str(k): str(ckpt_for_step(run_dir, k)) for k in available}, f, indent=2)

    all_metric_rows = []
    for module in list(MODULE_KEYS) + ["overall"]:
        module_dir = out_dir / module
        module_dir.mkdir(exist_ok=True)
        mats = {}
        vectors = {}
        for step, data in available.items():
            vectors[step] = data[module]
            mat = jaccard_matrix(data[module])
            mats[step] = mat
            pd.DataFrame(mat, index=LABELS, columns=LABELS).to_csv(module_dir / f"step{step}_closed_jaccard.csv")
            all_metric_rows.append(module_stats(module, step, mat))
        if mats:
            plot_heatmap_grid(mats, module, module_dir / "jaccard_heatmaps_all_steps.png")
            plot_pca_trajectory(vectors, module, module_dir / "pca_trajectory_all_steps.png", module_dir / "pca_coords_all_steps.csv")

    metrics = pd.DataFrame(all_metric_rows)
    metrics.to_csv(out_dir / "mask_similarity_metrics_by_step.csv", index=False)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="300,600,900,1200,1500,1800,2100,2400")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_steps, metrics = parse_log(run_dir / "run.log")
    rollout_summary, per_scale = write_rollout_tables(rollout_steps, metrics, out_dir)
    mask_metrics = analyze_masks(run_dir, parse_steps(args.steps), out_dir)

    report = {
        "rollout_best_step": None if rollout_summary.empty else int(rollout_summary.loc[rollout_summary["rollout_score"].idxmin(), "step"]),
        "rollout_best_score": None if rollout_summary.empty else float(rollout_summary["rollout_score"].min()),
        "latest_rollout_step": None if rollout_summary.empty else int(rollout_summary["step"].max()),
        "latest_rollout_score": None if rollout_summary.empty else float(rollout_summary.sort_values("step").iloc[-1]["rollout_score"]),
    }
    with open(out_dir / "analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
