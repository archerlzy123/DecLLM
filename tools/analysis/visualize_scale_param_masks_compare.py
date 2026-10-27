#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
        value = float(inter / union) if union else 1.0
        mat[i, j] = mat[j, i] = value
    return mat


def offdiag_values(mat):
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def scale_distance(i, j):
    a, b = SCALES[i], SCALES[j]
    return math.sqrt(((a[0] - b[0]) / 5.0) ** 2 + ((a[1] - b[1]) / 5.0) ** 2)


def pair_groups():
    near, far = [], []
    for i, j in itertools.combinations(range(len(SCALES)), 2):
        d = scale_distance(i, j)
        if d <= 1.01:
            near.append((i, j))
        if d >= 2.0:
            far.append((i, j))
    return near, far


def mean_pairs(mat, pairs):
    return float(np.mean([mat[i, j] for i, j in pairs])) if pairs else float("nan")


def rank_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def plot_heatmap(mat, title, path):
    plt.figure(figsize=(7.2, 6.2))
    sns.heatmap(
        mat, xticklabels=LABELS, yticklabels=LABELS,
        vmin=0.0, vmax=max(0.25, float(mat.max())),
        cmap="YlGnBu", square=True, cbar_kws={"label": "Jaccard"},
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def tsne_coords(x):
    mat = jaccard_matrix(x)
    dist = 1.0 - mat
    return TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        perplexity=3,
        learning_rate="auto",
        random_state=62,
    ).fit_transform(dist)


def plot_scatter(coords, title, path):
    plt.figure(figsize=(7.5, 5.7))
    for idx, label in enumerate(LABELS):
        plt.scatter(coords[idx, 0], coords[idx, 1], s=95, color=COLORS[label], edgecolor="black", linewidth=0.8)
        plt.text(coords[idx, 0], coords[idx, 1], " " + label, fontsize=8, va="center")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_module_grid(data, step_name, out_dir, kind):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    for ax, module in zip(axes, MODULE_KEYS):
        if kind == "tsne":
            coords = tsne_coords(data[module])
            ax.set_xticks([])
            ax.set_yticks([])
            for idx, label in enumerate(LABELS):
                ax.scatter(coords[idx, 0], coords[idx, 1], s=75, color=COLORS[label], edgecolor="black", linewidth=0.8)
                ax.text(coords[idx, 0], coords[idx, 1], " " + label, fontsize=7, va="center")
            ax.set_title(module)
        else:
            mat = jaccard_matrix(data[module])
            sns.heatmap(mat, ax=ax, xticklabels=LABELS, yticklabels=LABELS, vmin=0, vmax=max(0.25, float(mat.max())), cmap="YlGnBu", square=True, cbar=False)
            ax.set_title(module)
            ax.tick_params(axis="x", rotation=45, labelsize=6)
            ax.tick_params(axis="y", rotation=0, labelsize=6)
    fig.suptitle(f"{step_name} Closed-Mask {'t-SNE' if kind == 'tsne' else 'Jaccard Similarity'} by Module")
    plt.tight_layout()
    plt.savefig(out_dir / f"{step_name}_by_module_closed_mask_{kind}_grid.png", dpi=220)
    plt.close()


def pca_arrow_plot(x300, x600, title, path):
    combined = np.concatenate([x300.astype(np.float64), x600.astype(np.float64)], axis=0)
    coords = PCA(n_components=2, random_state=62).fit_transform(combined)
    c300, c600 = coords[:len(LABELS)], coords[len(LABELS):]
    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(LABELS):
        color = COLORS[label]
        plt.scatter(c300[idx, 0], c300[idx, 1], s=60, color=color, edgecolor="black", linewidth=0.7, alpha=0.65)
        plt.scatter(c600[idx, 0], c600[idx, 1], s=105, color=color, edgecolor="black", linewidth=1.0)
        plt.arrow(
            c300[idx, 0], c300[idx, 1],
            c600[idx, 0] - c300[idx, 0], c600[idx, 1] - c300[idx, 1],
            length_includes_head=True, head_width=0.018, alpha=0.65, color=color,
        )
        plt.text(c600[idx, 0], c600[idx, 1], " " + label, fontsize=8, va="center")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return c300, c600


def direction_stats(c300, c600):
    vectors = c600 - c300
    near, far = pair_groups()
    def cos(i, j):
        a, b = vectors[i], vectors[j]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else float("nan")
    all_pairs = list(itertools.combinations(range(len(LABELS)), 2))
    return {
        "mean_shift_norm": float(np.mean(np.linalg.norm(vectors, axis=1))),
        "near_direction_cosine_mean": float(np.nanmean([cos(i, j) for i, j in near])),
        "far_direction_cosine_mean": float(np.nanmean([cos(i, j) for i, j in far])),
        "all_direction_cosine_mean": float(np.nanmean([cos(i, j) for i, j in all_pairs])),
    }


def summarize_module(module, x300, x600, c300, c600):
    j300 = jaccard_matrix(x300)
    j600 = jaccard_matrix(x600)
    delta = j600 - j300
    near, far = pair_groups()
    dists = []
    sims300 = []
    sims600 = []
    deltas = []
    for i, j in itertools.combinations(range(len(LABELS)), 2):
        dists.append(scale_distance(i, j))
        sims300.append(j300[i, j])
        sims600.append(j600[i, j])
        deltas.append(delta[i, j])
    top_pos = sorted(
        [(LABELS[i], LABELS[j], float(delta[i, j])) for i, j in itertools.combinations(range(len(LABELS)), 2)],
        key=lambda z: z[2],
        reverse=True,
    )[:8]
    top_neg = sorted(
        [(LABELS[i], LABELS[j], float(delta[i, j])) for i, j in itertools.combinations(range(len(LABELS)), 2)],
        key=lambda z: z[2],
    )[:8]
    out = {
        "step300_offdiag_mean": float(np.mean(offdiag_values(j300))),
        "step600_offdiag_mean": float(np.mean(offdiag_values(j600))),
        "delta_offdiag_mean": float(np.mean(offdiag_values(delta))),
        "step300_near_mean": mean_pairs(j300, near),
        "step600_near_mean": mean_pairs(j600, near),
        "delta_near_mean": mean_pairs(delta, near),
        "step300_far_mean": mean_pairs(j300, far),
        "step600_far_mean": mean_pairs(j600, far),
        "delta_far_mean": mean_pairs(delta, far),
        "spearman_distance_similarity_step300": rank_corr(dists, sims300),
        "spearman_distance_similarity_step600": rank_corr(dists, sims600),
        "spearman_distance_delta": rank_corr(dists, deltas),
        "largest_similarity_increases": top_pos,
        "largest_similarity_decreases": top_neg,
    }
    out.update(direction_stats(c300, c600))
    return out, j300, j600, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step300", required=True)
    ap.add_argument("--step600", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data300 = load_masks(args.step300)
    data600 = load_masks(args.step600)

    summary = {}
    all_rows = []
    for module in list(MODULE_KEYS) + ["overall"]:
        module_dir = out_dir / module
        module_dir.mkdir(exist_ok=True)

        j300 = jaccard_matrix(data300[module])
        j600 = jaccard_matrix(data600[module])
        pd.DataFrame(j300, index=LABELS, columns=LABELS).to_csv(module_dir / "step300_closed_jaccard.csv")
        pd.DataFrame(j600, index=LABELS, columns=LABELS).to_csv(module_dir / "step600_closed_jaccard.csv")
        pd.DataFrame(j600 - j300, index=LABELS, columns=LABELS).to_csv(module_dir / "delta_step600_minus_step300.csv")

        plot_heatmap(j300, f"Step 300 {module} Closed-Mask Jaccard", module_dir / "step300_closed_jaccard_heatmap.png")
        plot_heatmap(j600, f"Step 600 {module} Closed-Mask Jaccard", module_dir / "step600_closed_jaccard_heatmap.png")
        plot_heatmap(j600 - j300, f"Delta {module} Jaccard Step600 - Step300", module_dir / "delta_closed_jaccard_heatmap.png")

        c300, c600 = pca_arrow_plot(
            data300[module], data600[module],
            f"{module} Closed-Mask PCA Shift: Step300 -> Step600",
            module_dir / "pca_shift_step300_to_step600.png",
        )
        plot_scatter(tsne_coords(data300[module]), f"Step 300 {module} Closed-Mask t-SNE", module_dir / "step300_closed_mask_tsne.png")
        plot_scatter(tsne_coords(data600[module]), f"Step 600 {module} Closed-Mask t-SNE", module_dir / "step600_closed_mask_tsne.png")

        stats, _, _, _ = summarize_module(module, data300[module], data600[module], c300, c600)
        summary[module] = stats
        row = {"module": module}
        row.update({k: v for k, v in stats.items() if not isinstance(v, list)})
        all_rows.append(row)

    plot_module_grid(data300, "step300", out_dir, "tsne")
    plot_module_grid(data600, "step600", out_dir, "tsne")
    plot_module_grid(data300, "step300", out_dir, "jaccard")
    plot_module_grid(data600, "step600", out_dir, "jaccard")

    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(all_rows).to_csv(out_dir / "comparison_metrics.csv", index=False)
    print(json.dumps(summary["overall"], indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
