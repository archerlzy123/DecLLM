#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/path/to/local_workspace")
OUT = ROOT / "ssnc_vs_step2700_comprehensive_comparison_20260727"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_NEW = ROOT / "eval_ssnc_epoch12_canonical_v2_greedy_random_50_per_scale_v1"
RANDOM_OLD = ROOT / "eval_step2700_phase1_fixed_legacy_greedy_random_50_per_scale_v1"
UNSEEN_NEW = ROOT / "eval_ssnc_epoch12_canonical_v2_greedy_unseen_6scales_n100_allone_v1"
UNSEEN_OLD = ROOT / "eval_step2700_phase1_fixed_legacy_greedy_unseen_6scales_n100_allone_v1"
PUBLIC_NEW = ROOT / "eval_ssnc_epoch12_canonical_v2_la_ta_greedy_bon8_top_p07_v1"

OLD_GREEDY_LA = ROOT / "eval_step2700_phase1_fixed_legacy_greedy_la"
OLD_GREEDY_TA = ROOT / "eval_step2700_phase1_fixed_legacy_greedy_ta"
OLD_BON = ROOT / "eval_step2700_phase1_fixed_legacy_bon8_top_p07_remaining"
OLD_BON_TA20X15 = ROOT / "eval_step2700_phase1_fixed_legacy_bon8_top_p07_ta_20x15"

RNG = np.random.default_rng(20260727)


def load_rows(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("instances", [])
    assert rows, path
    by_name = {str(row["instance_name"]): row for row in rows}
    assert len(by_name) == len(rows), path
    return by_name


def bootstrap_mean_ci(values, samples=40000):
    values = np.asarray(values, dtype=np.float64)
    means = []
    remaining = samples
    while remaining:
        batch = min(2000, remaining)
        indices = RNG.integers(0, len(values), size=(batch, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= batch
    means = np.concatenate(means)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_stats(pairs):
    gap_delta = np.asarray(
        [
            float(new["relative_gap"]) - float(old["relative_gap"])
            for new, old in pairs
        ],
        dtype=np.float64,
    )
    makespan_delta = np.asarray(
        [
            float(new["model_makespan"]) - float(old["model_makespan"])
            for new, old in pairs
        ],
        dtype=np.float64,
    )
    old_gap = np.asarray(
        [float(old["relative_gap"]) for _, old in pairs], dtype=np.float64
    )
    new_gap = np.asarray(
        [float(new["relative_gap"]) for new, _ in pairs], dtype=np.float64
    )
    lo, hi = bootstrap_mean_ci(gap_delta)
    return {
        "instances": len(pairs),
        "step2700_mean_gap_percent": float(old_gap.mean()),
        "ssnc_epoch12_mean_gap_percent": float(new_gap.mean()),
        "delta_percentage_points": float(gap_delta.mean()),
        "median_delta_percentage_points": float(np.median(gap_delta)),
        "bootstrap_95ci_low": lo,
        "bootstrap_95ci_high": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "ssnc_wins": int(np.sum(makespan_delta < 0)),
        "ties": int(np.sum(makespan_delta == 0)),
        "ssnc_losses": int(np.sum(makespan_delta > 0)),
        "mean_makespan_delta": float(makespan_delta.mean()),
    }


def align(new_path, old_path):
    new = load_rows(new_path)
    old = load_rows(old_path)
    assert new.keys() == old.keys(), (new_path, old_path)
    pairs = []
    for name in sorted(new):
        new_ref = float(new[name]["optimal_makespan"])
        old_ref = float(old[name]["optimal_makespan"])
        assert math.isclose(new_ref, old_ref, abs_tol=1e-9), (
            name,
            new_ref,
            old_ref,
        )
        pairs.append((new[name], old[name]))
    return pairs


def random_path(root, scale):
    return (
        root
        / scale
        / f"jssp_evaluation_results_jssp_random_{scale}_n50_seed20260716.json"
    )


def unseen_path(root, scale):
    return (
        root
        / scale
        / f"jssp_evaluation_results_jssp_unseen_random_{scale}_n100_seed20260717.json"
    )


def public_new_path(mode, family, scale):
    return (
        PUBLIC_NEW
        / mode
        / family
        / scale
        / f"jssp_evaluation_results_{family}_{scale}_instances.json"
    )


def public_old_path(mode, family, scale):
    filename = f"jssp_evaluation_results_{family}_{scale}_instances.json"
    if mode == "greedy":
        root = OLD_GREEDY_LA if family == "la" else OLD_GREEDY_TA
        return root / family / scale / filename
    if family == "ta" and scale == "20x15":
        return OLD_BON_TA20X15 / filename
    return OLD_BON / family / scale / filename


report = {
    "interpretation": {
        "delta_definition": "SSNC epoch12 mean gap minus repaired step2700 mean gap",
        "better_direction": "negative delta means SSNC is better",
        "confidence_interval": "paired nonparametric bootstrap, 40000 resamples",
        "caution": (
            "Public BoN runs are stochastic, so a single paired run also contains "
            "decoding-sampling variance."
        ),
    },
    "suites": {},
}
flat_rows = []


def add_suite(suite_name, scale_pairs):
    all_pairs = []
    scales = {}
    for scale, pairs in scale_pairs.items():
        stats = paired_stats(pairs)
        scales[scale] = stats
        all_pairs.extend(pairs)
        flat_rows.append({"suite": suite_name, "scale": scale, **stats})
    overall = paired_stats(all_pairs)
    improved_scales = sum(row["delta_percentage_points"] < 0 for row in scales.values())
    worsened_scales = sum(row["delta_percentage_points"] > 0 for row in scales.values())
    report["suites"][suite_name] = {
        "scales": scales,
        "overall": overall,
        "scale_wins": improved_scales,
        "scale_losses": worsened_scales,
    }
    flat_rows.append({"suite": suite_name, "scale": "ALL", **overall})
    return all_pairs


random_scales = ["10x5", "10x10", "15x10", "15x15", "20x10", "20x15", "20x20"]
add_suite(
    "environment_random_greedy",
    {
        scale: align(random_path(RANDOM_NEW, scale), random_path(RANDOM_OLD, scale))
        for scale in random_scales
    },
)

unseen_scales = ["8x5", "10x8", "12x10", "15x12", "18x15", "20x18"]
add_suite(
    "unseen_scale_greedy_all_one_mask",
    {
        scale: align(unseen_path(UNSEEN_NEW, scale), unseen_path(UNSEEN_OLD, scale))
        for scale in unseen_scales
    },
)

la_scales = ["10x5", "15x5", "20x5", "10x10", "15x10", "20x10"]
ta_scales = ["15x15", "20x15", "20x20"]
public_pairs = {}
for mode in ["greedy", "bon8_top_p07"]:
    for family, scales in [("la", la_scales), ("ta", ta_scales)]:
        suite = f"public_{family}_{mode}"
        public_pairs[suite] = add_suite(
            suite,
            {
                scale: align(
                    public_new_path(mode, family, scale),
                    public_old_path(mode, family, scale),
                )
                for scale in scales
            },
        )
    add_suite(
        f"public_la_ta_{mode}",
        {
            "LA_ALL": public_pairs[f"public_la_{mode}"],
            "TA_ALL": public_pairs[f"public_ta_{mode}"],
        },
    )


def mean_gap(rows):
    return float(np.mean([float(row["relative_gap"]) for row in rows]))


search_gain = {}
for family, scales in [("la", la_scales), ("ta", ta_scales)]:
    old_greedy = []
    old_bon = []
    new_greedy = []
    new_bon = []
    for scale in scales:
        new_g = load_rows(public_new_path("greedy", family, scale))
        new_b = load_rows(public_new_path("bon8_top_p07", family, scale))
        old_g = load_rows(public_old_path("greedy", family, scale))
        old_b = load_rows(public_old_path("bon8_top_p07", family, scale))
        assert new_g.keys() == new_b.keys() == old_g.keys() == old_b.keys()
        for name in sorted(new_g):
            new_greedy.append(new_g[name])
            new_bon.append(new_b[name])
            old_greedy.append(old_g[name])
            old_bon.append(old_b[name])
    old_gain = mean_gap(old_greedy) - mean_gap(old_bon)
    new_gain = mean_gap(new_greedy) - mean_gap(new_bon)
    search_gain[family] = {
        "instances": len(new_greedy),
        "step2700_greedy_gap": mean_gap(old_greedy),
        "step2700_bon8_gap": mean_gap(old_bon),
        "step2700_bon_search_gain_pp": old_gain,
        "ssnc_greedy_gap": mean_gap(new_greedy),
        "ssnc_bon8_gap": mean_gap(new_bon),
        "ssnc_bon_search_gain_pp": new_gain,
        "ssnc_minus_step2700_search_gain_pp": new_gain - old_gain,
    }

report["public_greedy_to_bon8_search_gain"] = search_gain

(OUT / "comprehensive_comparison.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

fields = [
    "suite",
    "scale",
    "instances",
    "step2700_mean_gap_percent",
    "ssnc_epoch12_mean_gap_percent",
    "delta_percentage_points",
    "median_delta_percentage_points",
    "bootstrap_95ci_low",
    "bootstrap_95ci_high",
    "ci_excludes_zero",
    "ssnc_wins",
    "ties",
    "ssnc_losses",
    "mean_makespan_delta",
]
with (OUT / "comprehensive_comparison.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(flat_rows)

summary_lines = [
    "# SSNC epoch12 vs repaired step2700",
    "",
    "Negative delta means SSNC is better.",
    "",
    "| Suite | N | step2700 Gap | SSNC Gap | Delta (pp) | 95% CI | W/T/L |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
suite_order = [
    "environment_random_greedy",
    "unseen_scale_greedy_all_one_mask",
    "public_la_greedy",
    "public_ta_greedy",
    "public_la_ta_greedy",
    "public_la_bon8_top_p07",
    "public_ta_bon8_top_p07",
    "public_la_ta_bon8_top_p07",
]
for suite in suite_order:
    row = report["suites"][suite]["overall"]
    summary_lines.append(
        f"| {suite} | {row['instances']} | "
        f"{row['step2700_mean_gap_percent']:.4f}% | "
        f"{row['ssnc_epoch12_mean_gap_percent']:.4f}% | "
        f"{row['delta_percentage_points']:+.4f} | "
        f"[{row['bootstrap_95ci_low']:+.4f}, "
        f"{row['bootstrap_95ci_high']:+.4f}] | "
        f"{row['ssnc_wins']}/{row['ties']}/{row['ssnc_losses']} |"
    )
summary_lines.extend(["", "## Greedy to BoN=8 search gain", ""])
for family, row in search_gain.items():
    summary_lines.append(
        f"- {family.upper()}: step2700 gain={row['step2700_bon_search_gain_pp']:.4f} pp; "
        f"SSNC gain={row['ssnc_bon_search_gain_pp']:.4f} pp; "
        f"difference={row['ssnc_minus_step2700_search_gain_pp']:+.4f} pp."
    )
(OUT / "summary.md").write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

print("\n".join(summary_lines))
print(f"\nSaved to: {OUT}")
