import json
from pathlib import Path

ROOT = Path("/path/to/local_workspace")
BON = ROOT / "eval_step2700_phase1_fixed_legacy_bon8_la_ta"
GREEDY = {
    "la": ROOT / "eval_step2700_phase1_fixed_legacy_greedy_la" / "la",
    "ta": ROOT / "eval_step2700_phase1_fixed_legacy_greedy_ta" / "ta",
}
SCALES = {
    "la": ("10x5", "15x5", "20x5", "10x10", "15x10", "20x10"),
    "ta": ("15x15", "20x15", "20x20"),
}


def load_rows(folder: Path, family: str, scale: str):
    path = folder / f"jssp_evaluation_results_{family}_{scale}_instances.json"
    return json.loads(path.read_text())["instances"]


def name(row):
    return row.get("instance_name", row.get("name"))


summary = {
    "bon_dir": str(BON),
    "checkpoint": "best_gap_model_step_2700",
    "state_tensorization_mode": "legacy_train_compatible",
    "bon_settings": {
        "rollouts": 8,
        "hybrid": "rollout 1 greedy; rollouts 2-8 sampled",
        "temperature": 0.7,
        "top_k": 5,
        "pick": "best makespan",
    },
    "families": {},
}

for family, scales in SCALES.items():
    family_bon = []
    family_greedy = []
    scale_summaries = {}
    details = []
    for scale in scales:
        bon_rows = load_rows(BON / family / scale, family, scale)
        greedy_rows = load_rows(GREEDY[family] / scale, family, scale)
        greedy_by_name = {name(row): row for row in greedy_rows}
        bg = []
        gg = []
        wins = ties = 0
        for row in bon_rows:
            instance = name(row)
            greedy = greedy_by_name[instance]
            bon_gap = float(row["relative_gap"])
            greedy_gap = float(greedy["relative_gap"])
            bg.append(bon_gap)
            gg.append(greedy_gap)
            wins += bon_gap < greedy_gap
            ties += bon_gap == greedy_gap
            details.append({
                "scale": scale,
                "instance": instance,
                "greedy_gap_percent": greedy_gap,
                "bon8_gap_percent": bon_gap,
                "delta_bon8_minus_greedy_pp": bon_gap - greedy_gap,
                "greedy_makespan": greedy["model_makespan"],
                "bon8_makespan": row["model_makespan"],
            })
        family_bon.extend(bg)
        family_greedy.extend(gg)
        scale_summaries[scale] = {
            "instances": len(bg),
            "greedy_mean_gap_percent": sum(gg) / len(gg),
            "bon8_mean_gap_percent": sum(bg) / len(bg),
            "delta_bon8_minus_greedy_pp": sum(bg) / len(bg) - sum(gg) / len(gg),
            "bon8_wins": wins,
            "ties": ties,
            "greedy_wins": len(bg) - wins - ties,
        }
    summary["families"][family] = {
        "scales": scale_summaries,
        "overall": {
            "instances": len(family_bon),
            "greedy_mean_gap_percent": sum(family_greedy) / len(family_greedy),
            "bon8_mean_gap_percent": sum(family_bon) / len(family_bon),
            "delta_bon8_minus_greedy_pp": sum(family_bon) / len(family_bon) - sum(family_greedy) / len(family_greedy),
            "bon8_wins": sum(row["bon8_gap_percent"] < row["greedy_gap_percent"] for row in details),
            "ties": sum(row["bon8_gap_percent"] == row["greedy_gap_percent"] for row in details),
        },
        "instances": details,
    }

path = BON / "summary_vs_fixed_greedy.json"
path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
