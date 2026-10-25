import json
from pathlib import Path

ROOT = Path("/path/to/local_workspace")
NEW = ROOT / "eval_step2700_phase1_fixed_legacy_greedy_ta"
OLD = ROOT / "eval_step2700_scale_param_mask_greedy_la_ta" / "ta"
SCALES = ("15x15", "20x15", "20x20")


def load_instances(folder: Path, scale: str):
    path = folder / f"jssp_evaluation_results_ta_{scale}_instances.json"
    return json.loads(path.read_text())["instances"]


def instance_name(row):
    return row.get("instance_name", row.get("name"))


summary = {
    "new_dir": str(NEW),
    "old_dir": str(OLD),
    "comparison": "same step2700 checkpoint and greedy settings; only phase-1 tensorization differs",
    "scales": {},
}
all_new = []
all_old = []
all_instances = []

for scale in SCALES:
    new_rows = load_instances(NEW / "ta" / scale, scale)
    old_rows = load_instances(OLD / scale, scale)
    old_by_name = {instance_name(row): row for row in old_rows}
    new_gaps = []
    old_gaps = []
    wins = 0

    for new_row in new_rows:
        name = instance_name(new_row)
        old_row = old_by_name[name]
        new_gap = float(new_row["relative_gap"])
        old_gap = float(old_row["relative_gap"])
        new_gaps.append(new_gap)
        old_gaps.append(old_gap)
        wins += new_gap < old_gap
        all_instances.append(
            {
                "scale": scale,
                "instance": name,
                "old_broken_gap_percent": old_gap,
                "phase1_fixed_gap_percent": new_gap,
                "delta_fixed_minus_old_pp": new_gap - old_gap,
                "old_broken_model_makespan": old_row["model_makespan"],
                "phase1_fixed_model_makespan": new_row["model_makespan"],
            }
        )

    all_new.extend(new_gaps)
    all_old.extend(old_gaps)
    summary["scales"][scale] = {
        "instances": len(new_gaps),
        "phase1_fixed_mean_gap_percent": sum(new_gaps) / len(new_gaps),
        "old_broken_mean_gap_percent": sum(old_gaps) / len(old_gaps),
        "delta_fixed_minus_old_pp": sum(new_gaps) / len(new_gaps) - sum(old_gaps) / len(old_gaps),
        "fixed_wins": wins,
        "old_wins_or_ties": len(new_gaps) - wins,
    }

summary["overall"] = {
    "instances": len(all_new),
    "phase1_fixed_mean_gap_percent": sum(all_new) / len(all_new),
    "old_broken_mean_gap_percent": sum(all_old) / len(all_old),
    "delta_fixed_minus_old_pp": sum(all_new) / len(all_new) - sum(all_old) / len(all_old),
    "fixed_wins": sum(row["phase1_fixed_gap_percent"] < row["old_broken_gap_percent"] for row in all_instances),
}
summary["instances"] = all_instances

path = NEW / "summary_vs_old.json"
path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
