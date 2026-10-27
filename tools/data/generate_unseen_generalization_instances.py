#!/usr/bin/env python3
"""Generate the fixed six-scale unseen-size JSSP benchmark.

This intentionally reuses the environment-compatible sampling primitives from
generate_random_generalization_instances.py.  Each scale contains 100 unique
instances with a 20/50/30 easy/medium/hard split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.data.generate_random_generalization_instances import (
    DIFFICULTY,
    generate_instance,
    instance_hash,
    make_difficulty_schedule,
    scale_seed,
    validate_scale,
    write_text_instance,
)


SCALES = [(8, 5), (10, 8), (12, 10), (15, 12), (18, 15), (20, 18)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/path/to/local_workspace/generalization_unseen_6scales_n100_v1",
    )
    parser.add_argument("--n_per_scale", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_per_scale != 100:
        raise ValueError("This benchmark is fixed at exactly 100 instances per scale")

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {out}. Pass --overwrite to replace generated files."
        )
    out.mkdir(parents=True, exist_ok=True)
    jsonl_dir, npz_dir, txt_dir = out / "jsonl", out / "npz", out / "txt"
    for directory in (jsonl_dir, npz_dir, txt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    combined_records = []
    manifest_scales = {}
    for n_jobs, n_machines in SCALES:
        s_seed = scale_seed(args.seed, n_jobs, n_machines)
        rng = np.random.default_rng(s_seed)
        schedule = make_difficulty_schedule(rng, args.n_per_scale)
        scale = f"{n_jobs}x{n_machines}"
        scale_txt_dir = txt_dir / scale
        scale_txt_dir.mkdir(parents=True, exist_ok=True)
        records, machines_list, times_list = [], [], []

        for index, difficulty in enumerate(schedule):
            instance_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            instance_rng = np.random.default_rng(instance_seed)
            machines, times, interleaved = generate_instance(
                instance_rng, n_jobs, n_machines, difficulty
            )
            name = f"unseen_{scale}_{index:03d}"
            record = {
                "name": name,
                "seed": instance_seed,
                "master_seed": int(args.seed),
                "scale_seed": s_seed,
                "n_jobs": n_jobs,
                "n_machines": n_machines,
                "difficulty": difficulty,
                "source": "JSSPEnv-compatible random generator",
                "generator_version": "unseen_generalization_random_v1",
                "split": "unseen_scale_test",
                "instance_hash": instance_hash(interleaved),
                "current_instance": interleaved.tolist(),
            }
            records.append(record)
            combined_records.append(record)
            machines_list.append(machines)
            times_list.append(times)
            write_text_instance(
                scale_txt_dir / f"{name}.txt", interleaved, n_jobs, n_machines
            )

        machines_all = np.stack(machines_list)
        times_all = np.stack(times_list)
        validate_scale(records, machines_all, times_all, n_jobs, n_machines)
        jsonl_path = jsonl_dir / (
            f"jssp_unseen_random_{scale}_n{args.n_per_scale}_seed{args.seed}.jsonl"
        )
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

        npz_path = npz_dir / (
            f"jssp_unseen_random_{scale}_n{args.n_per_scale}_seed{args.seed}.npz"
        )
        np.savez_compressed(
            npz_path,
            processing_times=times_all,
            machine_sequence_zero_based=machines_all,
            machines=machines_all + 1,
            times=times_all,
            names=np.asarray([r["name"] for r in records]),
            difficulties=np.asarray([r["difficulty"] for r in records]),
            instance_seeds=np.asarray([r["seed"] for r in records], dtype=np.int64),
        )
        counts = {d: sum(r["difficulty"] == d for r in records) for d in DIFFICULTY}
        manifest_scales[scale] = {
            "count": len(records),
            "difficulty_counts": counts,
            "jsonl": str(jsonl_path.relative_to(out)),
            "npz": str(npz_path.relative_to(out)),
            "text_dir": str(scale_txt_dir.relative_to(out)),
        }
        print(f"[OK] {scale}: n={len(records)}, difficulty={counts}")

    combined_path = jsonl_dir / (
        f"jssp_unseen_random_all6scales_n{len(combined_records)}_seed{args.seed}.jsonl"
    )
    with combined_path.open("w", encoding="utf-8") as f:
        for record in combined_records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "generator_version": "unseen_generalization_random_v1",
        "purpose": "zero-shot generalization to six held-out JSSP scales",
        "master_seed": int(args.seed),
        "n_per_scale": int(args.n_per_scale),
        "total_instances": len(combined_records),
        "scales": manifest_scales,
        "difficulty_definition": {
            name: {
                "count_per_scale": manifest_scales[next(iter(manifest_scales))]["difficulty_counts"][name],
                "time_low": cfg["time_low"],
                "time_high": cfg["time_high"],
                "shuffle_probability": cfg["shuffle_probability"],
            }
            for name, cfg in DIFFICULTY.items()
        },
        "formats": {
            "jsonl": "current_instance uses zero-based [machine, processing_time] pairs",
            "npz": "processing_times/times plus zero-based machine_sequence and one-based machines",
            "txt": "first line n_jobs n_machines; remaining rows are zero-based machine/time pairs",
        },
        "combined_jsonl": str(combined_path.relative_to(out)),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] total={len(combined_records)} output={out}")


if __name__ == "__main__":
    main()
