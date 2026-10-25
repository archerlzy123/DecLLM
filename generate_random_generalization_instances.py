#!/usr/bin/env python3
"""Generate a fixed, reproducible multi-scale JSSP generalization benchmark.

The sampling distribution mirrors JSSPEnv._generate_single_instance:
  easy:   processing times [1, 5],  machine-order shuffle probability 0.2
  medium: processing times [3, 8],  machine-order shuffle probability 0.5
  hard:   processing times [6, 15], machine-order shuffle probability 0.8

Every scale contains exactly 50 instances: 10 easy, 25 medium, 15 hard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCALES = [(10, 5), (10, 10), (15, 10), (15, 15), (20, 10), (20, 15), (20, 20)]
DIFFICULTY = {
    "easy": {"count": 10, "time_low": 1, "time_high": 5, "shuffle_probability": 0.2},
    "medium": {"count": 25, "time_low": 3, "time_high": 8, "shuffle_probability": 0.5},
    "hard": {"count": 15, "time_low": 6, "time_high": 15, "shuffle_probability": 0.8},
}


def scale_seed(master_seed: int, n_jobs: int, n_machines: int) -> int:
    return int(master_seed + n_jobs * 1009 + n_machines * 9173)


def make_difficulty_schedule(rng: np.random.Generator, n_per_scale: int) -> list[str]:
    if n_per_scale == 50:
        schedule = [name for name, cfg in DIFFICULTY.items() for _ in range(cfg["count"])]
    else:
        raw = {"easy": 0.2 * n_per_scale, "medium": 0.5 * n_per_scale, "hard": 0.3 * n_per_scale}
        counts = {name: int(np.floor(v)) for name, v in raw.items()}
        remainder = n_per_scale - sum(counts.values())
        for name in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:remainder]:
            counts[name] += 1
        schedule = [name for name in ("easy", "medium", "hard") for _ in range(counts[name])]
    rng.shuffle(schedule)
    return schedule


def generate_instance(rng: np.random.Generator, n_jobs: int, n_machines: int, difficulty: str):
    cfg = DIFFICULTY[difficulty]
    machines = np.empty((n_jobs, n_machines), dtype=np.int32)
    times = np.empty((n_jobs, n_machines), dtype=np.int32)
    for job in range(n_jobs):
        order = np.arange(n_machines, dtype=np.int32)
        if float(rng.random()) < float(cfg["shuffle_probability"]):
            rng.shuffle(order)
        machines[job] = order
        times[job] = rng.integers(
            int(cfg["time_low"]),
            int(cfg["time_high"]) + 1,
            size=n_machines,
            dtype=np.int32,
        )
    interleaved = np.empty((n_jobs, n_machines * 2), dtype=np.int32)
    interleaved[:, 0::2] = machines
    interleaved[:, 1::2] = times
    return machines, times, interleaved


def instance_hash(interleaved: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(interleaved, dtype=np.int32).tobytes()).hexdigest()[:10]


def validate_scale(records: list[dict], machines_all: np.ndarray, times_all: np.ndarray, n_jobs: int, n_machines: int):
    assert len(records) == machines_all.shape[0] == times_all.shape[0]
    expected = np.arange(n_machines, dtype=np.int32)
    for idx, record in enumerate(records):
        assert record["n_jobs"] == n_jobs and record["n_machines"] == n_machines
        for row in machines_all[idx]:
            assert np.array_equal(np.sort(row), expected), (record["name"], row.tolist())
        cfg = DIFFICULTY[record["difficulty"]]
        assert int(times_all[idx].min()) >= int(cfg["time_low"])
        assert int(times_all[idx].max()) <= int(cfg["time_high"])
    hashes = [r["instance_hash"] for r in records]
    assert len(hashes) == len(set(hashes)), f"Duplicate instances found in {n_jobs}x{n_machines}"


def write_text_instance(path: Path, interleaved: np.ndarray, n_jobs: int, n_machines: int):
    lines = [f"{n_jobs} {n_machines}"]
    lines.extend(" ".join(str(int(x)) for x in row) for row in interleaved)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/path/to/local_workspace/generalization_random_50_per_scale_v1",
    )
    parser.add_argument("--n_per_scale", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}. Pass --overwrite to replace generated files.")
    out.mkdir(parents=True, exist_ok=True)
    jsonl_dir = out / "jsonl"
    npz_dir = out / "npz"
    txt_dir = out / "txt"
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
        records = []
        machines_list = []
        times_list = []
        for index, difficulty in enumerate(schedule):
            instance_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            instance_rng = np.random.default_rng(instance_seed)
            machines, times, interleaved = generate_instance(
                instance_rng, n_jobs, n_machines, difficulty
            )
            name = f"rnd_{scale}_{index:03d}"
            record = {
                "name": name,
                "seed": instance_seed,
                "master_seed": int(args.seed),
                "scale_seed": s_seed,
                "n_jobs": n_jobs,
                "n_machines": n_machines,
                "difficulty": difficulty,
                "source": "JSSPEnv-compatible random generator",
                "generator_version": "generalization_random_v1",
                "instance_hash": instance_hash(interleaved),
                "current_instance": interleaved.tolist(),
            }
            records.append(record)
            combined_records.append(record)
            machines_list.append(machines)
            times_list.append(times)
            write_text_instance(scale_txt_dir / f"{name}.txt", interleaved, n_jobs, n_machines)

        machines_all = np.stack(machines_list)
        times_all = np.stack(times_list)
        validate_scale(records, machines_all, times_all, n_jobs, n_machines)
        jsonl_path = jsonl_dir / f"jssp_random_{scale}_n{args.n_per_scale}_seed{args.seed}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        np.savez_compressed(
            npz_dir / f"jssp_random_{scale}_n{args.n_per_scale}_seed{args.seed}.npz",
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
            "npz": str((npz_dir / f"jssp_random_{scale}_n{args.n_per_scale}_seed{args.seed}.npz").relative_to(out)),
            "text_dir": str(scale_txt_dir.relative_to(out)),
        }
        print(f"[OK] {scale}: n={len(records)}, difficulty={counts}")

    combined_path = jsonl_dir / f"jssp_random_all7scales_n{len(combined_records)}_seed{args.seed}.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for record in combined_records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "generator_version": "generalization_random_v1",
        "master_seed": int(args.seed),
        "n_per_scale": int(args.n_per_scale),
        "total_instances": len(combined_records),
        "scales": manifest_scales,
        "difficulty_definition": DIFFICULTY,
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
