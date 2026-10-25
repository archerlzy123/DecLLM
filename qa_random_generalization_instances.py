#!/usr/bin/env python3
"""Read-only QA for the generated multi-scale JSSP benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


SCALES = [(10, 5), (10, 10), (15, 10), (15, 15), (20, 10), (20, 15), (20, 20)]
EXPECTED_DIFFICULTY = {"easy": 10, "medium": 25, "hard": 15}
TIME_RANGES = {"easy": (1, 5), "medium": (3, 8), "hard": (6, 15)}


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception as exc:
                    raise AssertionError(f"Invalid JSON: {path}:{line_no}: {exc}") from exc
    return records


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_record(record, n_jobs, n_machines):
    assert int(record["n_jobs"]) == n_jobs
    assert int(record["n_machines"]) == n_machines
    difficulty = record["difficulty"]
    assert difficulty in TIME_RANGES
    current = np.asarray(record["current_instance"], dtype=np.int32)
    assert current.shape == (n_jobs, 2 * n_machines), (record["name"], current.shape)
    machines = current[:, 0::2]
    times = current[:, 1::2]
    expected = np.arange(n_machines, dtype=np.int32)
    for row in machines:
        assert np.array_equal(np.sort(row), expected), (record["name"], row.tolist())
    low, high = TIME_RANGES[difficulty]
    assert int(times.min()) >= low and int(times.max()) <= high
    expected_hash = hashlib.sha1(np.ascontiguousarray(current, dtype=np.int32).tobytes()).hexdigest()[:10]
    assert record["instance_hash"] == expected_hash
    return current, machines, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/path/to/local_workspace/generalization_random_50_per_scale_v1")
    args = parser.parse_args()
    root = Path(args.root)
    assert root.is_dir(), root
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert int(manifest["total_instances"]) == 350
    assert int(manifest["n_per_scale"]) == 50

    all_names, all_hashes, all_seeds = [], [], []
    concatenated = []
    summary = []
    for n_jobs, n_machines in SCALES:
        scale = f"{n_jobs}x{n_machines}"
        jsonl = root / "jsonl" / f"jssp_random_{scale}_n50_seed20260716.jsonl"
        npz_path = root / "npz" / f"jssp_random_{scale}_n50_seed20260716.npz"
        txt_dir = root / "txt" / scale
        assert jsonl.is_file() and npz_path.is_file() and txt_dir.is_dir()
        records = load_jsonl(jsonl)
        assert len(records) == 50
        assert Counter(r["difficulty"] for r in records) == Counter(EXPECTED_DIFFICULTY)
        archive = np.load(npz_path, allow_pickle=False)
        assert set(archive.files) == {
            "processing_times", "machine_sequence_zero_based", "machines", "times",
            "names", "difficulties", "instance_seeds",
        }
        assert archive["processing_times"].shape == (50, n_jobs, n_machines)
        assert archive["machine_sequence_zero_based"].shape == (50, n_jobs, n_machines)
        assert np.array_equal(archive["processing_times"], archive["times"])
        assert np.array_equal(archive["machines"], archive["machine_sequence_zero_based"] + 1)
        txt_files = sorted(txt_dir.glob("*.txt"))
        assert len(txt_files) == 50

        for idx, record in enumerate(records):
            current, machines, times = parse_record(record, n_jobs, n_machines)
            assert str(archive["names"][idx]) == record["name"]
            assert str(archive["difficulties"][idx]) == record["difficulty"]
            assert int(archive["instance_seeds"][idx]) == int(record["seed"])
            assert np.array_equal(archive["processing_times"][idx], times)
            assert np.array_equal(archive["machine_sequence_zero_based"][idx], machines)
            txt_path = txt_dir / f"{record['name']}.txt"
            lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
            assert lines[0].strip() == f"{n_jobs} {n_machines}"
            txt_current = np.asarray([[int(x) for x in line.split()] for line in lines[1:]], dtype=np.int32)
            assert np.array_equal(txt_current, current)
            all_names.append(record["name"])
            all_hashes.append(record["instance_hash"])
            all_seeds.append(int(record["seed"]))
        concatenated.extend(records)
        summary.append((scale, len(records), dict(Counter(r["difficulty"] for r in records)), sha256(jsonl)[:16]))

    assert len(all_names) == len(set(all_names)) == 350
    assert len(all_hashes) == len(set(all_hashes)) == 350
    assert len(all_seeds) == len(set(all_seeds)) == 350
    combined = load_jsonl(root / "jsonl" / "jssp_random_all7scales_n350_seed20260716.jsonl")
    assert combined == concatenated
    assert manifest["combined_jsonl"] == "jsonl/jssp_random_all7scales_n350_seed20260716.jsonl"
    for scale, info in manifest["scales"].items():
        assert (root / info["jsonl"]).is_file()
        assert (root / info["npz"]).is_file()
        assert (root / info["text_dir"]).is_dir()

    print("QA_STATUS=PASS")
    print(f"root={root}")
    print("total=350 unique_names=350 unique_hashes=350 unique_seeds=350")
    print("formats=jsonl+npz+txt cross_checked")
    for scale, count, difficulties, digest in summary:
        print(f"{scale}: count={count}, difficulty={difficulties}, jsonl_sha256={digest}...")


if __name__ == "__main__":
    main()
