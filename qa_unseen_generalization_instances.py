#!/usr/bin/env python3
"""Strict read-only QA for the six-scale unseen JSSP benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


SCALES = [(8, 5), (10, 8), (12, 10), (15, 12), (18, 15), (20, 18)]
EXPECTED_DIFFICULTY = {"easy": 20, "medium": 50, "hard": 30}
TIME_RANGES = {"easy": (1, 5), "medium": (3, 8), "hard": (6, 15)}
SEED = 20260717


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="/path/to/local_workspace/generalization_unseen_6scales_n100_v1"
    )
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator_version"] == "unseen_generalization_random_v1"
    assert manifest["total_instances"] == 600
    assert manifest["n_per_scale"] == 100
    assert list(manifest["scales"].keys()) == [f"{n}x{m}" for n, m in SCALES]

    all_names, all_hashes, all_seeds, concatenated = [], [], [], []
    for n_jobs, n_machines in SCALES:
        scale = f"{n_jobs}x{n_machines}"
        jsonl = root / "jsonl" / f"jssp_unseen_random_{scale}_n100_seed{SEED}.jsonl"
        npz_path = root / "npz" / f"jssp_unseen_random_{scale}_n100_seed{SEED}.npz"
        txt_dir = root / "txt" / scale
        records = load_jsonl(jsonl)
        archive = np.load(npz_path, allow_pickle=False)
        assert len(records) == 100
        assert Counter(r["difficulty"] for r in records) == Counter(EXPECTED_DIFFICULTY)
        assert archive["processing_times"].shape == (100, n_jobs, n_machines)
        assert archive["machine_sequence_zero_based"].shape == (100, n_jobs, n_machines)
        assert np.array_equal(archive["processing_times"], archive["times"])
        assert np.array_equal(archive["machines"], archive["machine_sequence_zero_based"] + 1)
        assert len(list(txt_dir.glob("*.txt"))) == 100
        expected_machines = np.arange(n_machines, dtype=np.int32)

        for idx, record in enumerate(records):
            assert record["split"] == "unseen_scale_test"
            assert (record["n_jobs"], record["n_machines"]) == (n_jobs, n_machines)
            current = np.asarray(record["current_instance"], dtype=np.int32)
            assert current.shape == (n_jobs, 2 * n_machines)
            machines, times = current[:, 0::2], current[:, 1::2]
            for row in machines:
                assert np.array_equal(np.sort(row), expected_machines)
            low, high = TIME_RANGES[record["difficulty"]]
            assert low <= int(times.min()) <= int(times.max()) <= high
            expected_hash = hashlib.sha1(
                np.ascontiguousarray(current, dtype=np.int32).tobytes()
            ).hexdigest()[:10]
            assert record["instance_hash"] == expected_hash
            assert str(archive["names"][idx]) == record["name"]
            assert str(archive["difficulties"][idx]) == record["difficulty"]
            assert int(archive["instance_seeds"][idx]) == int(record["seed"])
            assert np.array_equal(archive["processing_times"][idx], times)
            assert np.array_equal(archive["machine_sequence_zero_based"][idx], machines)
            txt = np.asarray(
                [[int(x) for x in line.split()] for line in
                 (txt_dir / f"{record['name']}.txt").read_text(encoding="utf-8").strip().splitlines()[1:]],
                dtype=np.int32,
            )
            assert np.array_equal(txt, current)
            all_names.append(record["name"])
            all_hashes.append(record["instance_hash"])
            all_seeds.append(int(record["seed"]))
        concatenated.extend(records)
        print(
            f"{scale}: count=100 difficulty={dict(Counter(r['difficulty'] for r in records))} "
            f"jsonl_sha256={digest(jsonl)[:16]}..."
        )

    assert len(all_names) == len(set(all_names)) == 600
    assert len(all_hashes) == len(set(all_hashes)) == 600
    assert len(all_seeds) == len(set(all_seeds)) == 600
    combined = load_jsonl(
        root / "jsonl" / f"jssp_unseen_random_all6scales_n600_seed{SEED}.jsonl"
    )
    assert combined == concatenated
    print("QA_STATUS=PASS")
    print("total=600 unique_names=600 unique_hashes=600 unique_seeds=600")
    print("formats=jsonl+npz+txt cross_checked")


if __name__ == "__main__":
    main()
