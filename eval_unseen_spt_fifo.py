#!/usr/bin/env python3
"""Evaluate deterministic SPT and FIFO dispatching rules on JSSP JSONL data.

The evaluator intentionally uses the same event-driven JSSPEnv and legal action
set as the Decision Transformer rollout.  The NeurIPS 2025 LLMCoSolver paper
names SPT/FIFO as baselines, but its official repository does not publish their
implementations.  We therefore record the exact rule semantics used here:

* SPT: minimum processing time of the next operation among legal jobs.
* FIFO: minimum queue-arrival time of the next operation among legal jobs;
  queue-arrival is zero for a first operation and the previous operation's end
  time otherwise.
* Ties: smallest zero-based job id.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCALES = ("8x5", "10x8", "12x10", "15x12", "18x15", "20x18")
RULES = ("spt", "fifo")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def step2700_instance_hash(instance: np.ndarray) -> str:
    """Match the MD5-based identifier written by dt_infer.py."""
    contiguous = np.ascontiguousarray(instance, dtype=np.int32)
    return hashlib.md5(contiguous.tobytes()).hexdigest()[:10]


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_reference_rows(results_root: Path, scale: str) -> dict[str, dict[str, Any]]:
    candidates = sorted((results_root / scale).glob("jssp_evaluation_results_*.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one reference result JSON for {scale}, found {len(candidates)}"
        )
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    rows = payload.get("instances", [])
    by_name = {str(row["instance_name"]): row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError(f"Duplicate instance names in {candidates[0]}")
    return by_name


def choose_action(env: Any, rule: str) -> int:
    legal = [int(job) for job in env.action_space]
    if not legal:
        raise ValueError("choose_action called with an empty legal action set")

    if rule == "spt":
        def spt_key(job: int) -> tuple[int, int]:
            op = int(env.job_status[job])
            processing_time = int(env.current_instance[job, 2 * op + 1])
            return processing_time, job

        return min(legal, key=spt_key)

    if rule == "fifo":
        def fifo_key(job: int) -> tuple[int, int]:
            op = int(env.job_status[job])
            queue_arrival = 0 if op == 0 else int(env.schedule[job, op - 1, 1])
            return queue_arrival, job

        return min(legal, key=fifo_key)

    raise ValueError(f"Unknown dispatching rule: {rule}")


def validate_schedule(env: Any) -> None:
    n_jobs, n_machines = int(env.n_jobs), int(env.n_machines)
    if int(env.completed_operations) != n_jobs * n_machines:
        raise AssertionError(
            f"Incomplete schedule: {env.completed_operations}/{n_jobs * n_machines} operations"
        )

    for job in range(n_jobs):
        for op in range(n_machines):
            start, end = map(int, env.schedule[job, op])
            processing_time = int(env.current_instance[job, 2 * op + 1])
            if end - start != processing_time:
                raise AssertionError(
                    f"Duration mismatch for job={job}, op={op}: {start}, {end}, {processing_time}"
                )
            if op > 0 and int(env.schedule[job, op - 1, 1]) > start:
                raise AssertionError(f"Job precedence violation for job={job}, op={op}")

    machine_intervals: dict[int, list[tuple[int, int, int, int]]] = {
        machine: [] for machine in range(n_machines)
    }
    for job in range(n_jobs):
        for op in range(n_machines):
            machine = int(env.current_instance[job, 2 * op])
            start, end = map(int, env.schedule[job, op])
            machine_intervals[machine].append((start, end, job, op))
    for machine, intervals in machine_intervals.items():
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            if left[1] > right[0]:
                raise AssertionError(
                    f"Machine overlap on machine={machine}: {left} then {right}"
                )


def run_rule(instance: np.ndarray, rule: str, seed: int, jssp_env_cls: Any) -> dict[str, Any]:
    n_jobs, two_machines = instance.shape
    n_machines = two_machines // 2
    env = jssp_env_cls(
        n_job_range=[n_jobs, n_jobs],
        n_mac_range=[n_machines, n_machines],
        seed=seed,
        external_instance=instance.copy(),
        n_jobs=n_jobs,
        n_machines=n_machines,
    )
    env.reset()
    env.max_steps = n_jobs * n_machines + 1

    actions: list[int] = []
    decision_trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    guard = 0
    while not env.done:
        guard += 1
        if guard > 4 * n_jobs * n_machines + 100:
            raise RuntimeError("Environment failed to terminate")
        if not env.action_space:
            _, done, info = env.step()
            if info.get("invalid"):
                raise RuntimeError(f"Invalid time-advance transition: {info}")
            if done:
                break
            continue

        action = choose_action(env, rule)
        op = int(env.job_status[action])
        machine = int(env.current_instance[action, 2 * op])
        processing_time = int(env.current_instance[action, 2 * op + 1])
        queue_arrival = 0 if op == 0 else int(env.schedule[action, op - 1, 1])
        decision_trace.append(
            {
                "time": int(env.current_time),
                "job": action,
                "operation": op,
                "machine": machine,
                "processing_time": processing_time,
                "queue_arrival_time": queue_arrival,
            }
        )
        _, _, info = env.step(action=action)
        if info.get("invalid"):
            raise RuntimeError(f"Rule {rule} produced invalid action {action}: {info}")
        actions.append(action)

    elapsed = time.perf_counter() - started
    validate_schedule(env)
    if len(actions) != n_jobs * n_machines:
        raise AssertionError(f"Expected {n_jobs * n_machines} actions, got {len(actions)}")

    makespan = int(env.get_makespan())
    schedule = [
        [
            {
                "operation": op,
                "machine": int(instance[job, 2 * op]),
                "start": int(env.schedule[job, op, 0]),
                "end": int(env.schedule[job, op, 1]),
            }
            for op in range(n_machines)
        ]
        for job in range(n_jobs)
    ]
    return {
        "makespan": makespan,
        "inference_time_sec": elapsed,
        "actions": actions,
        "decision_trace": decision_trace,
        "schedule_by_job": schedule,
    }


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [float(row["relative_gap"]) for row in rows]
    makespans = [float(row["heuristic_makespan"]) for row in rows]
    runtimes = [float(row["heuristic_inference_time_sec"]) for row in rows]
    deltas = [float(row["heuristic_makespan"] - row["model_makespan"]) for row in rows]
    return {
        "instances": len(rows),
        "mean_makespan": mean(makespans),
        "mean_gap_percent": mean(gaps),
        "median_gap_percent": statistics.median(gaps) if gaps else None,
        "population_std_gap_percent": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
        "mean_inference_time_sec": mean(runtimes),
        "total_inference_time_sec": sum(runtimes),
        "wins_vs_step2700_greedy": sum(delta < 0 for delta in deltas),
        "ties_vs_step2700_greedy": sum(delta == 0 for delta in deltas),
        "losses_vs_step2700_greedy": sum(delta > 0 for delta in deltas),
        "mean_makespan_delta_vs_step2700_greedy": mean(deltas),
    }


def write_rule_outputs(output_root: Path, rule: str, rows: list[dict[str, Any]]) -> None:
    rule_root = output_root / rule
    rule_root.mkdir(parents=True, exist_ok=True)
    scale_summaries: dict[str, Any] = {}
    for scale in SCALES:
        scale_rows = [row for row in rows if row["scale"] == scale]
        scale_payload = {
            "rule": rule.upper(),
            "scale": scale,
            "summary": summarize(scale_rows),
            "instances": scale_rows,
        }
        (rule_root / f"results_{rule}_{scale}.json").write_text(
            json.dumps(scale_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        scale_summaries[scale] = scale_payload["summary"]

    overall = summarize(rows)
    overall["macro_mean_gap_percent"] = mean(
        float(scale_summaries[scale]["mean_gap_percent"]) for scale in SCALES
    )
    summary_payload = {
        "rule": rule.upper(),
        "definition": (
            "minimum next-operation processing time among legal jobs; ties by job id"
            if rule == "spt"
            else "earliest next-operation queue arrival among legal jobs; ties by job id"
        ),
        "per_scale": scale_summaries,
        "overall": overall,
    }
    (rule_root / f"summary_{rule}.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "rule", "scale", "instance_name", "instance_hash", "dataset_instance_hash", "difficulty",
        "heuristic_makespan", "model_makespan", "optimal_makespan", "relative_gap",
        "heuristic_inference_time_sec", "reference_source",
    ]
    with (rule_root / f"instances_{rule}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})


def run_self_test(jssp_env_cls: Any) -> None:
    instance = np.array(
        [
            [0, 3, 1, 2],
            [0, 1, 1, 4],
            [1, 2, 0, 2],
        ],
        dtype=np.int32,
    )
    spt_a = run_rule(instance, "spt", 1, jssp_env_cls)
    spt_b = run_rule(instance, "spt", 999, jssp_env_cls)
    fifo_a = run_rule(instance, "fifo", 1, jssp_env_cls)
    fifo_b = run_rule(instance, "fifo", 999, jssp_env_cls)
    assert spt_a["actions"] == spt_b["actions"]
    assert spt_a["makespan"] == spt_b["makespan"]
    assert fifo_a["actions"] == fifo_b["actions"]
    assert fifo_a["makespan"] == fifo_b["makespan"]
    assert spt_a["actions"][0] == 1, spt_a["actions"]
    assert fifo_a["actions"][0] == 0, fifo_a["actions"]
    print(
        "SELF_TEST=PASS",
        f"SPT_makespan={spt_a['makespan']}",
        f"FIFO_makespan={fifo_a['makespan']}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/path/to/local_workspace"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/local_workspace/generalization_unseen_6scales_n100_v1"),
    )
    parser.add_argument(
        "--reference-results-root",
        type=Path,
        default=Path(
            "/path/to/local_workspace/eval_step2700_phase1_fixed_legacy_greedy_unseen_6scales_n100_allone_v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/path/to/local_workspace/eval_heuristics_spt_fifo_unseen_6scales_n100_v1"),
    )
    parser.add_argument("--limit-per-scale", type=int, default=None)
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.root))
    from Envs.JSSPEnv.JSSPEnv import JSSPEnv

    run_self_test(JSSPEnv)
    if args.self_test_only:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "RUN_COMPLETE").exists():
        print(f"Already complete: {args.output_root}")
        return

    manifest_path = args.data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_per_scale = int(manifest["n_per_scale"])
    if args.limit_per_scale is None and expected_per_scale != 100:
        raise ValueError(f"Expected 100 instances per scale, manifest says {expected_per_scale}")

    all_rows: dict[str, list[dict[str, Any]]] = {rule: [] for rule in RULES}
    status_path = args.output_root / "status.tsv"
    if not status_path.exists():
        status_path.write_text("timestamp\trule\tscale\tstatus\tdetail\n", encoding="utf-8")

    def record(rule: str, scale: str, status: str, detail: str = "") -> None:
        timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp}\t{rule}\t{scale}\t{status}\t{detail}\n")
        print(timestamp, rule, scale, status, detail, flush=True)

    for scale in SCALES:
        input_path = args.data_root / "jsonl" / f"jssp_unseen_random_{scale}_n100_seed20260717.jsonl"
        instances = load_jsonl(input_path, args.limit_per_scale)
        expected = args.limit_per_scale or expected_per_scale
        if len(instances) != expected:
            raise ValueError(f"{scale}: expected {expected} instances, got {len(instances)}")
        references = load_reference_rows(args.reference_results_root, scale)

        for rule in RULES:
            record(rule, scale, "STARTED", f"instances={len(instances)}")
            scale_rows: list[dict[str, Any]] = []
            for index, item in enumerate(instances):
                name = str(item["name"])
                reference = references.get(name)
                if reference is None:
                    raise KeyError(f"{scale}/{name}: missing model/reference result")
                instance = np.asarray(item["current_instance"], dtype=np.int32)
                n_jobs, n_machines = int(item["n_jobs"]), int(item["n_machines"])
                if instance.shape != (n_jobs, 2 * n_machines):
                    raise ValueError(f"{scale}/{name}: invalid instance shape {instance.shape}")
                model_hash = step2700_instance_hash(instance)
                if model_hash != str(reference["instance_hash"]):
                    raise ValueError(
                        f"{scale}/{name}: step2700 MD5 mismatch "
                        f"({model_hash} != {reference['instance_hash']})"
                    )

                result = run_rule(instance, rule, int(item["seed"]), JSSPEnv)
                optimal = float(reference["optimal_makespan"])
                makespan = int(result["makespan"])
                relative_gap = (makespan - optimal) / optimal * 100.0
                row = {
                    "rule": rule.upper(),
                    "scale": scale,
                    "instance_index": index,
                    "instance_name": name,
                    "instance_hash": model_hash,
                    "dataset_instance_hash": str(item["instance_hash"]),
                    "difficulty": str(item["difficulty"]),
                    "heuristic_makespan": makespan,
                    "model_makespan": float(reference["model_makespan"]),
                    "optimal_makespan": optimal,
                    "relative_gap": relative_gap,
                    "heuristic_inference_time_sec": float(result["inference_time_sec"]),
                    "reference_source": str(reference.get("reference_source", "unknown")),
                    "actions": result["actions"],
                    "decision_trace": result["decision_trace"],
                    "schedule_by_job": result["schedule_by_job"],
                }
                scale_rows.append(row)
                if (index + 1) % 20 == 0 or index + 1 == len(instances):
                    print(
                        f"{rule.upper()} {scale}: {index + 1}/{len(instances)} "
                        f"mean_gap={mean(float(r['relative_gap']) for r in scale_rows):.4f}%",
                        flush=True,
                    )
            all_rows[rule].extend(scale_rows)
            record(rule, scale, "COMPLETED", f"instances={len(scale_rows)}")

    for rule in RULES:
        write_rule_outputs(args.output_root, rule, all_rows[rule])

    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "hostname": platform.node(),
        "dataset_root": str(args.data_root),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "reference_results_root": str(args.reference_results_root),
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "scales": list(SCALES),
        "instances_per_scale": args.limit_per_scale or expected_per_scale,
        "rules": {
            "SPT": "minimum current next-operation processing time among env.action_space; ties by job id",
            "FIFO": (
                "minimum queue-arrival time among env.action_space, where queue arrival is 0 for op0 "
                "and previous-operation completion time otherwise; ties by job id"
            ),
        },
        "environment_semantics": (
            "same event-driven Envs.JSSPEnv.JSSPEnv and legal action set as step2700 rollout"
        ),
        "literature_note": (
            "NeurIPS 2025 LLMCoSolver names SPT/FIFO and reports literature values but its official "
            "repository does not publish SPT/FIFO code; this evaluation records its exact semantics."
        ),
        "source_urls": [
            "https://github.com/Summer142857/LLMCoSolver",
            "https://github.com/zcaicaros/L2D",
        ],
    }
    (args.output_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    comparison = {
        rule: json.loads(
            (args.output_root / rule / f"summary_{rule}.json").read_text(encoding="utf-8")
        )["overall"]
        for rule in RULES
    }
    (args.output_root / "comparison_summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "RUN_COMPLETE").touch()
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"RUN_COMPLETE={args.output_root / 'RUN_COMPLETE'}")


if __name__ == "__main__":
    main()
