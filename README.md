# DecLLM

> **DecLLM: Graph-Enhanced Decision-Making Large Language Model with Conflict-Aware Mask for Multi-Configuration Job Shop Scheduling Problems**
>
> by [Rongkai Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=l-zF-W0AAAAJ)<sup>1</sup>, Ziyi Lu<sup>1</sup>, Litao Zhang<sup>1</sup>, Yulong Zhang<sup>1</sup>, Zhezhuang Xu<sup>1</sup>, and [Shibo He](https://scholar.google.com/citations?hl=zh-CN&user=5GOcb4gAAAAJ&view_op=list_works&sortby=pubdate)<sup>2,*</sup>
>
> <sup>1</sup> School of Electrical Engineering and Automation, Fuzhou University, Fuzhou 350108, China.
>
> <sup>2</sup> State Key Laboratory of Industrial Control Technology, Zhejiang University, Hangzhou 310027, China.
>
> <sup>*</sup> Corresponding author: Shibo He.

DecLLM is a shared Decision Transformer scheduling policy for the job-shop scheduling problem (JSSP). This repository contains the cleaned SSNC-v1 implementation and instructions for using its best trained checkpoint.

The policy uses graph-based state encoding, a Qwen2.5-7B-Instruct backbone with LoRA, candidate-job scoring, configuration embeddings, bottleneck-based adaptation modules, and configuration-specific conflict-aware masks. The released model uses `canonical_v2` state tensorization and `normalized_budget` conditioning.

## Repository layout

```text
dt_train.py / dt_infer.py       Training and inference entrypoints
decision_transformer.py        Model and scheduling heads
dt_data_processing.py          JSSP trajectory processing
state_tensorization.py         Shared state tensorization
ssnc_mask.py                   Conflict-aware mask updates
ssnc_inference_routing.py      Mask routing at inference
dataset_optimization.py        Optional augmentation dependency
Envs/JSSPEnv/                  JSSP environment and solver
configs/                       Launcher configuration templates
scripts/                       Training/inference launcher and check runner
tests/                         Regression and smoke checks
tools/                         Optional data, evaluation and analysis tools
```

## Checkpoint

Download **DecLLM-ssnc-v1-best.zip** from the [SSNC-v1 release](https://github.com/archerlzy123/DecLLM/releases/tag/ssnc-v1), verify it against the accompanying `SHA256SUMS.txt`, and extract it under `checkpoints/`:

```text
checkpoints/ssnc-v1-best/
  adapter_config.json
  adapter_model.safetensors
  dt_custom_modules.pt
  checksums.json
  MODEL_CARD.md
  LICENSE_CODE_MIT.txt
  LICENSE_BASE_MODEL.txt
```

This is `best_gap_model_final_epoch_12` from the original SSNC-v1 training run, **not** the short cleanup-regression checkpoint. Its training log records a best validation-panel Gap of 7.3906%. This validation score is not a claim about every test benchmark. The ZIP contains trained LoRA and custom modules, not the complete Qwen base weights. The duplicated `adapter_model.bin` is omitted because the equivalent safetensors adapter is supplied.

Obtain [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) separately. Keep the base model, adapter, and custom modules together in the loading workflow; the adapter alone is not a standalone scheduler. The adapter configuration intentionally retains its original empty `base_model_name_or_path`; the launcher passes the base path explicitly.

Only load `.pt` checkpoints from trusted sources after checksum verification. This release preserves the existing PyTorch checkpoint format.

## Environment

The tested server environment uses Python 3.12.13, PyTorch 2.4.1+cu121, Transformers 4.57.3, PEFT 0.11.1, Accelerate 1.12.0, bitsandbytes 0.48.2, and NumPy 1.26.4. See `requirements.server-tested.txt` for the recorded versions. These are tested-environment records, not a clean-install-tested lockfile. The inherited `requirements.txt` is historical and contains older constraints; do not blindly apply it to an existing environment.

Use a Linux NVIDIA/CUDA environment for the real 7B training and inference commands. CPU tests do not load the base model. The convenience launcher checks for at least 30,000 MiB free GPU memory before running; this is a conservative smoke-test guard, not a measured minimum-memory claim or an exclusive GPU reservation.

## Configure local paths

Copy `configs/paths.example.json` to `configs/paths.json` and edit:

```json
{
  "python": "python",
  "base_model": "/absolute/path/to/Qwen2.5-7B-Instruct",
  "trajectories": "/absolute/path/to/training_trajectories.jsonl",
  "checkpoint": "/absolute/path/to/DecLLM/checkpoints/ssnc-v1-best",
  "rollout_panel": "/absolute/path/to/original_fixed_validation_panel.jsonl",
  "gpu": "0"
}
```

Local configuration is ignored by Git. A different file can be supplied with `--config PATH`. Base weights, full training trajectories, validation panels, internal backups and private server paths are not included. The original fixed validation-panel file must be supplied for exact full-training reproduction; no replacement panel is silently substituted.

## Inference

From the repository root:

```bash
python scripts/launch.py infer --gpu 0 --name greedy_10x5 --jobs 10 --machines 5
python scripts/launch.py infer --gpu 0 --name bon8_10x5 --jobs 10 --machines 5 --rollouts 8
```

The first command runs Greedy; the second uses eight top-p=0.7 samples. Each example evaluates one generated instance, not an entire paper benchmark. Outputs are written under `runs/NAME/`; existing output directories are never overwritten. The launcher uses the formal inference entrypoint with the five mask sites enabled and `all_one` routing for unseen configurations.

Run `python dt_infer.py --help` for the full evaluation interface. Optional data, evaluation, analysis and checkpoint helpers are grouped under `tools/`; see [tools/README.md](tools/README.md) for their scope and module commands. Helpers containing `/path/to/local_workspace` defaults require explicit local arguments.

## Training

```bash
python scripts/launch.py train --dry-run --name train_preview
python scripts/launch.py train --gpu 0 --name train_ssnc_v1
```

`configs/training_ssnc_v1.json` preserves the original 12-epoch command-line hyperparameters, with machine-specific paths supplied separately by configuration. Training starts from the base model, not from the released best adapter.

A two-update smoke mode is available:

```bash
python scripts/launch.py train --smoke --gpu 0 --name smoke_train
```

It requires a compatible small trajectory JSONL, provided either as `data/smoke_trajectories.jsonl` or through an additional `smoke_trajectories` field in the local configuration. No training-data subset is redistributed in this repository. Smoke mode shortens context to 16, uses gradient accumulation 1 and triggers all five mask-update paths early. It is a functional test, not a performance experiment.

## Tests and verification

```bash
python scripts/run_checks.py
python scripts/run_checks.py --checkpoint checkpoints/ssnc-v1-best --output test-results-with-checkpoint
```

Without a checkpoint, nine CPU checks run and the checkpoint-dependent neighbor-routing check is explicitly skipped. With a checkpoint, all ten run. Output directories must not already exist.

Tests live under `tests/`. Run an individual check from the repository root, for example `python -m tests.test_ssnc_mask` or `python -m tests.verify_canonical_v2_entrypoints`. Use the full runner above to include all script-based checks.

The cleanup regression compared the actual before/after code on the same server, GPU, seed and 88 trajectories, with two real 7B optimizer updates per side. All 24 observed loss calls, observed gradients and optimizer parameter updates matched bitwise. The 672 saved checkpoint tensors matched elementwise, and reloaded 10x5 Greedy trajectories matched. This regression used newly trained smoke checkpoints; the released asset is the separate original best checkpoint. See [VALIDATION.md](VALIDATION.md).

This does not establish full 12-epoch reproduction, all benchmark scores, arbitrary-device determinism, fresh-environment installation, or optimizer-resume compatibility. Existing rollout records include `-1` entries for clock advances; these are counted separately from job-selection actions. The production algorithm is unchanged by this packaging.

## License and provenance

DecLLM additions are released under the [MIT License](LICENSE). The project was developed from [LLMCoSolver](https://github.com/Summer142857/LLMCoSolver); its original MIT notice is retained in `LICENSES/LLMCoSolver-MIT.txt`. The Qwen base model has its own Apache-2.0 terms and is not redistributed here. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No claim is made that inherited components were originally authored for DecLLM.
