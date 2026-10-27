# Validation scope

## Repository organization (2026-10-27)

Four obsolete fixed-path step2700/phase1 analysis scripts were removed. Seven checks were moved into `tests/`, and thirteen optional helpers were grouped under `tools/`. The check runner and two shared imports were updated for module execution. The historical collection example's usage/path bootstrap and the drawing helper's output path were adjusted for the new locations. README files document the supported commands and the historical scope of optional helpers. The pre-cleanup source was retained separately, and removed scripts remain available in Git history before this cleanup.

The nine core/model/environment/augmentation files, launcher and SSNC-v1 training configuration are byte-identical to pre-cleanup commit `6c511cdc68bcc2ed546befd383169b1b526a1ded`. Sixteen of the twenty moved files are also byte-identical; the other four contain only the import, usage/bootstrap or output-path changes described above.

Verified in the existing Linux server environment:

- All ten existing CPU checks passed, including the saved-checkpoint routing check; none were skipped.
- All twelve CLI-equipped optional helpers completed `--help`, and the auxiliary figure module imported successfully.
- Full-training, two-update smoke-training and inference launcher arguments matched the previous public package, after normalizing entrypoint/output locations.
- The original best SSNC-v1 checkpoint completed the generated 10x5 Greedy instance with 50 job-selection actions, seven clock-advance records and makespan 43. Its complete action sequence and makespan matched the previous public-package validation.

See `validation_layout_summary.json`. This organization check did not perform new optimizer updates, full retraining or complete benchmark reproduction. The makespan above is a functional regression result, not a benchmark average. Previous training-regression evidence below refers to the earlier cleanup.

## Cleanup regression

The second-stage cleanup removed the unused root-level `utils.py` and `rewards.py` inherited from a multi-problem text-scoring workflow. The only production edit was deletion of two unused wildcard imports in `dt_train.py`. JSSP environment rewards, makespan computation, graph encoding, data processing, masking and inference were retained.

Verified on the existing server environment, on the same physical GPU with seed 62:

- Ten CPU checks passed, including masks, state tensorization, entrypoint parity, top-p sampling, saved-mask routing and mock forward/backward execution.
- All 168 CLI defaults matched before and after cleanup; scope-aware checks found no missing global bindings.
- Both code trees used the same 88 trajectories across 11 configurations and the same instance-hash split; two optimizer updates were performed per side using the actual Qwen2.5-7B model and 8-bit AdamW.
- All five SSNC mask sites updated twice per side.
- All 24 observed loss calls, logits and observed logit gradients matched bitwise.
- All 491 optimizer parameters per update, update-time gradients and post-update parameter hashes matched.
- All 672 saved LoRA/custom/mask/controller tensors matched elementwise, as did saved non-tensor state.
- Reloaded smoke checkpoints produced identical complete 10x5 Greedy action sequences: 50 job-selection actions, seven clock-advance records and makespan 55. This number is a workflow check, not a paper-performance result.

Original source, the previous cleanup version and complete backups were hash-verified unchanged. Those internal backups and machine-specific logs are not published.

## Public-package changes

Core model, train, infer, environment and state/mask implementations remain byte-identical to the tested second-stage version. Public packaging changes only convenience/configuration/documentation surfaces: portable training-argument JSON, local-path configuration, output placement inside the repository, standalone CPU-test reporting, and generic defaults in nine optional helper scripts. Historical machine-specific shell launchers, internal migration metadata, caches and training-data subsets are excluded from the public tree but preserved privately.

All ten public-package CPU checks passed. Parsed full-training and smoke-training arguments matched the preceding version except output paths, and the inference command matched except entrypoint/output locations. The public launcher loaded the original best checkpoint and completed the same 10x5 Greedy trajectory as the preceding cleaned entrypoint: 50 job selections, seven clock-advance records, makespan 43. This is a functional comparison, not a benchmark average. See `validation_summary.json`.

A full clean installation, 12-epoch retraining, complete benchmark/ablation reproduction and optimizer-state resume are not claimed. The original fixed validation panel must still be supplied for exact full-training reproduction. Bitwise agreement on one fixed regression is not a promise for every GPU, seed or data configuration.

## Released checkpoint

The Release asset is the original `best_gap_model_final_epoch_12`, not either regression checkpoint. Its training log records best validation Gap 7.3906%. LoRA and custom-module file hashes are preserved from the server source. The original formal checkpoint was also successfully loaded by the cleaned inference entrypoint during the previous functional validation.
