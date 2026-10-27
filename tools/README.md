# Optional tools

The default SSNC-v1 training and inference commands use `scripts/launch.py`, `dt_train.py` and `dt_infer.py`. The helpers below are optional and are not invoked by that launcher. Run module commands from the repository root, for example:

```bash
python -m tools.data.generate_random_generalization_instances --help
python -m tools.data.generate_unseen_generalization_instances --help
python -m tools.analysis.plot_rollout_gap_history --help
python -m tools.checkpoints.verify_unseen_allone --help
```

Module execution (`python -m tools...`) keeps imports of the root model/environment modules available. Direct execution of nested script paths is not the supported interface.

## Data tools

| Module under `tools.data` | Purpose and scope |
| --- | --- |
| `generate_random_generalization_instances` | Generate random JSSP evaluation instances. Pass an explicit output directory. |
| `generate_unseen_generalization_instances` | Generate the six unseen-size groups using the shared random-instance generator. Pass an explicit output directory. |
| `qa_random_generalization_instances` | Check the generated random-instance files; pass `--root` for your dataset. |
| `qa_unseen_generalization_instances` | Check the generated unseen-instance files; pass `--root` for your dataset. |
| `collect_multi_scale` | Historical small-scale trajectory-collection example with three hard-coded configurations; it does not reproduce the paper's 11-configuration training dataset. Its historical `5x10` label corresponds to 10 jobs and 5 machines. Review its settings before use. |
| `collect_dagger` | Optional DAgger collection experiment combining student rollouts and OR-Tools labels. It is not part of the default offline SSNC-v1 training pipeline. |

## Evaluation tools

`python -m tools.evaluation.eval_unseen_spt_fifo --help` shows the SPT/FIFO comparison interface. This helper expects compatible per-instance model results and matching instance identifiers. Some defaults refer to historical step2700 result directories. Supply the dataset, model-result, code-root and output arguments for your own experiment; it is not a ready-made current-checkpoint benchmark command.

## Analysis tools

- `tools.analysis.plot_rollout_gap_history`: plot rollout-panel Gap history from a training log.
- `tools.analysis.analyze_rollout_and_masks_all_steps`: inspect saved rollout and mask records across selected steps.
- `tools.analysis.visualize_scale_param_masks_compare`: compare two supplied mask checkpoints; the existing CLI arguments are named `--step300` and `--step600`.

Use `python -m MODULE --help` to inspect each interface. These tools require their own logs/checkpoints; moving them does not supply the experiment data.

## Checkpoint and figure tools

- `tools.checkpoints.inspect_ssnc_checkpoint`: inspect SSNC controller state in a supplied checkpoint.
- `tools.checkpoints.verify_unseen_allone`: check that the six configured unseen-size mask rows are all-one. This was formerly named `verify_step2700_unseen_allone.py`; its checks are unchanged. It checks stored rows, not an end-to-end rollout.
- `tools.figures.draw_dt_jssp_architecture`: draw an auxiliary architecture illustration into the repository's `figures/` directory. Run `python -m tools.figures.draw_dt_jssp_architecture`.

The four obsolete fixed-path scripts `analyze_ssnc_vs_step2700_all_results.py`, `inspect_step2700_unseen_scale_routing.py`, `summarize_phase1_fixed_bon8.py` and `summarize_phase1_fixed_ta.py` were removed from the current tree. They remain recoverable from Git history before the directory cleanup.

`dataset_optimization.py` remains in the repository root because the training entrypoint conditionally imports it when offline data augmentation is enabled. The released SSNC-v1 configuration does not enable that option.
