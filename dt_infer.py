"""Decision Transformer inference script for JSSP problem, supports loading and merging PEFT models."""

import argparse
import torch
import os
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from matplotlib.patches import Patch

# Import our custom modules
from decision_transformer import DecisionTransformer, create_decision_transformer_for_jssp
from dt_data_processing import ORToolsAgent
from state_tensorization import (
    LEGACY_TRAIN_COMPATIBLE,
    VALID_TENSORIZATION_MODES,
    get_state_tensorization_mode,
    set_state_tensorization_mode,
    tensorize_graph_state,
)
from ssnc_inference_routing import apply_unseen_scale_neighbor_routing


def _sample_candidate_position(candidate_logits, temperature, top_p=0.0):
    """Sample one local candidate index, optionally using nucleus (top-p) sampling.

    ``candidate_logits`` must already contain only the candidates allowed by any
    top-k filter. Illegal actions may remain as ``-inf``; they receive zero
    probability. ``top_p`` values 0 and 1 both disable nucleus truncation so the
    former preserves the historical default and the latter means all candidates.
    """
    probs = torch.softmax(candidate_logits / float(temperature), dim=-1)
    p = float(top_p or 0.0)

    if 0.0 < p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Keep the first candidate that makes the cumulative mass reach p.
        # This is the standard shifted nucleus mask and always retains >=1 item.
        remove_mask = cumulative_probs > p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled_sorted_pos = torch.multinomial(sorted_probs, num_samples=1)
        sampled_local_pos = sorted_indices.gather(-1, sampled_sorted_pos)
        return int(sampled_local_pos.item())

    return int(torch.multinomial(probs, num_samples=1).item())

def encode_jssp_state(obs, n_jobs=5, n_machines=5, state_dim=64, global_proc_time_max=None):
    """
    将JSSP环境返回的状态编码为固定维度向量（基于operation的特征设计）

    Args:
        obs: 环境返回的原始状态（dict）
        n_jobs: 作业数
        n_machines: 机器数
        state_dim: 模型训练时使用的 state_dim
        global_proc_time_max: 全局最大处理时间，用于一致的归一化

    Returns:
        encoded: numpy array of shape (state_dim,)
    """
    if isinstance(obs, dict):
        N = n_jobs * n_machines  # Total number of operations

        # Extract required information from state
        job_next_op = np.array(obs.get('job_next_op', np.zeros(n_jobs)))

        # Ensure processing_times is correctly shaped
        processing_times = np.array(obs.get('processing_times', np.zeros((n_jobs, n_machines))))
        if processing_times.shape != (n_jobs, n_machines):
            processing_times = np.zeros((n_jobs, n_machines))

        # Ensure machine_sequence is correctly shaped
        machine_sequence = np.array(obs.get('machine_sequence', np.zeros((n_jobs, n_machines), dtype=int)))
        if machine_sequence.shape != (n_jobs, n_machines):
            machine_sequence = np.zeros((n_jobs, n_machines), dtype=int)

        current_time = obs.get('current_time', 0)
        machine_status = np.array(obs.get('machine_status', np.zeros(n_machines)))

        completed_operations = obs.get('completed_operations', 0)
        makespan_estimate = obs.get('makespan_estimate', current_time)

        # Calculate total processing time as a robust upper bound
        total_proc_time = np.sum(processing_times)
        max_expected_time = max(makespan_estimate, total_proc_time)

        # Feature 1: Status (-1=未调度, 0=已调度未完成, 1=已完成)
        status = np.full(N, -1.0)
        for j in range(n_jobs):
            next_op = int(job_next_op[j])
            for op in range(n_machines):
                op_idx = j * n_machines + op
                if op < next_op:
                    status[op_idx] = 1.0  # completed
                elif op == next_op and next_op < n_machines:
                    status[op_idx] = 0.0  # ready/scheduled
                else:
                    status[op_idx] = -1.0  # unscheduled

        # Feature 2: Processing Time (已知，静态)
        proc_time = processing_times.flatten()
        # Normalize processing time using global maximum if available
        if global_proc_time_max and global_proc_time_max > 0:
            proc_time = proc_time / global_proc_time_max
        elif np.max(proc_time) > 0:
            proc_time = proc_time / np.max(proc_time)  # Fallback to instance-specific max

        # Feature 3: Remaining Operations in Job
        remaining_ops = np.zeros(N)
        for j in range(n_jobs):
            remaining = n_machines - job_next_op[j]  # 包括当前
            for op in range(n_machines):
                op_idx = j * n_machines + op
                remaining_ops[op_idx] = remaining
        # Normalize remaining operations
        if n_machines > 0:
            remaining_ops = remaining_ops / n_machines

        # Feature 4: Degree of Completion of Job
        job_total_time = np.sum(processing_times, axis=1)  # shape: [n_jobs]
        degree_comp = np.zeros(N)

        for j in range(n_jobs):
            total = job_total_time[j]
            if total > 0:
                for op in range(n_machines):
                    op_idx = j * n_machines + op
                    degree_comp[op_idx] = processing_times[j][op] / total

        # Feature 5: Estimated Completion Time of Operation
        est_complete = np.zeros(N)

        for j in range(n_jobs):
            for op in range(n_machines):
                op_idx = j * n_machines + op

                if op < job_next_op[j]:
                    # 已完成，设为当前时间
                    est_complete[op_idx] = current_time
                else:
                    # 未完成：预估开始时间 = max(前序完成, machine可用)
                    m = machine_sequence[j][op]
                    pt = processing_times[j][op]

                    # 粗略估计：假设从当前时间开始（或 machine 可用时间）
                    start_est = max(current_time, machine_status[m])
                    est_complete[op_idx] = start_est + pt

        # Normalize estimated completion time using global maximum if available
        reference_max = max_expected_time
        if global_proc_time_max and global_proc_time_max > reference_max:
            reference_max = global_proc_time_max

        if reference_max > 0:
            est_complete = est_complete / reference_max

        # Stack all operation features and flatten
        op_features = np.stack([status, proc_time, remaining_ops, degree_comp, est_complete], axis=1)  # [N, 5]
        flat_op_features = op_features.flatten()  # [N*5]

        # Add global information
        global_info = np.array([
            current_time / (reference_max if reference_max > 0 else 1),
            completed_operations / (n_jobs * n_machines),
            makespan_estimate / (reference_max if reference_max > 0 else 1)
        ])

        # Combine operation features and global info
        final_state = np.concatenate([flat_op_features, global_info])  # shape = [N*5 + 3]

        # Pad or truncate to fixed length
        if len(final_state) > state_dim:
            return final_state[:state_dim]
        elif len(final_state) < state_dim:
            return np.pad(final_state, (0, state_dim - len(final_state)))
        return final_state

    else:
        # 如果是列表等格式，将其转换为numpy数组并处理
        features = np.array(obs)
        features = features.flatten()

        # Pad or truncate to fixed length
        if len(features) > state_dim:
            features = features[:state_dim]
        else:
            features = np.pad(features, (0, state_dim - len(features)))

        return features.astype(np.float32)

def _flatten_feature(x):
    if isinstance(x, torch.Tensor):
        t = x.detach().to(dtype=torch.float32)
    else:
        t = torch.as_tensor(x, dtype=torch.float32)
    return t.flatten()


def _ensure_btd(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    while x.dim() < 3:
        x = x.unsqueeze(-1)
    return x


def _pad_time_to(x, target_t):
    x = _ensure_btd(x)
    b, t, d = x.shape
    if t == target_t:
        return x
    if t > target_t:
        return x[:, -target_t:, :]
    pad = torch.zeros((b, target_t - t, d), dtype=x.dtype, device=x.device)
    return torch.cat([pad, x], dim=1)


def _obs_to_state_dict(
    obs,
    keys,
    problem_n_jobs=None,
    problem_n_machines=None,
    model_max_jobs=None,
    model_max_machines=None,
    state_tensorization_mode=None,
):
    if not isinstance(obs, dict):
        return None
    actual_jobs = int(problem_n_jobs or len(obs.get("job_next_op", [])) or 0)
    actual_machines = int(
        problem_n_machines
        or len(obs.get("normalized_machine_availability", []))
        or 0
    )
    max_jobs = int(model_max_jobs or actual_jobs)
    max_machines = int(model_max_machines or actual_machines)
    return tensorize_graph_state(
        obs,
        actual_n_jobs=actual_jobs,
        actual_n_machines=actual_machines,
        max_jobs=max_jobs,
        max_machines=max_machines,
        keys=keys,
        mode=state_tensorization_mode or get_state_tensorization_mode(),
    )


def _to_jsonable(x):
    import numpy as _np
    import torch as _torch

    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, _np.generic):
        return x.item()
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, _torch.Tensor):
        return x.detach().cpu().to(dtype=_torch.float32).tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def _estimate_jssp_makespan_target(env):
    """Estimate a reasonable makespan target from the current instance."""
    try:
        instance = getattr(env, "current_instance", None)
        if isinstance(instance, np.ndarray) and instance.ndim == 2 and instance.shape[1] >= 2:
            machine_ids = instance[:, 0::2].astype(np.int64, copy=False)
            proc_times = instance[:, 1::2].astype(np.float32, copy=False)

            job_totals = proc_times.sum(axis=1)
            job_load_lb = float(np.max(job_totals)) if job_totals.size > 0 else 0.0
            n_machines = int(getattr(env, "n_machines", proc_times.shape[1]) or proc_times.shape[1])
            machine_loads = np.zeros((max(1, n_machines),), dtype=np.float32)
            for j in range(proc_times.shape[0]):
                for o in range(proc_times.shape[1]):
                    m_id = int(machine_ids[j, o])
                    if 0 <= m_id < machine_loads.shape[0]:
                        machine_loads[m_id] += float(proc_times[j, o])
            machine_load_lb = float(np.max(machine_loads)) if machine_loads.size > 0 else 0.0
            lower_bound = max(job_load_lb, machine_load_lb)
            if lower_bound > 0:
                return lower_bound
    except Exception:
        pass

    fallback = float(getattr(env, "max_makespan", 0) or 0.0)
    if fallback > 0:
        return fallback
    return 1.0


def _resolve_rtg_condition(env, rtg_mode, target_return=1.0, target_makespan=None):
    rtg_mode = str(rtg_mode).lower().strip()
    if rtg_mode not in {"constant", "budget", "normalized_budget"}:
        raise ValueError(f"Unknown rtg_mode={rtg_mode}. Use 'constant', 'budget' or 'normalized_budget'.")

    if rtg_mode == "constant":
        return rtg_mode, None

    base_budget = float(target_makespan) if target_makespan is not None else _estimate_jssp_makespan_target(env)
    base_budget = max(1.0, base_budget)
    budget_scale = float(target_return) if target_return is not None else 1.0
    conditioned_budget = max(1.0, base_budget * budget_scale)
    return rtg_mode, conditioned_budget


def _compute_rtg_value(rtg_mode, target_return=1.0, conditioned_budget=None, current_time=0.0):
    current_time = float(current_time)
    if rtg_mode == "constant":
        return -float(target_return) if target_return is not None else -1.0
    if conditioned_budget is None:
        conditioned_budget = 1.0
    remaining_budget = max(0.0, float(conditioned_budget) - current_time)
    if rtg_mode == "budget":
        return remaining_budget
    if rtg_mode == "normalized_budget":
        return -remaining_budget / float(conditioned_budget)
    raise ValueError(f"Unknown rtg_mode={rtg_mode}")


def _sync_env_max_steps(env, max_steps):
    """Keep the environment's internal operation cap aligned with inference."""
    try:
        total_ops = int(getattr(env, "n_jobs")) * int(getattr(env, "n_machines"))
    except Exception:
        total_ops = 0
    try:
        requested = int(max_steps)
    except Exception:
        requested = 0
    try:
        current = int(getattr(env, "max_steps", 0) or 0)
    except Exception:
        current = 0
    if hasattr(env, "max_steps"):
        env.max_steps = max(current, requested, total_ops)


def _env_all_jobs_completed(env):
    try:
        return all(int(op) >= int(env.n_machines) for op in env.job_status)
    except Exception:
        return False


@torch.no_grad()
def rollout_episode_with_trajectory(
    model,
    env,
    target_makespan=None,
    target_return=1.0,
    rtg_mode="normalized_budget",
    max_steps=200,
    max_length=64,
    temperature=0.0,
    top_k=0,
    top_p=0.0,
    device="cuda",
    debug=False,
):
    model.eval()

    _sync_env_max_steps(env, max_steps)
    obs = env.reset()
    _sync_env_max_steps(env, max_steps)
    done = False

    state_keys = [
        "job_next_op",
        "job_completion_mask",
        "remaining_processing_times",
        "normalized_machine_availability",
        "machine_busy_mask",
        "overall_info",
        "machine_sequence",
        "scheduled_machine_adj",
        "problem_n_jobs",
        "problem_n_machines",
    ]

    states_history = {k: [] for k in state_keys}
    actions_history = [-1]
    timesteps_history = [0]

    rtg_mode, conditioned_budget = _resolve_rtg_condition(
        env,
        rtg_mode=rtg_mode,
        target_return=target_return,
        target_makespan=target_makespan,
    )
    rtg_history = [
        _compute_rtg_value(
            rtg_mode,
            target_return=target_return,
            conditioned_budget=conditioned_budget,
            current_time=float(getattr(env, "current_time", 0.0)),
        )
    ]

    actions = []
    rewards = []
    states = [obs]

    for step in range(int(max_steps)):
        n_jobs = getattr(env, "n_jobs", None)
        n_machines = getattr(env, "n_machines", None)
        model_max_jobs = getattr(getattr(model, "state_encoder", None), "max_jobs", None)
        model_max_machines = getattr(getattr(model, "state_encoder", None), "max_machines", None)
        state_dict_step = _obs_to_state_dict(
            obs,
            state_keys,
            problem_n_jobs=n_jobs,
            problem_n_machines=n_machines,
            model_max_jobs=model_max_jobs,
            model_max_machines=model_max_machines,
        )
        if state_dict_step is None:
            raise TypeError("Environment observation must be a dict for this inference script.")

        for k in state_keys:
            states_history[k].append(state_dict_step[k])

        start_idx = max(0, len(actions_history) - int(max_length))
        seq_len = len(actions_history) - start_idx

        model_states = {}
        for k in state_keys:
            seq = states_history[k][start_idx:]
            model_states[k] = _pad_time_to(torch.stack(seq, dim=0), seq_len).to(device)

        if debug:
            shapes = {k: tuple(v.shape) for k, v in model_states.items()}
            print(f"[DEBUG] model_states shapes: {shapes}")

        model_actions = torch.tensor(actions_history[start_idx:], dtype=torch.long, device=device).unsqueeze(0)
        model_returns = torch.tensor(rtg_history[start_idx:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        model_timesteps = torch.tensor(timesteps_history[start_idx:], dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

        outputs = model(
            states=model_states,
            actions=model_actions,
            returns_to_go=model_returns,
            timesteps=model_timesteps,
            attention_mask=attn_mask,
        )
        logits = outputs.logits[:, -1, :]

        action_mask = None
        action_space = None
        if hasattr(env, "get_available_actions"):
            try:
                action_space = env.get_available_actions()
            except Exception:
                action_space = None
        if action_space is None and hasattr(env, "action_space"):
            action_space = getattr(env, "action_space", None)

        if action_space is not None:
            action_mask = torch.zeros((int(logits.shape[-1]),), dtype=torch.bool, device=device)
            for a in action_space:
                ai = int(a)
                if 0 <= ai < action_mask.numel():
                    action_mask[ai] = True
        elif hasattr(env, "get_action_mask"):
            try:
                action_mask = env.get_action_mask()
            except Exception:
                action_mask = None
            if action_mask is not None:
                if not isinstance(action_mask, torch.Tensor):
                    action_mask = torch.as_tensor(action_mask, dtype=torch.bool)
                action_mask = action_mask.to(device=device).view(-1)

        if action_mask is not None:
            if int(action_mask.numel()) < int(logits.shape[-1]):
                pad = torch.zeros((int(logits.shape[-1]) - int(action_mask.numel()),), dtype=torch.bool, device=device)
                action_mask = torch.cat([action_mask, pad], dim=0)
            elif int(action_mask.numel()) > int(logits.shape[-1]):
                action_mask = action_mask[: int(logits.shape[-1])]
            masked_logits = logits.clone()
            masked_logits[:, ~action_mask] = -float("inf")
        else:
            masked_logits = logits

        if torch.isinf(masked_logits).all():
            chosen = -1
        else:
            if temperature and float(temperature) > 0:
                k = int(top_k or 0)
                if k > 0 and action_mask is not None:
                    legal_count = int(action_mask.sum().item())
                    k = max(1, min(k, legal_count))
                    top_vals, top_idx = torch.topk(masked_logits, k=k, dim=-1)
                    pos = _sample_candidate_position(top_vals, temperature, top_p)
                    chosen = int(top_idx[0, pos].item())
                else:
                    chosen = _sample_candidate_position(masked_logits, temperature, top_p)
            else:
                chosen = int(torch.argmax(masked_logits, dim=-1).item())

        if action_mask is not None and int(action_mask.sum().item()) == 0:
            chosen = -1

        if chosen == -1:
            step_result = env.step()
        else:
            step_result = env.step(action=chosen)

        if isinstance(step_result, tuple) and len(step_result) == 4:
            next_obs, reward, done, info = step_result
        elif isinstance(step_result, tuple) and len(step_result) == 3:
            next_obs, done, info = step_result
            reward = float(info.get("reward", 0.0))
        else:
            raise ValueError(f"Unexpected env.step return: {step_result}")

        if isinstance(info, dict) and "invalid" in info:
            step_result2 = env.step()
            if isinstance(step_result2, tuple) and len(step_result2) == 4:
                next_obs, reward2, done2, info2 = step_result2
            elif isinstance(step_result2, tuple) and len(step_result2) == 3:
                next_obs, done2, info2 = step_result2
                reward2 = float(info2.get("reward", 0.0))
            else:
                raise ValueError(f"Unexpected env.step return: {step_result2}")
            reward = float(reward) + float(reward2)
            done = bool(done) or bool(done2)
            info = info2 if isinstance(info2, dict) else info

        if debug:
            valid_count = int(action_mask.sum().item()) if isinstance(action_mask, torch.Tensor) else -1
            print(f"Step {step} action={chosen} reward={reward:.4f} done={done} valid_jobs={valid_count}")
            if isinstance(info, dict) and "invalid" in info:
                print(f"  invalid={info.get('invalid')}")

        actions.append(chosen)
        rewards.append(float(reward))

        obs = next_obs
        states.append(obs)
        actions_history.append(chosen)
        timesteps_history.append(timesteps_history[-1] + 1)
        rtg_history.append(
            _compute_rtg_value(
                rtg_mode,
                target_return=target_return,
                conditioned_budget=conditioned_budget,
                current_time=float(getattr(env, "current_time", 0.0)),
            )
        )

        if done:
            break

    makespan = getattr(env, "get_makespan", lambda: -1)()
    if not _env_all_jobs_completed(env):
        if debug:
            completed = getattr(env, "completed_operations", None)
            total = int(getattr(env, "n_jobs", 0) or 0) * int(getattr(env, "n_machines", 0) or 0)
            print(f"[DEBUG] Incomplete rollout: completed_operations={completed}/{total}; makespan set to NaN")
        makespan = float("nan")

    if len(rtg_history) < len(states):
        rtg_history = rtg_history + [rtg_history[-1]] * (len(states) - len(rtg_history))
    elif len(rtg_history) > len(states):
        rtg_history = rtg_history[: len(states)]

    trajectory = {
        "states": _to_jsonable(states),
        "actions": [int(a) for a in actions],
        "rewards": [float(r) for r in rewards],
        "returns_to_go": [float(x) for x in rtg_history],
        "timesteps": list(range(len(states))),
        "metrics": {
            "inference": True,
            "rtg_mode": str(rtg_mode),
            "target_return": float(target_return),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "actual_makespan": float(makespan) if makespan is not None else float("nan"),
        },
    }
    return trajectory, makespan


@torch.no_grad()
def rollout_episode(
    model,
    env,
    target_makespan=None,
    target_return=1.0,
    rtg_mode="normalized_budget",
    max_steps=200,
    max_length=64,
    temperature=0.0,
    top_k=0,
    top_p=0.0,
    device="cuda",
    debug=False,
):
    model.eval()

    _sync_env_max_steps(env, max_steps)
    obs = env.reset()
    _sync_env_max_steps(env, max_steps)
    done = False

    state_keys = [
        "job_next_op",
        "job_completion_mask",
        "remaining_processing_times",
        "normalized_machine_availability",
        "machine_busy_mask",
        "overall_info",
        "machine_sequence",
        "scheduled_machine_adj",
        "problem_n_jobs",
        "problem_n_machines",
    ]

    states_history = {k: [] for k in state_keys}
    actions_history = [-1]
    timesteps_history = [0]

    rtg_mode, conditioned_budget = _resolve_rtg_condition(
        env,
        rtg_mode=rtg_mode,
        target_return=target_return,
        target_makespan=target_makespan,
    )
    rtg_history = [
        _compute_rtg_value(
            rtg_mode,
            target_return=target_return,
            conditioned_budget=conditioned_budget,
            current_time=float(getattr(env, "current_time", 0.0)),
        )
    ]

    actions = []
    rewards = []

    for step in range(int(max_steps)):
        n_jobs = getattr(env, "n_jobs", None)
        n_machines = getattr(env, "n_machines", None)
        model_max_jobs = getattr(getattr(model, "state_encoder", None), "max_jobs", None)
        model_max_machines = getattr(getattr(model, "state_encoder", None), "max_machines", None)
        state_dict_step = _obs_to_state_dict(
            obs,
            state_keys,
            problem_n_jobs=n_jobs,
            problem_n_machines=n_machines,
            model_max_jobs=model_max_jobs,
            model_max_machines=model_max_machines,
        )
        if state_dict_step is None:
            raise TypeError("Environment observation must be a dict for this inference script.")

        for k in state_keys:
            states_history[k].append(state_dict_step[k])

        start_idx = max(0, len(actions_history) - int(max_length))
        seq_len = len(actions_history) - start_idx

        model_states = {}
        for k in state_keys:
            seq = states_history[k][start_idx:]
            model_states[k] = _pad_time_to(torch.stack(seq, dim=0), seq_len).to(device)

        if debug:
            shapes = {k: tuple(v.shape) for k, v in model_states.items()}
            print(f"[DEBUG] model_states shapes: {shapes}")

        model_actions = torch.tensor(actions_history[start_idx:], dtype=torch.long, device=device).unsqueeze(0)
        model_returns = torch.tensor(rtg_history[start_idx:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        model_timesteps = torch.tensor(timesteps_history[start_idx:], dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

        outputs = model(
            states=model_states,
            actions=model_actions,
            returns_to_go=model_returns,
            timesteps=model_timesteps,
            attention_mask=attn_mask,
        )
        logits = outputs.logits[:, -1, :]  # (1, n_jobs)

        action_mask = None
        action_space = None
        if hasattr(env, "get_available_actions"):
            try:
                action_space = env.get_available_actions()
            except Exception:
                action_space = None
        if action_space is None and hasattr(env, "action_space"):
            action_space = getattr(env, "action_space", None)

        if action_space is not None:
            action_mask = torch.zeros((int(logits.shape[-1]),), dtype=torch.bool, device=device)
            for a in action_space:
                ai = int(a)
                if 0 <= ai < action_mask.numel():
                    action_mask[ai] = True
        elif hasattr(env, "get_action_mask"):
            try:
                action_mask = env.get_action_mask()
            except Exception:
                action_mask = None
            if action_mask is not None:
                if not isinstance(action_mask, torch.Tensor):
                    action_mask = torch.as_tensor(action_mask, dtype=torch.bool)
                action_mask = action_mask.to(device=device).view(-1)

        if action_mask is not None:
            if int(action_mask.numel()) < int(logits.shape[-1]):
                pad = torch.zeros((int(logits.shape[-1]) - int(action_mask.numel()),), dtype=torch.bool, device=device)
                action_mask = torch.cat([action_mask, pad], dim=0)
            elif int(action_mask.numel()) > int(logits.shape[-1]):
                action_mask = action_mask[: int(logits.shape[-1])]
            masked_logits = logits.clone()
            masked_logits[:, ~action_mask] = -float("inf")
        else:
            masked_logits = logits

        if torch.isinf(masked_logits).all():
            chosen = -1
        else:
            if temperature and float(temperature) > 0:
                k = int(top_k or 0)
                if k > 0 and action_mask is not None:
                    legal_count = int(action_mask.sum().item())
                    k = max(1, min(k, legal_count))
                    top_vals, top_idx = torch.topk(masked_logits, k=k, dim=-1)
                    pos = _sample_candidate_position(top_vals, temperature, top_p)
                    chosen = int(top_idx[0, pos].item())
                else:
                    chosen = _sample_candidate_position(masked_logits, temperature, top_p)
            else:
                chosen = int(torch.argmax(masked_logits, dim=-1).item())

        if action_mask is not None and int(action_mask.sum().item()) == 0:
            chosen = -1

        if chosen == -1:
            step_result = env.step()
        else:
            step_result = env.step(action=chosen)

        if isinstance(step_result, tuple) and len(step_result) == 4:
            next_obs, reward, done, info = step_result
        elif isinstance(step_result, tuple) and len(step_result) == 3:
            next_obs, done, info = step_result
            reward = float(info.get("reward", 0.0))
        else:
            raise ValueError(f"Unexpected env.step return: {step_result}")

        if isinstance(info, dict) and "invalid" in info:
            step_result2 = env.step()
            if isinstance(step_result2, tuple) and len(step_result2) == 4:
                next_obs, reward2, done2, info2 = step_result2
            elif isinstance(step_result2, tuple) and len(step_result2) == 3:
                next_obs, done2, info2 = step_result2
                reward2 = float(info2.get("reward", 0.0))
            else:
                raise ValueError(f"Unexpected env.step return: {step_result2}")
            reward = float(reward) + float(reward2)
            done = bool(done) or bool(done2)
            info = info2 if isinstance(info2, dict) else info

        if debug:
            valid_count = int(action_mask.sum().item()) if isinstance(action_mask, torch.Tensor) else -1
            print(f"Step {step} action={chosen} reward={reward:.4f} done={done} valid_jobs={valid_count}")
            if isinstance(info, dict) and "invalid" in info:
                print(f"  invalid={info.get('invalid')}")

        actions.append(chosen)
        rewards.append(float(reward))

        obs = next_obs
        actions_history.append(chosen)
        timesteps_history.append(timesteps_history[-1] + 1)
        rtg_history.append(
            _compute_rtg_value(
                rtg_mode,
                target_return=target_return,
                conditioned_budget=conditioned_budget,
                current_time=float(getattr(env, "current_time", 0.0)),
            )
        )

        if done:
            break

    makespan = getattr(env, "get_makespan", lambda: -1)()
    if not _env_all_jobs_completed(env):
        if debug:
            completed = getattr(env, "completed_operations", None)
            total = int(getattr(env, "n_jobs", 0) or 0) * int(getattr(env, "n_machines", 0) or 0)
            print(f"[DEBUG] Incomplete rollout: completed_operations={completed}/{total}; makespan set to NaN")
        makespan = float("nan")
    return actions, rewards, makespan

def get_schedule_from_env(env):
    """
    从JSSP环境中提取调度信息

    Args:
        env: JSSP环境实例

    Returns:
        list: 包含每个操作执行记录的列表
    """
    schedule = []

    # 从环境的schedule数组中提取信息
    for job_id in range(env.n_jobs):
        for op_idx in range(env.n_machines):
            start_time, end_time = env.schedule[job_id, op_idx]
            if start_time > 0 or end_time > 0:  # 只包含已调度的操作
                # 获取该操作需要的机器
                machine_id = int(env.current_instance[job_id, op_idx * 2])
                duration = end_time - start_time

                schedule.append({
                    "machine_id": machine_id,
                    "job_id": job_id,
                    "operation_id": op_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration
                })

    return schedule

def get_schedule_from_ortools(ortools_schedule, current_instance):
    """
    从OR-Tools的调度结果中提取调度信息

    Args:
        ortools_schedule: OR-Tools求解得到的调度结果，格式为list[list]，每个子列表包含机器上的作业顺序
        current_instance: JSSP实例矩阵

    Returns:
        list: 包含每个操作执行记录的列表
    """
    schedule = []
    n_jobs = current_instance.shape[0]
    n_machines = current_instance.shape[1] // 2

    # 计算每个作业的操作完成时间
    job_finish = [0] * n_jobs
    # 计算每个机器的可用时间
    machine_finish = [0] * n_machines
    # 记录每个作业的当前操作索引
    job_next_op = [0] * n_jobs

    # 遍历每个机器的作业顺序
    for machine_id in range(len(ortools_schedule)):
        job_order = ortools_schedule[machine_id]
        for job_id in job_order:
            op_idx = job_next_op[job_id]
            if op_idx >= n_machines:
                continue  # 作业已完成所有操作

            # 获取当前操作的处理时间
            duration = int(current_instance[job_id, op_idx * 2 + 1])

            # 计算开始时间：作业的前一个操作完成时间和机器可用时间的最大值
            start_time = max(job_finish[job_id], machine_finish[machine_id])
            end_time = start_time + duration

            # 更新作业的完成时间和操作索引
            job_finish[job_id] = end_time
            job_next_op[job_id] += 1

            # 更新机器的可用时间
            machine_finish[machine_id] = end_time

            # 记录操作信息
            schedule.append({
                "machine_id": machine_id,
                "job_id": job_id,
                "operation_id": op_idx,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            })

    return schedule

def plot_gantt_chart(schedule, n_jobs, n_machines, title="JSSP Gantt Chart", save_path=None):
    """
    绘制JSSP调度甘特图

    Args:
        schedule: 列表，每个元素为dict，包含：
                  {'job_id': int, 'machine_id': int,
                   'start_time': float, 'duration': float}
        n_jobs: 作业数量
        n_machines: 机器数量
        title: 图表标题
        save_path: 保存图表的路径
    """
    # 按machine_id分组
    machine_ops = defaultdict(list)
    for op in schedule:
        mid = op['machine_id']
        machine_ops[mid].append(op)

    # 排序：按开始时间排序
    for mid in machine_ops:
        machine_ops[mid].sort(key=lambda x: x['start_time'])

    # 创建颜色映射（每个job一种颜色）
    colors = plt.cm.tab20(np.linspace(0, 1, n_jobs))
    color_map = {j: colors[j] for j in range(n_jobs)}

    fig, ax = plt.subplots(figsize=(14, 6))

    # 设置y轴为机器编号
    machines = sorted(machine_ops.keys())
    ax.set_ylim(-0.5, len(machines) - 0.5)
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f'Machine {m}' for m in machines])

    # 设置x轴为时间轴
    max_time = max((op['start_time'] + op['duration']) for op in schedule) if schedule else 10
    ax.set_xlim(0, max_time * 1.05)
    ax.set_xlabel("Time")
    ax.set_title(title)

    # 绘制每个操作的条形
    for i, mid in enumerate(machines):
        ops = machine_ops[mid]
        for op in ops:
            job_id = op['job_id']
            start = op['start_time']
            dur = op['duration']
            color = color_map[job_id]

            # 添加矩形条
            ax.barh(i, dur, left=start, height=0.6, color=color, edgecolor='black', linewidth=0.5)

            # 添加标签
            text_x = start + dur / 2
            ax.text(text_x, i, f'Job{job_id}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # 添加图例
    legend_elements = [Patch(facecolor=color_map[j], label=f'Job {j}') for j in range(n_jobs)]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 网格线
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"甘特图已保存为: {save_path}")

    plt.close()  # 关闭图表，避免内存泄漏
    return fig

def load_and_merge_model(
    base_model_name,
    peft_adapter_path,
    custom_modules_path=None,
    n_jobs=10,
    n_machines=10,
    merge_weights=False,
    load_in_4bit=False,
    save_merged_path=None,
    model_source="huggingface",
    dtype="bfloat16",
    max_length=64,
    max_ep_len=100,
    **kwargs,
):
    """
    Load a Decision Transformer model with PEFT adapter and optionally merge weights.

    Args:
        base_model_name: Name or path of the base model
        peft_adapter_path: Path to the PEFT adapter weights
        n_jobs: Number of jobs in JSSP
        n_machines: Number of machines in JSSP
        merge_weights: Whether to merge the adapter weights with the base model
        load_in_4bit: Whether to load the model in 4-bit quantization
        save_merged_path: Path to save the merged model (if merge_weights is True)
        kwargs: Additional parameters for model creation

    Returns:
        Loaded Decision Transformer model
    """
    print(f"Loading base model: {base_model_name}")
    print(f"Loading PEFT adapter: {peft_adapter_path}")
    print(f"Merge weights: {merge_weights}")
    print(f"Load in 4bit: {load_in_4bit}")

    # Create the Decision Transformer model
    dt_model = create_decision_transformer_for_jssp(
        model_name=base_model_name,
        n_jobs=n_jobs,
        n_machines=n_machines,
        load_in_4bit=load_in_4bit,
        model_source=model_source,
        dtype=dtype,
        max_length=max_length,
        max_ep_len=max_ep_len,
        use_peft=False,
        **kwargs
    )

    # Load PEFT adapter
    from peft import PeftModel
    dt_model.base_model = PeftModel.from_pretrained(dt_model.base_model, peft_adapter_path)

    # Merge weights if requested
    if merge_weights:
        print("Merging PEFT adapter weights with base model...")
        dt_model.base_model = dt_model.base_model.merge_and_unload()

        # Save merged model if path is provided
        if save_merged_path:
            os.makedirs(save_merged_path, exist_ok=True)
            print(f"Saving merged model to: {save_merged_path}")
            dt_model.base_model.save_pretrained(save_merged_path)

    # === 关键修复：将整个模型移动到正确设备 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt_model = dt_model.to(device)
    print(f"✅ Model moved to device: {device}")

    # 额外检查：确保state_encoder被移动到正确设备
    if hasattr(dt_model, 'state_encoder') and dt_model.state_encoder is not None:
        dt_model.state_encoder = dt_model.state_encoder.to(device)
        print(f"✅ state_encoder manually moved to device: {device}")

    if custom_modules_path is None:
        candidate = os.path.join(peft_adapter_path, "dt_custom_modules.pt") if os.path.isdir(peft_adapter_path) else None
        if candidate and os.path.exists(candidate):
            custom_modules_path = candidate
        else:
            parent = os.path.dirname(peft_adapter_path)
            candidate2 = os.path.join(parent, "dt_custom_modules.pt")
            if os.path.exists(candidate2):
                custom_modules_path = candidate2

    if custom_modules_path and os.path.exists(custom_modules_path):
        try:
            custom_state = torch.load(custom_modules_path, map_location="cpu", weights_only=True)
        except TypeError:
            custom_state = torch.load(custom_modules_path, map_location="cpu")

        def _safe_load_module(module, state_dict, name):
            import torch
            if module is None or state_dict is None:
                return
            if isinstance(module, torch.nn.Embedding) and "weight" in state_dict:
                w_src = state_dict["weight"]
                w_tgt = module.weight.data
                if w_src.shape[1] == w_tgt.shape[1] and w_src.shape[0] != w_tgt.shape[0]:
                    n = min(w_src.shape[0], w_tgt.shape[0])
                    w_tgt[:n].copy_(w_src[:n])
                    print(f"[LOAD] {name}: embedding weight partial {n}/{w_tgt.shape[0]}")
                    return

            current = module.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in current and hasattr(v, "shape") and current[k].shape == v.shape}
            module.load_state_dict(filtered, strict=False)
            if len(filtered) == 0:
                print(f"[LOAD] {name}: 0/{len(current)} tensors matched (skip)")
            else:
                print(f"[LOAD] {name}: {len(filtered)}/{len(current)} tensors matched")

        for k in ["rtg_proj", "state_proj", "state_proj_norm", "state_encoder", "action_proj", "action_head",
                  "candidate_job_proj", "candidate_context_proj", "candidate_fuse_norm", "candidate_action_head",
                  "scale_context_adapter_norm", "scale_context_adapter_down", "scale_context_adapter_up",
                  "scale_candidate_residual_norm", "scale_candidate_residual_down", "scale_candidate_residual_up",
                  "value_head", "timestep_embedding", "type_embedding", "scale_embedding"]:
            sd = custom_state.get(k, None)
            if sd is None:
                continue
            m = getattr(dt_model, k, None)
            if m is None:
                continue
            _safe_load_module(m, sd, k)
        if getattr(dt_model, "scale_alpha", None) is not None and custom_state.get("scale_alpha", None) is not None:
            with torch.no_grad():
                dt_model.scale_alpha.copy_(custom_state["scale_alpha"].to(device=dt_model.scale_alpha.device, dtype=dt_model.scale_alpha.dtype))
            print("[LOAD] scale_alpha: matched")
        for param_name in ["scale_context_adapter_alpha", "scale_candidate_residual_alpha"]:
            param = getattr(dt_model, param_name, None)
            src = custom_state.get(param_name, None)
            if param is not None and src is not None:
                with torch.no_grad():
                    rows = min(int(src.shape[0]), int(param.shape[0]))
                    cols = min(int(src.shape[1]), int(param.shape[1]))
                    param[:rows, :cols].copy_(src[:rows, :cols].to(device=param.device, dtype=param.dtype))
                print(f"[LOAD] {param_name}: partial {rows}x{cols}")
        meta = custom_state.get("meta", {}) if isinstance(custom_state, dict) else {}
        saved_group_by = meta.get("candidate_harmony_group_by", None) if isinstance(meta, dict) else None
        current_group_by = getattr(dt_model, "candidate_harmony_group_by", None)
        if saved_group_by and current_group_by and str(saved_group_by) != str(current_group_by):
            print(
                f"[LOAD] Warning: checkpoint candidate_harmony_group_by={saved_group_by} "
                f"but current model uses {current_group_by}"
            )
        if getattr(dt_model, "candidate_harmony_mask", None) is not None and custom_state.get("candidate_harmony_mask", None) is not None:
            src = custom_state["candidate_harmony_mask"]
            dst = dt_model.candidate_harmony_mask
            if torch.is_tensor(src) and torch.is_tensor(dst):
                with torch.no_grad():
                    rows = min(int(src.shape[0]), int(dst.shape[0]))
                    cols = min(int(src.shape[1]), int(dst.shape[1]))
                    dst[:rows, :cols].copy_(src[:rows, :cols].to(device=dst.device, dtype=dst.dtype))
                print(f"[LOAD] candidate_harmony_mask: partial {rows}x{cols}")
        for mask_name in [
            "scale_context_harmony_mask", "scale_candidate_harmony_mask",
            "scale_param_mask_gnn_encoder", "scale_param_mask_state_projection",
            "scale_param_mask_lora", "scale_param_mask_candidate_head",
            "scale_param_mask_scale_adapter",
        ]:
            src = custom_state.get(mask_name, None)
            dst = getattr(dt_model, mask_name, None)
            if torch.is_tensor(src) and torch.is_tensor(dst):
                with torch.no_grad():
                    rows = min(int(src.shape[0]), int(dst.shape[0]))
                    cols = min(int(src.shape[1]), int(dst.shape[1]))
                    dst[:rows, :cols].copy_(src[:rows, :cols].to(device=dst.device, dtype=dst.dtype))
                print(f"[LOAD] {mask_name}: partial {rows}x{cols}")
        if isinstance(custom_state, dict) and custom_state.get("ssnc_mask_state") is not None:
            dt_model._loaded_ssnc_mask_state = custom_state["ssnc_mask_state"]
            print("[LOAD] ssnc_mask_state: retained for optional unseen-scale routing")
        print(f"✅ Loaded dt_custom_modules.pt: {custom_modules_path}")

    return dt_model

def test_model_inference(dt_model, n_jobs=10, n_machines=10, seed=42, max_steps=200, max_length=64, target_makespan=None, target_return=1.0, rtg_mode="normalized_budget", temperature=0.0, top_k=0, top_p=0.0, visualize=True, save_gantt=True):
    """
    Test the model by generating a trajectory in the JSSP environment.

    Args:
        dt_model: Decision Transformer model
        n_jobs: Number of jobs in JSSP
        n_machines: Number of machines in JSSP
        seed: Random seed for environment
        max_steps: Maximum number of steps to generate
        target_return: Target return for trajectory generation
        visualize: Whether to visualize the scheduling result as Gantt chart
        save_gantt: Whether to save the Gantt chart to file

    Returns:
        Dictionary containing test results
    """
    # Import JSSP environment
    try:
        from Envs.JSSPEnv.JSSPEnv import JSSPEnv
    except ImportError:
        print("Error importing JSSPEnv. Trying alternative import paths...")
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Envs'))
        from JSSPEnv.JSSPEnv import JSSPEnv

    # Create JSSP environment and explicitly set n_jobs and n_machines
    env = JSSPEnv(n_job_range=[n_jobs, n_jobs], n_mac_range=[n_machines, n_machines], seed=seed, n_jobs=n_jobs, n_machines=n_machines)
    # Reset environment to initialize state
    env.reset()

    # Ensure n_jobs and n_machines are properly set
    if env.n_jobs is None:
        env.n_jobs = n_jobs
    if env.n_machines is None:
        env.n_machines = n_machines

    # Print debug information
    print(f"[DEBUG] Environment initialized: n_jobs={env.n_jobs}, n_machines={env.n_machines}")

    # Print the actual JSSP instance to verify ground truth
    print("\n=== Ground Truth Instance ===")
    for job_id in range(env.n_jobs):
        ops = []
        for op_id in range(env.n_machines):
            machine = int(env.current_instance[job_id, op_id * 2])
            time = int(env.current_instance[job_id, op_id * 2 + 1])
            ops.append(f"M{machine}({time})")
        print(f"Job {job_id}: {' -> '.join(ops)}")

    # Get current instance for OR-Tools solving
    current_instance = env.current_instance

    # Generate trajectory using explicit rollout
    print("\n=== Model Inference ===")
    actions, rewards, makespan = rollout_episode(
        model=dt_model,
        env=env,
        target_makespan=target_makespan,
        target_return=target_return,
        rtg_mode=rtg_mode,
        max_steps=max_steps,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Calculate makespan
    model_makespan = env.get_makespan()

    # Calculate total reward
    total_reward = sum(rewards)

    # === OR-Tools Solving ===
    print("\n=== OR-Tools Solving ===")
    from Envs.JSSPEnv.JSSPEnv import solve_jssp
    import time

    start_time = time.time()
    ortools_result = solve_jssp(current_instance)
    ortools_time = time.time() - start_time

    if isinstance(ortools_result, str) and ortools_result == "infeasible":
        print("❌ OR-Tools found the instance infeasible!")
        ortools_makespan = float('inf')
        ortools_schedule = None
    else:
        ortools_makespan, ortools_schedule = ortools_result
        print(f"✅ OR-Tools solved the instance in {ortools_time:.2f} seconds")
        print(f"✅ OR-Tools makespan: {ortools_makespan}")

    # Print results with OR-Tools comparison
    print(f"\nTest Results:")
    print(f"- Model Makespan: {model_makespan}")
    print(f"- OR-Tools Makespan: {ortools_makespan}")
    print(f"- Makespan Improvement: {model_makespan / ortools_makespan:.3f}x worse")
    print(f"- Total Reward: {total_reward}")
    print(f"- Number of Steps: {len(actions)}")
    print(f"- Scheduling Sequence: {actions}")

    # Visualize the scheduling result as Gantt chart
    if visualize:
        print(f"\nExtracting schedule information from environment...")
        schedule = get_schedule_from_env(env)

        if schedule:
            print("\nSchedule Details:")
            for op in schedule:
                print(f"  Machine {op['machine_id']}: Job {op['job_id']}(Operation {op['operation_id']}) - Start: {op['start_time']}, Duration: {op['duration']}")

            title = f"JSSP Scheduling Result (Makespan: {model_makespan})\nSeed: {seed}, Jobs: {n_jobs}, Machines: {n_machines}"

            if save_gantt:
                gantt_filename = f"jssp_gantt_seed{seed}_jobs{n_jobs}_machines{n_machines}.png"
                save_path = os.path.join(os.getcwd(), gantt_filename)

                print(f"\nGenerating Model Gantt chart...")
                plot_gantt_chart(schedule, n_jobs, n_machines, title=title, save_path=save_path)
            else:
                print(f"\nGenerating Model Gantt chart...")
                plot_gantt_chart(schedule, n_jobs, n_machines, title=title)

        # Visualize OR-Tools scheduling result if available
        if ortools_schedule is not None:
            print(f"\nExtracting OR-Tools schedule information...")
            ortools_schedule_list = get_schedule_from_ortools(ortools_schedule, current_instance)

            if ortools_schedule_list:
                print("\nOR-Tools Schedule Details:")
                for op in ortools_schedule_list:
                    print(f"  Machine {op['machine_id']}: Job {op['job_id']}(Operation {op['operation_id']}) - Start: {op['start_time']}, Duration: {op['duration']}")

                title = f"OR-Tools Optimal Schedule (Makespan: {ortools_makespan})\nSeed: {seed}, Jobs: {n_jobs}, Machines: {n_machines}"

                if save_gantt:
                    gantt_filename = f"jssp_gantt_ortools_seed{seed}_jobs{n_jobs}_machines{n_machines}.png"
                    save_path = os.path.join(os.getcwd(), gantt_filename)

                    print(f"\nGenerating OR-Tools Gantt chart...")
                    plot_gantt_chart(ortools_schedule_list, n_jobs, n_machines, title=title, save_path=save_path)
                else:
                    print(f"\nGenerating OR-Tools Gantt chart...")
                    plot_gantt_chart(ortools_schedule_list, n_jobs, n_machines, title=title)
        else:
            print("\nWarning: No schedule information found in the environment.")

    return {
        "makespan": makespan,
        "total_reward": total_reward,
        "num_steps": len(actions),
        "scheduling_sequence": actions,
        "rewards": rewards,
        "seed": seed,
        "n_jobs": n_jobs,
        "n_machines": n_machines
    }

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Decision Transformer Inference for JSSP problem')

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model_dir', type=str, default=None, help='Checkpoint dir containing adapter_model.bin and dt_custom_modules.pt')
    model_group.add_argument('--peft_adapter', type=str, default=None, help='Path to PEFT adapter weights')
    parser.add_argument('--custom_modules', type=str, default=None, help='Path to dt_custom_modules.pt (optional)')
    parser.add_argument('--base_model', type=str, default='unsloth/Qwen2.5-7B', help='Base model name or path')
    parser.add_argument('--model_source', type=str, default='huggingface', choices=['huggingface', 'modelscope'], help='Model source')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'], help='Model dtype')
    parser.add_argument('--merge_weights', action='store_true', default=False, help='Merge adapter weights with base model')
    parser.add_argument('--save_merged', type=str, default=None, help='Path to save merged model')
    parser.add_argument('--load_in_4bit', action='store_true', default=False, help='Load model in 4-bit quantization')

    # JSSP environment parameters
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs in JSSP (environment/problem size)')
    parser.add_argument('--n_machines', type=int, default=10, help='Number of machines in JSSP (environment/problem size)')

    # Model capacity (max size used when training mixed-scale models)
    parser.add_argument('--model_n_jobs', type=int, default=None, help='Number of jobs used to build the DT model (defaults to --n_jobs)')
    parser.add_argument('--model_n_machines', type=int, default=None, help='Number of machines used to build the DT model (defaults to --n_machines)')
    parser.add_argument('--state_encoder_type', type=str, default='jssp', choices=['jssp', 'graph'], help='State encoder type')
    parser.add_argument('--graph_disable_queue_pos', action='store_true', default=False,
                        help='Disable graph queue-position feature to reduce absolute job-index bias for cross-size generalization')
    parser.add_argument('--use_candidate_action_head', action='store_true', default=False,
                        help='Use per-job candidate action scorer when the checkpoint was trained with it')
    parser.add_argument('--use_candidate_harmony_mask', action='store_true', default=False,
                        help='Create and apply machine-specific candidate-head harmony masks')
    parser.add_argument('--candidate_harmony_group_by', type=str, default='machines',
                        choices=['machines', 'scale'],
                        help='Group candidate harmony masks by problem_n_machines or exact (n_jobs, n_machines) scale')
    parser.add_argument('--use_scale_adapter_expert', action='store_true', default=False,
                        help='Create scale-conditioned adapter/expert residual branches for compatible checkpoints')
    parser.add_argument('--use_scale_context_adapter', action='store_true', default=False,
                        help='Create the scale-conditioned transformer-context adapter branch')
    parser.add_argument('--use_scale_candidate_residual', action='store_true', default=False,
                        help='Create the scale-conditioned candidate residual expert branch')
    parser.add_argument('--use_scale_adapter_harmony_mask', action='store_true', default=False,
                        help='Create HarmoDT-style masks for scale adapter/expert branches')
    parser.add_argument('--scale_adapter_group_by', type=str, default='scale',
                        choices=['machines', 'scale'],
                        help='Group scale adapter/expert masks by machine count or exact (n_jobs, n_machines) scale')
    parser.add_argument('--scale_adapter_bottleneck', type=int, default=256,
                        help='Bottleneck width for scale adapter/expert branches')
    parser.add_argument('--scale_adapter_alpha_init', type=float, default=0.01,
                        help='Initial residual strength for scale adapter/expert branches')
    parser.add_argument('--use_scale_embedding', action='store_true', default=False,
                        help='Create the scale-conditioned state embedding module for checkpoints trained with it')
    parser.add_argument('--use_scale_param_mask', action='store_true', default=False,
                        help='Create scale-specific channel/rank masks for compatible checkpoints')
    parser.add_argument('--mask_gnn_encoder', action='store_true', default=False,
                        help='Create GNN output scale-param mask')
    parser.add_argument('--mask_state_projection', action='store_true', default=False,
                        help='Create state projection scale-param mask')
    parser.add_argument('--mask_lora', action='store_true', default=False,
                        help='Create LoRA rank scale-param mask')
    parser.add_argument('--mask_candidate_head', action='store_true', default=False,
                        help='Create candidate-head scale-param mask')
    parser.add_argument('--mask_scale_adapter', action='store_true', default=False,
                        help='Create scale adapter/expert bottleneck scale-param mask')
    parser.add_argument('--scale_param_mask_group_by', type=str, default='scale',
                        choices=['machines', 'scale'],
                        help='Group scale-param masks by machine count or exact (n_jobs, n_machines) scale')
    parser.add_argument(
        '--unseen_scale_mask_routing',
        type=str,
        default='all_one',
        choices=['all_one', 'nearest', 'neighbor_vote'],
        help=(
            'Inference-only routing for an unseen exact scale. all_one preserves '
            'the original checkpoint behavior; nearest copies the nearest seen '
            'row; neighbor_vote combines nearby seen SSNC mask rows.'
        ),
    )
    parser.add_argument('--unseen_scale_neighbor_k', type=int, default=3,
                        help='Number of seen SSNC domains used for unseen-scale routing')
    parser.add_argument(
        '--unseen_scale_vote_rule',
        type=str,
        default='density_preserving',
        choices=['density_preserving', 'threshold'],
        help='Binary aggregation rule for neighbor-voted mask rows',
    )
    parser.add_argument('--unseen_scale_vote_threshold', type=float, default=0.5,
                        help='Vote threshold used when --unseen_scale_vote_rule=threshold')
    parser.add_argument('--unseen_scale_routing_report', type=str, default=None,
                        help='Optional JSON path for unseen-scale routing diagnostics')

    # Inference parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for environment')
    parser.add_argument('--fixed_seed_stream', action='store_true', default=False, help='Generate instances by repeatedly resetting a single env with the same seed')
    parser.add_argument('--instances_jsonl', type=str, default=None, help='Evaluate fixed instances loaded from jsonl (each line has current_instance)')
    parser.add_argument('--n_rollouts_per_instance', type=int, default=1, help='Number of rollouts per instance')
    parser.add_argument('--hybrid_bon', action='store_true', default=False,
                        help='For best-of-N, run the first rollout greedily and use sampling parameters for the remaining rollouts')
    parser.add_argument('--pick', type=str, default='best', choices=['best', 'mean'], help='Aggregation over multiple rollouts')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum steps for trajectory generation')
    parser.add_argument('--max_length', type=int, default=64, help='Maximum history length for DT context')
    parser.add_argument('--max_ep_len', type=int, default=100, help='Maximum episode length for timestep embedding')
    parser.add_argument('--target_makespan', type=float, default=None, help='Target makespan budget for RTG construction')
    parser.add_argument('--target_return', type=float, default=1.0, help='RTG strength. For budget-based modes it scales the target makespan.')
    parser.add_argument('--rtg_mode', type=str, default='normalized_budget', choices=['constant', 'budget', 'normalized_budget'], help='RTG construction mode')
    parser.add_argument(
        '--state_tensorization_mode',
        type=str,
        default=LEGACY_TRAIN_COMPATIBLE,
        choices=VALID_TENSORIZATION_MODES,
        help=(
            'Graph-state layout. legacy_train_compatible evaluates existing '
            'checkpoints on their training adjacency layout; canonical_v2 '
            'also remaps scheduled-machine node ids for new training.'
        ),
    )

    parser.add_argument('--n_instances', type=int, default=100, help='Number of instances to evaluate')
    parser.add_argument('--allow_duplicate_instances', action='store_true', default=False, help='Allow duplicate instances (disable uniqueness filter)')
    parser.add_argument('--evaluate_multiple', action='store_true', default=True, help='Evaluate multiple instances')
    parser.add_argument('--no-evaluate_multiple', dest='evaluate_multiple', action='store_false', help='Evaluate only one instance')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable per-step debug prints')
    parser.add_argument('--print_instance_hash', action='store_true', default=False, help='Print per-instance hash to detect duplicates')

    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', default=True, help='Whether to visualize the scheduling result as Gantt chart')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false', help='Disable visualization')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling (0.0=greedy, higher=more exploration)')
    parser.add_argument('--top_k', type=int, default=0, help='If >0, sample only from top-k legal actions when temperature > 0')
    parser.add_argument('--top_p', type=float, default=0.0,
                        help='Nucleus sampling threshold in [0,1] when temperature > 0; 0 or 1 disables truncation. Applied after top-k.')
    parser.add_argument('--no-save-gantt', dest='save_gantt', action='store_false', help='Disable saving Gantt chart')
    parser.add_argument('--save_selected_jsonl', type=str, default=None, help='Append selected rollout trajectory per instance to jsonl')

    return parser.parse_args()

def evaluate_multiple_instances(
    dt_model,
    n_instances=100,
    n_jobs=10,
    n_machines=10,
    seed=42,
    max_steps=200,
    max_length=64,
    target_makespan=None,
    target_return=1.0,
    rtg_mode="normalized_budget",
    temperature=0.0,
    top_k=0,
    top_p=0.0,
    visualize=False,
    save_gantt=False,
    debug=False,
    ensure_unique_instances=True,
    max_unique_tries=10000,
    fixed_seed_stream=False,
    n_rollouts_per_instance=1,
    hybrid_bon=False,
    pick='best',
    print_instance_hash=False,
    save_selected_jsonl=None,
):
    """
    Evaluate the model on multiple JSSP instances and generate relative gap plots.

    Args:
        dt_model: Decision Transformer model
        n_instances: Number of instances to evaluate
        n_jobs: Number of jobs in JSSP
        n_machines: Number of machines in JSSP
        seed: Base random seed for environment
        max_steps: Maximum number of steps per instance
        target_return: Target return for trajectory generation
        visualize: Whether to visualize each instance's scheduling result
        save_gantt: Whether to save Gantt charts for each instance

    Returns:
        Dictionary containing evaluation results and relative gaps
    """
    # Import JSSP environment
    try:
        from Envs.JSSPEnv.JSSPEnv import JSSPEnv, solve_jssp
    except ImportError:
        print("Error importing JSSPEnv. Trying alternative import paths...")
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Envs'))
        from JSSPEnv.JSSPEnv import JSSPEnv, solve_jssp

    results = []
    relative_gaps = []
    model_makespans = []
    optimal_makespans = []
    instance_indices = []
    solver_times = []
    model_inference_times = []

    full_model_makespans = []
    full_ortools_makespans = []
    full_seeds = []
    unique_instances = set()
    instance_hash_counts = {}

    print(f"\n=== Evaluating {n_instances} JSSP Instances ===")

    instance_idx = 0
    tries = 0
    env = None
    if fixed_seed_stream:
        env = JSSPEnv(
            n_job_range=[n_jobs, n_jobs],
            n_mac_range=[n_machines, n_machines],
            seed=int(seed),
            n_jobs=n_jobs,
            n_machines=n_machines,
        )

    while instance_idx < n_instances:
        if tries >= int(max_unique_tries):
            raise RuntimeError(f"Unable to sample {n_instances} instances after {max_unique_tries} tries.")
        tries += 1

        if fixed_seed_stream:
            current_seed = int(seed)
            env.reset()
        else:
            current_seed = int(seed + tries - 1)
            env = JSSPEnv(
                n_job_range=[n_jobs, n_jobs],
                n_mac_range=[n_machines, n_machines],
                seed=current_seed,
                n_jobs=n_jobs,
                n_machines=n_machines,
            )
            env.reset()

        if env.n_jobs is None:
            env.n_jobs = n_jobs
        if env.n_machines is None:
            env.n_machines = n_machines

        current_instance = env.current_instance
        if ensure_unique_instances and isinstance(current_instance, np.ndarray):
            key = current_instance.tobytes()
            if key in unique_instances:
                continue
            unique_instances.add(key)

        instance_hash = None
        if isinstance(current_instance, np.ndarray):
            try:
                import hashlib
                instance_hash = hashlib.md5(current_instance.tobytes()).hexdigest()[:10]
            except Exception:
                instance_hash = None

        if instance_hash is not None:
            instance_hash_counts[instance_hash] = instance_hash_counts.get(instance_hash, 0) + 1

        label = f"seed={current_seed}" if not fixed_seed_stream else f"seed={current_seed}, reset_idx={tries}"
        print(f"\nInstance {instance_idx + 1}/{n_instances} ({label}):")
        if print_instance_hash and instance_hash is not None:
            dup_tag = "DUP" if instance_hash_counts.get(instance_hash, 0) > 1 else "NEW"
            print(f"Instance hash: {instance_hash} ({dup_tag})")
        print("=" * 60)

        # --- 强制同一实例做多次rollout（best-of-N公平特性） ---
        # 将 current_instance 拆分为 processing_times 与 machine_order，并在每次rollout前恢复到环境
        if isinstance(current_instance, np.ndarray):
            try:
                pj = int(current_instance.shape[0])
                pm = int(current_instance.shape[1] // 2)
                processing_times_arr = np.zeros((pj, pm), dtype=np.int32)
                machine_order_arr = np.zeros((pj, pm), dtype=np.int32)
                for j in range(pj):
                    for op in range(pm):
                        machine_order_arr[j, op] = int(current_instance[j, 2 * op])
                        processing_times_arr[j, op] = int(current_instance[j, 2 * op + 1])
            except Exception as _:
                processing_times_arr = None
                machine_order_arr = None
        else:
            processing_times_arr = None
            machine_order_arr = None

        rollout_times = []
        rollout_makespans = []
        rollout_rewards = []
        rollout_actions = []
        rollout_trajs = []
        for rollout_idx in range(max(1, int(n_rollouts_per_instance))):
            rollout_temperature = 0.0 if hybrid_bon and rollout_idx == 0 else temperature
            rollout_top_k = 1 if hybrid_bon and rollout_idx == 0 else top_k
            rollout_top_p = 0.0 if hybrid_bon and rollout_idx == 0 else top_p
            # 恢复同一实例到环境并reset，确保best-of-N在同一实例上比较
            if processing_times_arr is not None and machine_order_arr is not None and hasattr(env, "set_external_instance"):
                try:
                    env.set_external_instance(processing_times_arr, machine_order_arr)
                except Exception:
                    pass
            try:
                env.reset()
            except Exception:
                pass

            start_time = time.time()
            if save_selected_jsonl:
                traj, mksp = rollout_episode_with_trajectory(
                    model=dt_model,
                    env=env,
                    max_steps=max_steps,
                    max_length=max_length,
                    target_makespan=target_makespan,
                    target_return=target_return,
                    rtg_mode=rtg_mode,
                    temperature=rollout_temperature,
                    top_k=rollout_top_k,
                    top_p=rollout_top_p,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    debug=debug,
                )
                acts = traj.get("actions", [])
                rews = traj.get("rewards", [])
            else:
                acts, rews, mksp = rollout_episode(
                    model=dt_model,
                    env=env,
                    max_steps=max_steps,
                    max_length=max_length,
                    target_makespan=target_makespan,
                    target_return=target_return,
                    rtg_mode=rtg_mode,
                    temperature=rollout_temperature,
                    top_k=rollout_top_k,
                    top_p=rollout_top_p,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    debug=debug,
                )
                traj = None
            rollout_times.append(time.time() - start_time)
            rollout_makespans.append(float(mksp))
            rollout_rewards.append(sum(rews))
            rollout_actions.append(acts)
            rollout_trajs.append(traj)

        if pick == 'mean':
            model_inference_time = float(sum(rollout_times) / len(rollout_times))
            model_makespan = float(np.mean(rollout_makespans))
            rewards = [float(r) for r in rollout_rewards]
            idx_selected = int(np.argmin([abs(r) for r in rollout_rewards]))
            actions = rollout_actions[idx_selected]
        else:
            model_inference_time = float(sum(rollout_times))
            finite_mask = [np.isfinite(m) for m in rollout_makespans]
            if any(finite_mask):
                idx_selected = int(np.argmin([m if np.isfinite(m) else float('inf') for m in rollout_makespans]))
                model_makespan = float(rollout_makespans[idx_selected])
                rewards = [float(rollout_rewards[idx_selected])]
                actions = rollout_actions[idx_selected]
            else:
                model_makespan = float('nan')
                rewards = []
                actions = []
                idx_selected = None
        print(f"Model inference time: {model_inference_time:.4f} seconds")

        if save_selected_jsonl and idx_selected is not None:
            selected_traj = rollout_trajs[idx_selected]
            if isinstance(selected_traj, dict):
                metrics = selected_traj.get("metrics", {}) if isinstance(selected_traj.get("metrics", {}), dict) else {}
                metrics["seed"] = int(current_seed)
                metrics["instance_idx"] = int(instance_idx)
                metrics["instance_hash"] = instance_hash
                metrics["pick"] = str(pick)
                metrics["n_rollouts_per_instance"] = int(n_rollouts_per_instance)
                metrics["hybrid_bon"] = bool(hybrid_bon)
                selected_traj["metrics"] = metrics
                with open(save_selected_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(selected_traj, ensure_ascii=False) + "\n")

        expected_operations = int(n_jobs) * int(n_machines)
        scheduled_actions = sum(1 for a in actions if int(a) >= 0)
        episode_completed = scheduled_actions >= expected_operations or _env_all_jobs_completed(env)
        if not episode_completed:
            if np.isfinite(model_makespan):
                print(
                    f"⚠️ Model rollout incomplete ({scheduled_actions}/{expected_operations} operations); "
                    "makespan marked as NaN."
                )
            model_makespan = float("nan")
        elif not np.isfinite(model_makespan):
            model_makespan = float(env.get_makespan())

        full_model_makespans.append(float(model_makespan))
        full_seeds.append(current_seed)

        # Calculate total reward
        total_reward = sum(rewards)

        # OR-Tools Solving
        try:
            solver_start_time = time.time()
            ortools_result = solve_jssp(current_instance)
            solver_time = time.time() - solver_start_time
            print(f"OR-Tools solver time: {solver_time:.4f} seconds")

            if isinstance(ortools_result, str) and ortools_result == "infeasible":
                print("❌ OR-Tools found the instance infeasible!")
                ortools_makespan = float('inf')
            else:
                ortools_makespan, _ = ortools_result
                print(f"✅ OR-Tools makespan: {ortools_makespan}")
        except Exception as e:
            print(f"❌ OR-Tools Solver Failed: {e}")
            ortools_makespan = float('inf')
            solver_time = float('inf')
        full_ortools_makespans.append(float(ortools_makespan) if ortools_makespan != float("inf") else float("nan"))

        # Calculate relative gap
        if np.isfinite(model_makespan) and ortools_makespan != float('inf') and ortools_makespan > 0:
            relative_gap = (model_makespan - ortools_makespan) / ortools_makespan * 100
            print(f"Relative Gap: {relative_gap:.2f}%")
        else:
            relative_gap = float('nan')
            print(f"Relative Gap: N/A (incomplete rollout or OR-Tools failed)")

        # Store results (convert numpy types to Python native types for JSON serialization)
        results.append({
            "instance_idx": int(instance_idx),
            "seed": int(current_seed),
            "instance_hash": instance_hash,
            "model_makespan": int(model_makespan) if isinstance(model_makespan, (int, np.integer)) else float(model_makespan),
            "ortools_makespan": int(ortools_makespan) if isinstance(ortools_makespan, (int, np.integer)) else float(ortools_makespan),
            "relative_gap": float(relative_gap) if not np.isnan(relative_gap) else None,
            "total_reward": float(total_reward) if isinstance(total_reward, (int, float)) else None,
            "num_steps": int(len(actions)),
            "solver_time": float(solver_time) if solver_time != float('inf') else None,
            "model_inference_time": float(model_inference_time),
            "actions": [int(a) for a in actions],
            "rewards": [float(r) for r in rewards]
        })

        # Collect data for plotting
        if not np.isnan(relative_gap):
            relative_gaps.append(relative_gap)
            model_makespans.append(model_makespan)
            optimal_makespans.append(ortools_makespan)
            instance_indices.append(instance_idx + 1)
            solver_times.append(solver_time if solver_time != float('inf') else float("nan"))
            model_inference_times.append(model_inference_time)

        instance_idx += 1

    # Generate and save relative gap plots
    print(f"\n=== Generating Evaluation Results ===")

    # 1. Generate relative gap distribution plot
    plt.figure(figsize=(12, 6))
    plt.hist(relative_gaps, bins=20, alpha=0.7, color='blue')
    plt.title(f'Relative Gap Distribution ({len(relative_gaps)}/{n_instances} Valid Instances)')
    plt.xlabel('Relative Gap (%)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Add mean and median lines
    if relative_gaps:
        mean_gap = np.mean(relative_gaps)
        median_gap = np.median(relative_gaps)
        plt.axvline(mean_gap, color='red', linestyle='--', label=f'Mean: {mean_gap:.2f}%')
        plt.axvline(median_gap, color='green', linestyle='--', label=f'Median: {median_gap:.2f}%')
        plt.legend()

    # Save the histogram
    gap_hist_path = os.path.join(os.getcwd(), f'relative_gap_distribution_{n_instances}instances.png')
    plt.savefig(gap_hist_path, dpi=150, bbox_inches='tight')
    print(f"Saved relative gap distribution to: {gap_hist_path}")
    plt.close()

    # 2. Generate relative gap vs instance index plot
    plt.figure(figsize=(12, 6))
    if relative_gaps:
        plt.scatter(instance_indices, relative_gaps, alpha=0.7, color='blue')
        plt.plot(instance_indices, relative_gaps, 'b-', alpha=0.3)
        plt.title(f'Relative Gap vs Instance Index ({len(relative_gaps)}/{n_instances} Valid Instances)')
        plt.xlabel('Instance Index')
        plt.ylabel('Relative Gap (%)')
        plt.grid(True)

        # Add mean line
        if relative_gaps:
            mean_gap = np.mean(relative_gaps)
            plt.axhline(mean_gap, color='red', linestyle='--', label=f'Mean: {mean_gap:.2f}%')
            plt.legend()

        # Save the plot
        gap_vs_instance_path = os.path.join(os.getcwd(), f'relative_gap_vs_instance_{n_instances}instances.png')
        plt.savefig(gap_vs_instance_path, dpi=150, bbox_inches='tight')
        print(f"Saved relative gap vs instance plot to: {gap_vs_instance_path}")
    plt.close()

    # 3. Generate makespan comparison plot
    plt.figure(figsize=(12, 6))
    if model_makespans and optimal_makespans:
        plt.scatter(optimal_makespans, model_makespans, alpha=0.7, color='blue')

        # Add ideal line (model makespan = optimal makespan)
        max_makespan = max(max(model_makespans), max(optimal_makespans))
        plt.plot([0, max_makespan], [0, max_makespan], 'r--', label='Ideal (Model = Optimal)')

        plt.title(f'Model Makespan vs Optimal Makespan ({len(relative_gaps)}/{n_instances} Valid Instances)')
        plt.xlabel('Optimal Makespan (OR-Tools)')
        plt.ylabel('Model Makespan')
        plt.grid(True)
        plt.legend()

        # Save the plot
        makespan_compare_path = os.path.join(os.getcwd(), f'makespan_comparison_{n_instances}instances.png')
        plt.savefig(makespan_compare_path, dpi=150, bbox_inches='tight')
        print(f"Saved makespan comparison plot to: {makespan_compare_path}")
    plt.close()

    # 4. Generate time comparison plot (OR-Tools vs Model)
    plt.figure(figsize=(12, 6))
    if solver_times and model_inference_times:
        x = np.array(instance_indices, dtype=np.int32)
        y_solver = np.array(solver_times, dtype=np.float32)
        y_model = np.array(model_inference_times, dtype=np.float32)

        plt.plot(x, y_solver, 'b-', label='OR-Tools Solver Time', linewidth=2)
        plt.plot(x, y_model, 'r-', label='Model Inference Time', linewidth=2)

        plt.title(f'Solver Time vs Model Inference Time ({len(relative_gaps)}/{n_instances} Valid Instances)')
        plt.xlabel('Instance Index')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        y_solver_f = y_solver[np.isfinite(y_solver)]
        y_model_f = y_model[np.isfinite(y_model)]
        if y_solver_f.size > 0:
            avg_solver_time = float(np.mean(y_solver_f))
            plt.axhline(avg_solver_time, color='blue', linestyle='--', alpha=0.7, label=f'Avg Solver: {avg_solver_time:.4f}s')
        if y_model_f.size > 0:
            avg_model_time = float(np.mean(y_model_f))
            plt.axhline(avg_model_time, color='red', linestyle='--', alpha=0.7, label=f'Avg Model: {avg_model_time:.4f}s')

        plt.legend(fontsize=12)

        time_compare_path = os.path.join(os.getcwd(), f'time_comparison_{n_instances}instances.png')
        plt.savefig(time_compare_path, dpi=150, bbox_inches='tight')
        print(f"Saved time comparison plot to: {time_compare_path}")
    plt.close()

    # 5. Generate makespan lines per instance (Model vs OR-Tools)
    plt.figure(figsize=(14, 6))
    x_all = np.arange(1, n_instances + 1, dtype=np.int32)
    y_model_all = np.array(full_model_makespans, dtype=np.float32)
    y_ortools_all = np.array(full_ortools_makespans, dtype=np.float32)
    plt.plot(x_all, y_model_all, color='tab:blue', linewidth=2, marker='o', markersize=3, label='Model Makespan')
    plt.plot(x_all, y_ortools_all, color='tab:red', linewidth=2, marker='o', markersize=3, label='OR-Tools Makespan')
    plt.title(f'Makespan per Instance (Unique={ensure_unique_instances})')
    plt.xlabel('Instance Index (1..N)')
    plt.ylabel('Makespan')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    makespan_lines_path = os.path.join(os.getcwd(), f'makespan_lines_{n_instances}instances.png')
    plt.savefig(makespan_lines_path, dpi=150, bbox_inches='tight')
    print(f"Saved makespan lines plot to: {makespan_lines_path}")
    plt.close()

    # Calculate and print summary statistics
    if relative_gaps:
        mean_gap = np.mean(relative_gaps)
        median_gap = np.median(relative_gaps)
        min_gap = np.min(relative_gaps)
        max_gap = np.max(relative_gaps)
        std_gap = np.std(relative_gaps)

        print(f"\n=== Evaluation Summary ({len(relative_gaps)}/{n_instances} Valid Instances) ===")
        print(f"Mean Relative Gap: {mean_gap:.2f}%")
        print(f"Median Relative Gap: {median_gap:.2f}%")
        print(f"Min Relative Gap: {min_gap:.2f}%")
        print(f"Max Relative Gap: {max_gap:.2f}%")
        print(f"Standard Deviation: {std_gap:.2f}%")
        print(f"Average Model Makespan: {np.mean(model_makespans):.2f}")
        print(f"Average Optimal Makespan: {np.mean(optimal_makespans):.2f}")
        print(f"Average OR-Tools Solver Time: {np.mean(solver_times):.4f} seconds")
        print(f"Average Model Inference Time: {np.mean(model_inference_times):.4f} seconds")
        print(f"Speedup Factor: {np.mean(solver_times)/np.mean(model_inference_times):.2f}x")
    else:
        print(f"\n⚠️  No valid instances to evaluate.")

    # Save evaluation results to JSON
    results_data = {
        "n_instances": n_instances,
        "n_valid_instances": len(relative_gaps),
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "state_tensorization_mode": get_state_tensorization_mode(),
        "settings": {
            "rtg_mode": str(rtg_mode),
            "target_return": float(target_return),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "n_rollouts_per_instance": int(n_rollouts_per_instance),
            "hybrid_bon": bool(hybrid_bon),
            "pick": str(pick),
        },
        "seeds": [int(s) for s in full_seeds],
        "instance_hash_counts": instance_hash_counts,
        "summary": {
            "mean_gap": float(mean_gap) if relative_gaps else None,
            "median_gap": float(median_gap) if relative_gaps else None,
            "min_gap": float(min_gap) if relative_gaps else None,
            "max_gap": float(max_gap) if relative_gaps else None,
            "std_gap": float(std_gap) if relative_gaps else None,
            "avg_model_makespan": float(np.mean(model_makespans)) if model_makespans else None,
            "avg_optimal_makespan": float(np.mean(optimal_makespans)) if optimal_makespans else None,
            "avg_solver_time": float(np.mean(solver_times)) if solver_times else None,
            "avg_model_time": float(np.mean(model_inference_times)) if model_inference_times else None,
            "speedup_factor": float(np.mean(solver_times)/np.mean(model_inference_times)) if solver_times and model_inference_times else None
        },
        "makespans": {
            "model": [float(x) for x in full_model_makespans],
            "ortools": [float(x) for x in full_ortools_makespans],
        },
        "instances": results
    }

    results_path = os.path.join(os.getcwd(), f'jssp_evaluation_results_{n_instances}instances.json')
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nEvaluation results saved to: {results_path}")

    return results_data


def _load_instances_from_jsonl(instances_jsonl: str):
    path = str(instances_jsonl).strip()
    if not path:
        return []
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                continue
            if "current_instance" not in obj:
                continue
            arr = np.asarray(obj["current_instance"], dtype=np.int32)
            name = str(obj.get("name", f"instance_{len(out)}"))
            n_jobs = int(obj.get("n_jobs", arr.shape[0]))
            n_machines = int(obj.get("n_machines", arr.shape[1] // 2 if arr.ndim == 2 else 0))
            if arr.shape != (n_jobs, 2 * n_machines):
                raise ValueError(f"{name}: current_instance has shape {arr.shape}, expected {(n_jobs, 2 * n_machines)}")
            item = {"name": name, "n_jobs": n_jobs, "n_machines": n_machines, "current_instance": arr}
            for key in (
                "optimal_makespan",
                "ortools_makespan",
                "best_known_makespan",
                "reference_makespan",
                "target_makespan",
                "best_known_upper_bound",
                "best_known_lower_bound",
                "bks_status",
            ):
                if key in obj:
                    item[key] = obj.get(key)
            out.append(item)
    return out


def _reference_makespan_from_item(item):
    for key in (
        "reference_makespan",
        "optimal_makespan",
        "best_known_makespan",
        "ortools_makespan",
        "best_known_upper_bound",
        "target_makespan",
    ):
        value = item.get(key, None)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value, key
    return None, None


def evaluate_instances_from_jsonl(
    dt_model,
    instances_jsonl,
    n_jobs=10,
    n_machines=10,
    max_steps=200,
    max_length=64,
    target_makespan=None,
    target_return=1.0,
    rtg_mode="normalized_budget",
    temperature=0.0,
    top_k=0,
    top_p=0.0,
    visualize=False,
    save_gantt=False,
    debug=False,
    n_rollouts_per_instance=1,
    hybrid_bon=False,
    pick="best",
    print_instance_hash=False,
    save_selected_jsonl=None,
):
    try:
        from Envs.JSSPEnv.JSSPEnv import JSSPEnv, solve_jssp
    except ImportError:
        print("Error importing JSSPEnv. Trying alternative import paths...")
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Envs'))
        from JSSPEnv.JSSPEnv import JSSPEnv, solve_jssp

    instances = _load_instances_from_jsonl(instances_jsonl)
    if not instances:
        raise ValueError(f"No instances loaded from: {instances_jsonl}")

    for it in instances:
        if int(it["n_jobs"]) != int(n_jobs) or int(it["n_machines"]) != int(n_machines):
            raise ValueError(
                f"Instance {it['name']} is {it['n_jobs']}x{it['n_machines']} but args are {n_jobs}x{n_machines}. "
                f"Model/env must match the instance size."
            )

    results = []
    relative_gaps = []
    model_makespans = []
    optimal_makespans = []
    instance_indices = []
    solver_times = []
    model_inference_times = []

    full_model_makespans = []
    full_ortools_makespans = []
    full_names = []
    instance_hash_counts = {}

    print(f"\n=== Evaluating {len(instances)} Instances from JSONL ===")
    for instance_idx, item in enumerate(instances):
        current_instance = item["current_instance"]
        name = str(item.get("name", f"instance_{instance_idx}"))

        instance_hash = None
        try:
            instance_hash = hashlib.md5(current_instance.tobytes()).hexdigest()[:10]
        except Exception:
            instance_hash = None

        if instance_hash is not None:
            instance_hash_counts[instance_hash] = instance_hash_counts.get(instance_hash, 0) + 1

        label = f"name={name}"
        print(f"\nInstance {instance_idx + 1}/{len(instances)} ({label}):")
        if print_instance_hash and instance_hash is not None:
            dup_tag = "DUP" if instance_hash_counts.get(instance_hash, 0) > 1 else "NEW"
            print(f"Instance hash: {instance_hash} ({dup_tag})")
        print("=" * 60)

        env = JSSPEnv(
            n_job_range=[n_jobs, n_jobs],
            n_mac_range=[n_machines, n_machines],
            seed=0,
            external_instance=current_instance,
            n_jobs=n_jobs,
            n_machines=n_machines,
        )

        processing_times_arr = None
        machine_order_arr = None
        if isinstance(current_instance, np.ndarray):
            try:
                pj = int(current_instance.shape[0])
                pm = int(current_instance.shape[1] // 2)
                processing_times_arr = np.zeros((pj, pm), dtype=np.int32)
                machine_order_arr = np.zeros((pj, pm), dtype=np.int32)
                for j in range(pj):
                    for op in range(pm):
                        machine_order_arr[j, op] = int(current_instance[j, 2 * op])
                        processing_times_arr[j, op] = int(current_instance[j, 2 * op + 1])
            except Exception:
                processing_times_arr = None
                machine_order_arr = None

        rollout_times = []
        rollout_makespans = []
        rollout_rewards = []
        rollout_actions = []
        rollout_trajs = []
        for rollout_idx in range(max(1, int(n_rollouts_per_instance))):
            rollout_temperature = 0.0 if hybrid_bon and rollout_idx == 0 else temperature
            rollout_top_k = 1 if hybrid_bon and rollout_idx == 0 else top_k
            rollout_top_p = 0.0 if hybrid_bon and rollout_idx == 0 else top_p
            if processing_times_arr is not None and machine_order_arr is not None and hasattr(env, "set_external_instance"):
                try:
                    env.set_external_instance(processing_times_arr, machine_order_arr)
                except Exception:
                    pass
            try:
                env.reset()
            except Exception:
                pass

            start_time = time.time()
            if save_selected_jsonl:
                traj, mksp = rollout_episode_with_trajectory(
                    model=dt_model,
                    env=env,
                    max_steps=max_steps,
                    max_length=max_length,
                    target_makespan=target_makespan,
                    target_return=target_return,
                    rtg_mode=rtg_mode,
                    temperature=rollout_temperature,
                    top_k=rollout_top_k,
                    top_p=rollout_top_p,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    debug=debug,
                )
                acts = traj.get("actions", [])
                rews = traj.get("rewards", [])
            else:
                acts, rews, mksp = rollout_episode(
                    model=dt_model,
                    env=env,
                    max_steps=max_steps,
                    max_length=max_length,
                    target_makespan=target_makespan,
                    target_return=target_return,
                    rtg_mode=rtg_mode,
                    temperature=rollout_temperature,
                    top_k=rollout_top_k,
                    top_p=rollout_top_p,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    debug=debug,
                )
                traj = None
            rollout_times.append(time.time() - start_time)
            rollout_makespans.append(float(mksp))
            rollout_rewards.append(sum(rews))
            rollout_actions.append(acts)
            rollout_trajs.append(traj)

        if pick == "mean":
            model_inference_time = float(sum(rollout_times) / len(rollout_times))
            model_makespan = float(np.mean(rollout_makespans))
            rewards = [float(r) for r in rollout_rewards]
            idx_selected = int(np.argmin([abs(r) for r in rollout_rewards]))
            actions = rollout_actions[idx_selected]
        else:
            model_inference_time = float(sum(rollout_times))
            finite_mask = [np.isfinite(m) for m in rollout_makespans]
            if any(finite_mask):
                idx_selected = int(np.argmin([m if np.isfinite(m) else float("inf") for m in rollout_makespans]))
                model_makespan = float(rollout_makespans[idx_selected])
                rewards = [float(rollout_rewards[idx_selected])]
                actions = rollout_actions[idx_selected]
            else:
                model_makespan = float("nan")
                rewards = []
                actions = []
                idx_selected = None

        print(f"Model inference time: {model_inference_time:.4f} seconds")

        if save_selected_jsonl and idx_selected is not None:
            selected_traj = rollout_trajs[idx_selected]
            if isinstance(selected_traj, dict):
                metrics = selected_traj.get("metrics", {}) if isinstance(selected_traj.get("metrics", {}), dict) else {}
                metrics["instance_idx"] = int(instance_idx)
                metrics["instance_name"] = str(name)
                metrics["instance_hash"] = instance_hash
                metrics["pick"] = str(pick)
                metrics["n_rollouts_per_instance"] = int(n_rollouts_per_instance)
                metrics["hybrid_bon"] = bool(hybrid_bon)
                selected_traj["metrics"] = metrics
                with open(save_selected_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(selected_traj, ensure_ascii=False) + "\n")

        expected_operations = int(n_jobs) * int(n_machines)
        scheduled_actions = sum(1 for a in actions if int(a) >= 0)
        episode_completed = scheduled_actions >= expected_operations or _env_all_jobs_completed(env)
        if not episode_completed:
            if np.isfinite(model_makespan):
                print(
                    f"⚠️ Model rollout incomplete ({scheduled_actions}/{expected_operations} operations); "
                    "makespan marked as NaN."
                )
            model_makespan = float("nan")
        elif not np.isfinite(model_makespan):
            model_makespan = float(env.get_makespan())

        full_model_makespans.append(float(model_makespan))
        full_names.append(name)

        total_reward = sum(rewards)

        reference_makespan, reference_source = _reference_makespan_from_item(item)
        if reference_makespan is not None:
            ortools_makespan = float(reference_makespan)
            solver_time = 0.0
            print(f"✅ Reference makespan ({reference_source}): {ortools_makespan}")
        else:
            try:
                solver_start_time = time.time()
                ortools_result = solve_jssp(current_instance)
                solver_time = time.time() - solver_start_time
                print(f"OR-Tools solver time: {solver_time:.4f} seconds")

                if isinstance(ortools_result, str) and ortools_result == "infeasible":
                    print("❌ OR-Tools found the instance infeasible!")
                    ortools_makespan = float("inf")
                    reference_source = "ortools_infeasible"
                else:
                    ortools_makespan, _ = ortools_result
                    reference_source = "ortools"
                    print(f"✅ OR-Tools makespan: {ortools_makespan}")
            except Exception as e:
                print(f"❌ OR-Tools Solver Failed: {e}")
                ortools_makespan = float("inf")
                solver_time = float("inf")
                reference_source = "ortools_failed"

        full_ortools_makespans.append(float(ortools_makespan))

        if ortools_makespan == float("inf") or not np.isfinite(ortools_makespan) or ortools_makespan <= 0:
            relative_gap = float("nan")
        else:
            relative_gap = ((float(model_makespan) - float(ortools_makespan)) / float(ortools_makespan)) * 100.0

        print(f"Model makespan: {model_makespan}")
        print(f"Optimal makespan: {ortools_makespan}")
        print(f"Relative gap: {relative_gap:.2f}%")

        results.append(
            {
                "instance_name": name,
                "instance_hash": instance_hash,
                "model_makespan": float(model_makespan),
                "optimal_makespan": float(ortools_makespan),
                "reference_source": str(reference_source),
                "relative_gap": float(relative_gap) if np.isfinite(relative_gap) else None,
                "model_inference_time": float(model_inference_time),
                "ortools_solver_time": float(solver_time),
                "total_reward": float(total_reward),
                "actions": actions,
            }
        )

        if np.isfinite(relative_gap):
            relative_gaps.append(relative_gap)
            model_makespans.append(float(model_makespan))
            optimal_makespans.append(float(ortools_makespan))
            instance_indices.append(int(instance_idx))
            solver_times.append(float(solver_time))
            model_inference_times.append(float(model_inference_time))

    valid_n = len(relative_gaps)
    if valid_n > 0:
        mean_gap = float(np.mean(relative_gaps))
        median_gap = float(np.median(relative_gaps))
        min_gap = float(np.min(relative_gaps))
        max_gap = float(np.max(relative_gaps))
        std_gap = float(np.std(relative_gaps))
        avg_model_mksp = float(np.mean(model_makespans))
        avg_opt_mksp = float(np.mean(optimal_makespans))
        avg_solver_time = float(np.mean(solver_times))
        avg_infer_time = float(np.mean(model_inference_times))
    else:
        mean_gap = median_gap = min_gap = max_gap = std_gap = float("nan")
        avg_model_mksp = avg_opt_mksp = avg_solver_time = avg_infer_time = float("nan")

    print(f"\n=== Evaluation Summary ({valid_n}/{len(instances)} Valid Instances) ===")
    print(f"Mean Relative Gap: {mean_gap:.2f}%")
    print(f"Median Relative Gap: {median_gap:.2f}%")
    print(f"Min Relative Gap: {min_gap:.2f}%")
    print(f"Max Relative Gap: {max_gap:.2f}%")
    print(f"Standard Deviation: {std_gap:.2f}%")
    print(f"Average Model Makespan: {avg_model_mksp:.2f}")
    print(f"Average Optimal Makespan: {avg_opt_mksp:.2f}")
    print(f"Average OR-Tools Solver Time: {avg_solver_time:.4f} seconds")
    print(f"Average Model Inference Time: {avg_infer_time:.4f} seconds")

    base = os.path.splitext(os.path.basename(str(instances_jsonl)))[0]
    results_path = os.path.join(os.getcwd(), f"jssp_evaluation_results_{base}.json")
    results_data = {
        "dataset": {
            "source": str(instances_jsonl),
            "n_jobs": int(n_jobs),
            "n_machines": int(n_machines),
            "n_instances": int(len(instances)),
        },
        "settings": {
            "rtg_mode": str(rtg_mode),
            "target_return": float(target_return),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "n_rollouts_per_instance": int(n_rollouts_per_instance),
            "hybrid_bon": bool(hybrid_bon),
            "pick": str(pick),
            "max_steps": int(max_steps),
            "max_length": int(max_length),
            "state_tensorization_mode": get_state_tensorization_mode(),
            "unseen_scale_routing": getattr(
                dt_model,
                "_unseen_scale_routing_report",
                {"mode": "all_one", "status": "original_checkpoint_row_unchanged"},
            ),
        },
        "summary": {
            "valid_instances": int(valid_n),
            "mean_gap": mean_gap,
            "median_gap": median_gap,
            "min_gap": min_gap,
            "max_gap": max_gap,
            "std_gap": std_gap,
            "avg_model_makespan": avg_model_mksp,
            "avg_optimal_makespan": avg_opt_mksp,
            "avg_ortools_solver_time": avg_solver_time,
            "avg_model_inference_time": avg_infer_time,
        },
        "instances": results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to: {results_path}")

    return results_data

def main():
    """Main function for model inference."""
    args = parse_args()
    if not 0.0 <= float(args.top_p) <= 1.0:
        raise ValueError(f"--top_p must be in [0, 1], got {args.top_p}")
    if int(args.top_k) < 0:
        raise ValueError(f"--top_k must be >= 0, got {args.top_k}")
    set_state_tensorization_mode(args.state_tensorization_mode)
    print(f"State tensorization mode: {get_state_tensorization_mode()}")
    model_n_jobs = args.model_n_jobs if args.model_n_jobs is not None else args.n_jobs
    model_n_machines = args.model_n_machines if args.model_n_machines is not None else args.n_machines

    # Load and optionally merge the model
    model_dir = getattr(args, "model_dir", None)
    peft_path = args.peft_adapter or model_dir

    dt_model = load_and_merge_model(
        base_model_name=args.base_model,
        peft_adapter_path=peft_path,
        custom_modules_path=args.custom_modules,
        n_jobs=model_n_jobs,
        n_machines=model_n_machines,
        merge_weights=args.merge_weights,
        load_in_4bit=args.load_in_4bit,
        save_merged_path=args.save_merged,
        model_source=args.model_source,
        dtype=args.dtype,
        max_length=args.max_length,
        max_ep_len=args.max_ep_len,
        state_encoder_type=args.state_encoder_type,
        graph_use_queue_pos=not args.graph_disable_queue_pos,
        use_candidate_action_head=args.use_candidate_action_head,
        use_candidate_harmony_mask=args.use_candidate_harmony_mask,
        candidate_harmony_group_by=args.candidate_harmony_group_by,
        use_scale_adapter_expert=args.use_scale_adapter_expert,
        use_scale_context_adapter=(args.use_scale_context_adapter or args.use_scale_adapter_expert),
        use_scale_candidate_residual=(args.use_scale_candidate_residual or args.use_scale_adapter_expert),
        use_scale_adapter_harmony_mask=args.use_scale_adapter_harmony_mask,
        scale_adapter_group_by=args.scale_adapter_group_by,
        scale_adapter_bottleneck=args.scale_adapter_bottleneck,
        scale_adapter_alpha_init=args.scale_adapter_alpha_init,
        use_scale_embedding=args.use_scale_embedding,
        use_scale_param_mask=args.use_scale_param_mask,
        mask_gnn_encoder=args.mask_gnn_encoder,
        mask_state_projection=args.mask_state_projection,
        mask_lora=args.mask_lora,
        mask_candidate_head=args.mask_candidate_head,
        mask_scale_adapter=args.mask_scale_adapter,
        scale_param_mask_group_by=args.scale_param_mask_group_by,
    )

    routing_mode = str(args.unseen_scale_mask_routing).strip().lower()
    if routing_mode == "all_one":
        routing_report = {
            "mode": "all_one",
            "status": "original_checkpoint_row_unchanged",
            "target": {
                "n_jobs": int(args.n_jobs),
                "n_machines": int(args.n_machines),
            },
        }
    else:
        if not args.use_scale_param_mask:
            raise ValueError(
                "Unseen-scale routing requires --use_scale_param_mask and the "
                "same five mask switches used by the checkpoint."
            )
        ssnc_state = getattr(dt_model, "_loaded_ssnc_mask_state", None)
        if ssnc_state is None:
            raise ValueError(
                "Checkpoint has no retained ssnc_mask_state; cannot identify "
                "seen domains for unseen-scale routing."
            )
        routing_report = apply_unseen_scale_neighbor_routing(
            model=dt_model,
            ssnc_state=ssnc_state,
            target_n_jobs=int(args.n_jobs),
            target_n_machines=int(args.n_machines),
            neighbor_k=int(args.unseen_scale_neighbor_k),
            mode=routing_mode,
            vote_rule=args.unseen_scale_vote_rule,
            vote_threshold=float(args.unseen_scale_vote_threshold),
        )
    dt_model._unseen_scale_routing_report = routing_report
    print("[SSNC ROUTING] " + json.dumps(routing_report, ensure_ascii=False))
    if args.unseen_scale_routing_report:
        report_dir = os.path.dirname(os.path.abspath(args.unseen_scale_routing_report))
        os.makedirs(report_dir, exist_ok=True)
        with open(args.unseen_scale_routing_report, "w", encoding="utf-8") as report_file:
            json.dump(routing_report, report_file, indent=2, ensure_ascii=False)

    if args.instances_jsonl:
        evaluate_instances_from_jsonl(
            dt_model=dt_model,
            instances_jsonl=args.instances_jsonl,
            n_jobs=args.n_jobs,
            n_machines=args.n_machines,
            max_steps=args.max_steps,
            max_length=args.max_length,
            target_makespan=args.target_makespan,
            target_return=args.target_return,
            rtg_mode=args.rtg_mode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            visualize=args.visualize,
            save_gantt=args.save_gantt,
            debug=args.debug,
            n_rollouts_per_instance=args.n_rollouts_per_instance,
            hybrid_bon=args.hybrid_bon,
            pick=args.pick,
            print_instance_hash=args.print_instance_hash,
            save_selected_jsonl=args.save_selected_jsonl,
        )
    elif args.evaluate_multiple:
        # Evaluate multiple instances and generate relative gap plots
        evaluate_multiple_instances(
            dt_model=dt_model,
            n_instances=args.n_instances,
            n_jobs=args.n_jobs,
            n_machines=args.n_machines,
            seed=args.seed,
            max_steps=args.max_steps,
            max_length=args.max_length,
            target_makespan=args.target_makespan,
            target_return=args.target_return,
            rtg_mode=args.rtg_mode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            visualize=args.visualize,
            save_gantt=args.save_gantt,
            debug=args.debug,
            ensure_unique_instances=not args.allow_duplicate_instances,
            fixed_seed_stream=args.fixed_seed_stream,
            n_rollouts_per_instance=args.n_rollouts_per_instance,
            hybrid_bon=args.hybrid_bon,
            pick=args.pick,
            print_instance_hash=args.print_instance_hash,
            save_selected_jsonl=args.save_selected_jsonl,
        )
    else:
        # Test the model with a single instance (original behavior)
        test_results = test_model_inference(
            dt_model=dt_model,
            n_jobs=args.n_jobs,
            n_machines=args.n_machines,
            seed=args.seed,
            max_steps=args.max_steps,
            max_length=args.max_length,
            target_makespan=args.target_makespan,
            target_return=args.target_return,
            rtg_mode=args.rtg_mode,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            visualize=args.visualize,
            save_gantt=args.save_gantt
        )

        # Save test results if requested
        if args.save_merged:
            results_path = os.path.join(args.save_merged, 'test_results.json')
            with open(results_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"Test results saved to: {results_path}")

if __name__ == "__main__":
    main()
