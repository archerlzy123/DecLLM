import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from dt_data_processing import JSSPTrajectoryCollector, ORToolsAgent
from dt_infer import load_and_merge_model, _obs_to_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DAgger trajectories for JSSP (student rollout + OR-Tools labels)")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_dir", type=str, default=None)
    model_group.add_argument("--peft_adapter", type=str, default=None)
    parser.add_argument("--custom_modules", type=str, default=None)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--model_source", type=str, default="huggingface", choices=["huggingface", "modelscope"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--n_machines", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--max_ep_len", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--target_return", type=float, default=1.0)
    parser.add_argument("--rtg_mode", type=str, default="constant", choices=["constant", "budget"])
    parser.add_argument("--fixed_seed_stream", action="store_true", default=False)

    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--append", action="store_true", default=False)

    return parser.parse_args()


@torch.no_grad()
def collect_one_episode(dt_model, env, max_steps, max_length, max_ep_len, temperature, target_return, rtg_mode, device):
    state_keys = [
        "job_next_op",
        "job_completion_mask",
        "remaining_processing_times",
        "normalized_machine_availability",
        "machine_busy_mask",
        "overall_info",
        "machine_sequence",
    ]

    collector = JSSPTrajectoryCollector(env, max_trajectories=1)
    expert = ORToolsAgent(env)
    expert.reset()

    obs = env.reset()

    instance = getattr(env, "current_instance", None)
    if isinstance(instance, np.ndarray):
        instance_hash = hashlib.md5(instance.tobytes()).hexdigest()[:10]
        total_proc = float(np.sum(instance[:, 1::2]))
    else:
        instance_hash = None
        total_proc = 0.0

    target_makespan = max(total_proc, float(getattr(env, "max_makespan", 0) or 0.0))
    if target_makespan <= 0:
        target_makespan = 1.0

    temp_states = [obs]
    temp_actions_student = []
    temp_actions_expert = []
    temp_rewards = []

    states_history = {k: [] for k in state_keys}
    actions_history = [-1]
    timesteps_history = [0]
    rtg_mode = str(rtg_mode).lower().strip()
    if rtg_mode not in {"constant", "budget"}:
        raise ValueError(f"Unknown rtg_mode={rtg_mode}. Use 'constant' or 'budget'.")
    if rtg_mode == "budget":
        rtg_history = [max(0.0, float(target_makespan) - float(getattr(env, "current_time", 0.0)))]
    else:
        # 与训练集 / 推理脚本统一：constant 模式使用负的归一化 RTG 条件。
        rtg_history = [-float(target_return)]

    done = False
    step = 0
    while not done and step < int(max_steps):
        state_dict_step = _obs_to_state_dict(obs, state_keys)
        if state_dict_step is None:
            raise TypeError("Environment observation must be a dict.")

        for k in state_keys:
            if k in state_dict_step:
                states_history[k].append(state_dict_step[k].to(dtype=torch.float32))

        action_space = None
        if hasattr(env, "get_available_actions"):
            try:
                action_space = env.get_available_actions()
            except Exception:
                action_space = None
        if action_space is None and hasattr(env, "action_space"):
            action_space = getattr(env, "action_space", None)

        if not action_space:
            step_result = env.step()
            if isinstance(step_result, tuple) and len(step_result) == 4:
                next_obs, _, done, _ = step_result
            elif isinstance(step_result, tuple) and len(step_result) == 3:
                next_obs, done, _ = step_result
            else:
                raise ValueError(f"Unexpected env.step return: {step_result}")
            obs = next_obs
            temp_states.append(obs)
            step += 1
            actions_history.append(-1)
            timesteps_history.append(min(timesteps_history[-1] + 1, int(max_ep_len)))
            if rtg_mode == "budget":
                rtg_history.append(max(0.0, float(target_makespan) - float(getattr(env, "current_time", 0.0))))
            else:
                rtg_history.append(-float(target_return))
            continue

        expert_action = int(expert.select_action(obs))

        start_idx = max(0, len(actions_history) - int(max_length))
        seq_len = len(actions_history) - start_idx

        model_states = {}
        for k in state_keys:
            if len(states_history[k]) == 0:
                continue
            seq = states_history[k][start_idx:]
            model_states[k] = torch.stack(seq, dim=0).unsqueeze(0).to(device)

        model_actions = torch.tensor(actions_history[start_idx:], dtype=torch.long, device=device).unsqueeze(0)
        model_returns = torch.tensor(rtg_history[start_idx:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        model_timesteps = torch.tensor(timesteps_history[start_idx:], dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

        outputs = dt_model(
            states=model_states,
            actions=model_actions,
            returns_to_go=model_returns,
            timesteps=model_timesteps,
            attention_mask=attn_mask,
        )
        logits = outputs.logits[:, -1, :]

        action_mask = torch.zeros((int(logits.shape[-1]),), dtype=torch.bool, device=device)
        for a in action_space:
            ai = int(a)
            if 0 <= ai < action_mask.numel():
                action_mask[ai] = True

        masked_logits = logits.clone()
        masked_logits[:, ~action_mask] = -float("inf")

        if torch.isinf(masked_logits).all():
            chosen = -1
        else:
            if temperature and float(temperature) > 0:
                probs = torch.softmax(masked_logits / float(temperature), dim=-1)
                chosen = int(torch.multinomial(probs, num_samples=1).item())
            else:
                chosen = int(torch.argmax(masked_logits, dim=-1).item())

        if chosen == -1:
            step_result = env.step()
        else:
            step_result = env.step(action=chosen)

        if isinstance(step_result, tuple) and len(step_result) == 4:
            next_obs, reward, done, info = step_result
        elif isinstance(step_result, tuple) and len(step_result) == 3:
            next_obs, done, info = step_result
            reward = float(info.get("reward", 0.0)) if isinstance(info, dict) else 0.0
        else:
            raise ValueError(f"Unexpected env.step return: {step_result}")

        if isinstance(info, dict) and "invalid" in info:
            step_result2 = env.step()
            if isinstance(step_result2, tuple) and len(step_result2) == 4:
                next_obs, reward2, done2, info2 = step_result2
            elif isinstance(step_result2, tuple) and len(step_result2) == 3:
                next_obs, done2, info2 = step_result2
                reward2 = float(info2.get("reward", 0.0)) if isinstance(info2, dict) else 0.0
            else:
                raise ValueError(f"Unexpected env.step return: {step_result2}")
            reward = float(reward) + float(reward2)
            done = bool(done) or bool(done2)

        temp_actions_student.append(int(chosen) if chosen != -1 else 0)
        temp_actions_expert.append(int(expert_action))
        temp_rewards.append(float(reward))

        obs = next_obs
        temp_states.append(obs)
        step += 1
        actions_history.append(chosen)
        timesteps_history.append(min(timesteps_history[-1] + 1, int(max_ep_len)))
        if rtg_mode == "budget":
            rtg_history.append(max(0.0, float(target_makespan) - float(getattr(env, "current_time", 0.0))))
        else:
            rtg_history.append(-float(target_return))

    returns_to_go = [float(x) for x in rtg_history]
    if len(returns_to_go) < len(temp_states):
        returns_to_go = returns_to_go + [returns_to_go[-1]] * (len(temp_states) - len(returns_to_go))
    elif len(returns_to_go) > len(temp_states):
        returns_to_go = returns_to_go[: len(temp_states)]

    if len(temp_states) > 0:
        if len(temp_actions_student) == len(temp_states) - 1:
            actions_student = temp_actions_student
        elif len(temp_actions_student) > len(temp_states) - 1:
            actions_student = temp_actions_student[: len(temp_states) - 1]
        else:
            actions_student = temp_actions_student + [0] * (len(temp_states) - 1 - len(temp_actions_student))

        if len(temp_actions_expert) == len(temp_states) - 1:
            actions_expert = temp_actions_expert
        elif len(temp_actions_expert) > len(temp_states) - 1:
            actions_expert = temp_actions_expert[: len(temp_states) - 1]
        else:
            actions_expert = temp_actions_expert + [0] * (len(temp_states) - 1 - len(temp_actions_expert))

        rewards = temp_rewards
        if len(rewards) > len(actions_student):
            rewards = rewards[: len(actions_student)]
        elif len(rewards) < len(actions_student):
            rewards = rewards + [0.0] * (len(actions_student) - len(rewards))

        timesteps = list(range(len(temp_states)))
        makespan = float(getattr(env, "get_makespan", lambda: -1)())
        traj = {
            "states": temp_states,
            "actions": actions_student,
            "expert_actions": actions_expert,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "metrics": {
                "dagger": True,
                "teacher": "ortools",
                "student_temperature": float(temperature),
                "rtg_mode": str(rtg_mode),
                "target_return": float(target_return),
                "model_condition_rtg": (-float(target_return) if rtg_mode == "constant" else None),
                "actual_makespan": makespan,
                "instance_hash": instance_hash,
                "n_steps": int(len(actions_student)),
            },
        }
        return collector._convert_to_serializable(traj)

    traj = {
        "states": temp_states,
        "actions": temp_actions_student,
        "expert_actions": temp_actions_expert,
        "rewards": temp_rewards,
        "returns_to_go": returns_to_go,
        "timesteps": list(range(len(temp_states))),
        "metrics": {
            "dagger": True,
            "teacher": "ortools",
            "student_temperature": float(temperature),
            "rtg_mode": str(rtg_mode),
            "target_return": float(target_return),
            "model_condition_rtg": (-float(target_return) if rtg_mode == "constant" else None),
            "actual_makespan": float(getattr(env, "get_makespan", lambda: -1)()),
            "instance_hash": instance_hash,
            "n_steps": int(len(temp_actions_student)),
        },
    }
    return collector._convert_to_serializable(traj)


def main():
    args = parse_args()

    peft_path = args.peft_adapter or args.model_dir
    dt_model = load_and_merge_model(
        base_model_name=args.base_model,
        peft_adapter_path=peft_path,
        custom_modules_path=args.custom_modules,
        n_jobs=args.n_jobs,
        n_machines=args.n_machines,
        merge_weights=False,
        load_in_4bit=args.load_in_4bit,
        save_merged_path=None,
        model_source=args.model_source,
        dtype=args.dtype,
        max_length=args.max_length,
        max_ep_len=args.max_ep_len,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt_model.eval()

    from Envs.JSSPEnv.JSSPEnv import JSSPEnv

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    if not args.append:
        with open(args.output_path, "w", encoding="utf-8") as f:
            pass

    t0 = time.time()
    for i in range(int(args.n_episodes)):
        current_seed = int(args.seed) if args.fixed_seed_stream else int(args.seed) + i
        env = JSSPEnv(
            n_job_range=[args.n_jobs, args.n_jobs],
            n_mac_range=[args.n_machines, args.n_machines],
            seed=current_seed,
            n_jobs=args.n_jobs,
            n_machines=args.n_machines,
        )
        traj = collect_one_episode(
            dt_model=dt_model,
            env=env,
            max_steps=args.max_steps,
            max_length=args.max_length,
            max_ep_len=args.max_ep_len,
            temperature=args.temperature,
            target_return=args.target_return,
            rtg_mode=args.rtg_mode,
            device=device,
        )
        with open(args.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"Collected {i+1}/{args.n_episodes} episodes ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
