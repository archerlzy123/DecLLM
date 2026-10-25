"""Data processing module for Decision Transformer on JSSP problem."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from collections import deque
from state_tensorization import (
    LEGACY_TRAIN_COMPATIBLE,
    tensorize_graph_state,
    validate_state_tensorization_mode,
)
# 导入solve_jssp函数用于计算最优解
from Envs.JSSPEnv.JSSPEnv import solve_jssp


def mutate_processing_times(instance, pct):
    if pct is None or pct <= 0:
        return instance
    inst = np.array(instance, copy=True)
    times = inst[:, 1::2].astype(np.float32)
    scale = np.random.uniform(1.0 - pct, 1.0 + pct, size=times.shape).astype(np.float32)
    times = np.maximum(1.0, np.round(times * scale))
    inst[:, 1::2] = times.astype(inst.dtype)
    return inst


LIGHT_TRAJECTORY_FORMAT = "jssp_light_v1"


def is_light_jssp_trajectory(trajectory):
    """Return True for compact JSSP trajectories that need env replay."""
    return (
        isinstance(trajectory, dict)
        and (
            trajectory.get("trajectory_format") == LIGHT_TRAJECTORY_FORMAT
            or trajectory.get("light_trajectory") is True
        )
    )


def _instance_from_trajectory(trajectory, env=None):
    if env is not None and getattr(env, "current_instance", None) is not None:
        return np.asarray(env.current_instance, dtype=np.int32)

    instance = trajectory.get("instance", None)
    if instance is not None:
        return np.asarray(instance, dtype=np.int32)

    states = trajectory.get("states", [])
    if states and isinstance(states[0], dict):
        first = states[0]
        machine_sequence = np.asarray(first.get("machine_sequence"), dtype=np.int32)
        processing_times = np.asarray(first.get("processing_times"), dtype=np.int32)
        if machine_sequence.ndim == 2 and processing_times.shape == machine_sequence.shape:
            n_jobs, n_machines = machine_sequence.shape
            inst = np.zeros((n_jobs, n_machines * 2), dtype=np.int32)
            inst[:, 0::2] = machine_sequence
            inst[:, 1::2] = processing_times
            return inst

    raise ValueError("Cannot build light trajectory: missing JSSP instance data")


def make_light_jssp_trajectory(trajectory, env=None):
    """Compress a full trajectory to instance + actions; states are replayed later."""
    instance = _instance_from_trajectory(trajectory, env=env)
    n_jobs = int(instance.shape[0])
    n_machines = int(instance.shape[1] // 2)
    metrics = dict(trajectory.get("metrics", {}) or {})
    metrics.setdefault("n_jobs", n_jobs)
    metrics.setdefault("n_machines", n_machines)

    light = {
        "trajectory_format": LIGHT_TRAJECTORY_FORMAT,
        "light_trajectory": True,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "instance": instance.tolist(),
        "actions": [int(a) for a in trajectory.get("actions", [])],
        "metrics": metrics,
    }
    for key in ("tier", "epsilon", "mutation", "total_return", "expert_actions"):
        if key in trajectory:
            light[key] = trajectory[key]
    if "rewards" in trajectory:
        light["rewards"] = [float(r) for r in trajectory.get("rewards", [])]
    if "returns_to_go" in trajectory:
        light["returns_to_go"] = trajectory.get("returns_to_go", [])
    if "timesteps" in trajectory:
        light["timesteps"] = trajectory.get("timesteps", [])
    return light


def replay_light_jssp_trajectory(trajectory):
    """Rebuild decision states from a compact trajectory by replaying actions."""
    if not is_light_jssp_trajectory(trajectory):
        return trajectory

    from Envs.JSSPEnv.JSSPEnv import JSSPEnv

    instance = np.asarray(trajectory["instance"], dtype=np.int32)
    n_jobs = int(trajectory.get("n_jobs", instance.shape[0]))
    n_machines = int(trajectory.get("n_machines", instance.shape[1] // 2))
    actions = [int(a) for a in trajectory.get("actions", [])]
    metrics = dict(trajectory.get("metrics", {}) or {})

    env = JSSPEnv(
        n_job_range=[n_jobs, n_jobs],
        n_mac_range=[n_machines, n_machines],
        external_instance=instance,
        n_jobs=n_jobs,
        n_machines=n_machines,
    )
    env.max_steps = max(int(metrics.get("n_steps", 0) or 0), len(actions) + 5, n_jobs * n_machines + 5)

    state = env.reset()
    states = [state]
    replay_rewards = []
    done = False

    for step_idx, action in enumerate(actions):
        guard = 0
        while not done and action not in getattr(env, "action_space", []):
            advanced = env._advance_time()
            guard += 1
            if not advanced or guard > max(10, n_jobs * n_machines * 2):
                raise ValueError(
                    f"Light trajectory replay failed before step {step_idx}: "
                    f"action {action} not legal, action_space={getattr(env, 'action_space', [])}"
                )
            states[-1] = env._get_dynamic_state()

        next_state, done, info = env.step(action=action)
        if info.get("invalid"):
            raise ValueError(
                f"Light trajectory replay got invalid action at step {step_idx}: "
                f"action={action}, invalid={info.get('invalid')}"
            )
        replay_rewards.append(float(info.get("reward", 0.0)))
        states.append(next_state)

    future_return = 0.0
    returns_to_go = []
    for reward in reversed(replay_rewards):
        future_return += float(reward)
        returns_to_go.insert(0, future_return)
    returns_to_go.append(0.0)

    rebuilt = dict(trajectory)
    rebuilt["states"] = states
    rebuilt["actions"] = actions
    rebuilt["rewards"] = replay_rewards
    rebuilt["returns_to_go"] = returns_to_go
    rebuilt["timesteps"] = list(range(len(states)))
    rebuilt["trajectory_length"] = len(actions)
    rebuilt["metrics"] = metrics
    rebuilt["metrics"]["replayed_from_light"] = True
    return rebuilt


def validate_light_replay_equivalence(full_trajectory, light_trajectory):
    """Cheap safety check used during light data generation."""
    rebuilt = replay_light_jssp_trajectory(light_trajectory)
    if len(rebuilt.get("actions", [])) != len(full_trajectory.get("actions", [])):
        raise ValueError("Light replay action length mismatch")
    if rebuilt.get("actions", []) != [int(a) for a in full_trajectory.get("actions", [])]:
        raise ValueError("Light replay actions mismatch")

    full_states = full_trajectory.get("states", [])
    rebuilt_states = rebuilt.get("states", [])
    if len(full_states) != len(rebuilt_states):
        raise ValueError(f"Light replay state length mismatch: {len(full_states)} != {len(rebuilt_states)}")

    keys = ("job_next_op", "legal_action_mask", "scheduled_machine_adj", "current_time")
    for i, (a, b) in enumerate(zip(full_states, rebuilt_states)):
        for key in keys:
            if key not in a or key not in b:
                continue
            av = np.asarray(a[key])
            bv = np.asarray(b[key])
            if av.shape != bv.shape or not np.allclose(av, bv, atol=1e-6):
                raise ValueError(f"Light replay mismatch at state {i}, key={key}")

    full_ms = (full_trajectory.get("metrics", {}) or {}).get("actual_makespan")
    rebuilt_ms = rebuilt.get("metrics", {}).get("actual_makespan")
    if full_ms is not None and rebuilt_ms is not None and float(full_ms) != float(rebuilt_ms):
        raise ValueError("Light replay actual_makespan mismatch")
    return True


def validate_light_replay_complete(light_trajectory):
    """Validate the exact representation used by training for light trajectories."""
    rebuilt = replay_light_jssp_trajectory(light_trajectory)
    error = _trajectory_decision_field_error(rebuilt) if "_trajectory_decision_field_error" in globals() else None
    if error is not None:
        metrics = rebuilt.get("metrics", {}) if isinstance(rebuilt, dict) else {}
        raise ValueError(
            "Light replay is not a complete decision trajectory: "
            f"{error}; actions={len(rebuilt.get('actions', []))}, "
            f"states={len(rebuilt.get('states', []))}, "
            f"expected_ops={int(metrics.get('n_jobs', 0) or 0) * int(metrics.get('n_machines', 0) or 0)}"
        )
    return True


class JSSPTrajectoryCollector:
    """Collector for JSSP trajectories from environment interactions."""
    
    def __init__(self, env, max_trajectories=1000):
        """
        Initialize the trajectory collector.
        
        Args:
            env: JSSP environment
            max_trajectories: Maximum number of trajectories to store
        """
        self.env = env
        self.max_trajectories = max_trajectories
        self.trajectories = []
        
    def collect_trajectory(self, agent, max_steps=100, render=False, verbose=True):
        """
        Collect a single trajectory using the provided agent.
        Records all states, actions, and rewards to create a complete trajectory.
        
        Args:
            agent: Agent to use for action selection
            max_steps: Maximum number of steps per trajectory
            render: Whether to render the environment
            
        Returns:
            Trajectory dictionary
        """
        log = print if verbose else (lambda *args, **kwargs: None)
        state = self.env.reset()
        mutation_pct = float(getattr(self.env, "processing_time_mutation_pct", 0.0) or 0.0)
        if mutation_pct > 0 and hasattr(self.env, "current_instance") and self.env.current_instance is not None:
            self.env.current_instance = mutate_processing_times(self.env.current_instance, mutation_pct)
            if hasattr(self.env, "flat_instance"):
                self.env.flat_instance = self.env.current_instance.flatten()
            if hasattr(self.env, "_calculate_upper_bound"):
                self.env.max_makespan = self.env._calculate_upper_bound()
            state = self.env._get_dynamic_state()
        log(f"\n=== Starting trajectory collection ===")
        log(f"Initial state - n_jobs: {self.env.n_jobs}, n_machines: {self.env.n_machines}")
        log(f"Initial job status: {self.env.job_status}")
        log(f"Initial machine status: {self.env.machine_status}")
        
        # Initialize trajectory storage
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'returns_to_go': [],
            'timesteps': []
        }
        
        # Calculate total return for returns-to-go
        total_return = 0
        done = False
        
        # First, collect all steps to calculate returns-to-go
        temp_states = []
        temp_actions = []
        temp_rewards = []
        
        # Always record the initial state
        temp_states.append(state)
        
        # Use while loop to collect all steps
        step_count = 0

        while not done and step_count < max_steps:
            log(f"\n--- Step {step_count} ---")
            log(f"Current time: {self.env.current_time}")
            log(f"Current job status: {self.env.job_status}")
            log(f"Current machine status: {self.env.machine_status}")
            log(f"Available actions: {self.env.action_space}")

            # Get available actions
            available_actions = self.env.action_space if hasattr(self.env, 'action_space') else []

            if not available_actions:
                # No actions available, automatically advance time to next event.
                # 这里只推进环境，不把 time-advance 事件写入轨迹，保证轨迹只包含真正的决策点。
                log("No available actions, advancing time...")
                time_advanced = self.env._advance_time()
                if time_advanced:
                    state = self.env._get_dynamic_state()
                    if temp_states:
                        temp_states[-1] = state
                    log(f"Time advanced to: {self.env.current_time}")
                    log(f"New action space: {self.env.action_space}")

                step_count += 1
                continue

            # Get action from agent
            try:
                action = agent.select_action(state, epsilon=getattr(agent, "epsilon", 0.0))
            except TypeError:
                action = agent.select_action(state)

            if action == -1:
                # 当 agent 无法给出合法动作时，仅推进时间，不把该事件记为决策样本。
                time_advanced = self.env._advance_time()

                if time_advanced:
                    state = self.env._get_dynamic_state()
                    if temp_states:
                        temp_states[-1] = state

                step_count += 1
                continue

            log(f"Selected action: {action}")

            # Step in environment - 修复参数传递问题
            next_state, done, info = self.env.step(action=action)
            log(f"Step result: done={done}")
            log(f"New job status: {self.env.job_status}")
            log(f"New machine status: {self.env.machine_status}")
            log(f"Environment returned reward: {info.get('reward', 0.0)}")

            # 只使用环境返回的 reward，避免把等待时间惩罚延迟混入到下一个动作上
            reward = float(info.get('reward', 0.0))

            # Store only true decision steps
            temp_actions.append(action)
            temp_rewards.append(reward)
            temp_states.append(next_state)
            total_return += reward

            # Render if requested
            if render:
                self.env.render()

            state = next_state
            step_count += 1

        # Calculate returns-to-go (future returns from each timestep)
        returns_to_go = []
        future_return = 0
        for reward in reversed(temp_rewards):
            future_return += reward
            returns_to_go.insert(0, future_return)

        # Add the final state's RTG (which should be 0.0)
        returns_to_go.append(0.0)

        # Handle edge cases
        if len(temp_states) == 0:
            # If no states, return empty RTG
            returns_to_go = []
        elif len(temp_states) < len(returns_to_go):
            # If states are shorter than RTG (shouldn't happen with correct calculation)
            returns_to_go = returns_to_go[:len(temp_states)]
        elif len(temp_states) > len(returns_to_go):
            # If states are longer than RTG (shouldn't happen with correct calculation)
            # This is a safety measure only
            returns_to_go = returns_to_go + [0.0] * (len(temp_states) - len(returns_to_go))

        # Populate trajectory with all data
        # Ensure proper Decision Transformer trajectory structure:
        # len(states) = len(returns_to_go) = len(timesteps) = len(actions) + 1
        if len(temp_states) > 0:
            # states: [S_0, S_1, ..., S_T] (T+1 elements)
            trajectory['states'] = temp_states

            # actions: [A_0, A_1, ..., A_{T-1}] (T elements)
            # Ensure actions has length len(states) - 1
            if len(temp_actions) == len(temp_states) - 1:
                trajectory['actions'] = temp_actions
            elif len(temp_actions) > len(temp_states) - 1:
                # Truncate if actions is too long
                trajectory['actions'] = temp_actions[:len(temp_states) - 1]
            else:
                # Pad with 0s if actions is too short (should rarely happen with proper event-driven sampling)
                trajectory['actions'] = temp_actions + [0] * (len(temp_states) - 1 - len(temp_actions))

            # rewards: [R_1, R_2, ..., R_T] (T elements)
            trajectory['rewards'] = temp_rewards

            # Ensure rewards has length len(actions)
            if len(trajectory['rewards']) > len(trajectory['actions']):
                trajectory['rewards'] = trajectory['rewards'][:len(trajectory['actions'])]
            elif len(trajectory['rewards']) < len(trajectory['actions']):
                trajectory['rewards'] = trajectory['rewards'] + [0.0] * (len(trajectory['actions']) - len(trajectory['rewards']))

            # returns_to_go: [G_0, G_1, ..., G_T] (T+1 elements, matches states)
            # We already calculated returns_to_go with correct length (T+1) in the earlier step
            # This should be consistent with the trajectory structure
            trajectory['returns_to_go'] = returns_to_go

            # timesteps: [0, 1, ..., T] (T+1 elements, matches states)
            trajectory['timesteps'] = list(range(len(trajectory['states'])))

            # Ensure all sequences have correct lengths according to DT structure
            T = len(trajectory['actions'])
            assert len(trajectory['states']) == T + 1, f"States length ({len(trajectory['states'])}) must be T + 1 ({T + 1})"
            assert len(trajectory['returns_to_go']) == T + 1, f"Returns_to_go length ({len(trajectory['returns_to_go'])}) must be T + 1 ({T + 1})"
            assert len(trajectory['timesteps']) == T + 1, f"Timesteps length ({len(trajectory['timesteps'])}) must be T + 1 ({T + 1})"
            assert len(trajectory['rewards']) == T, f"Rewards length ({len(trajectory['rewards'])}) must be T ({T})"
        else:
            # Handle empty trajectory case
            trajectory['states'] = temp_states
            trajectory['actions'] = temp_actions
            trajectory['rewards'] = temp_rewards
            trajectory['returns_to_go'] = returns_to_go
            trajectory['timesteps'] = list(range(len(temp_states)))

        trajectory['total_return'] = total_return

        # Calculate optimal makespan using OR-Tools and efficiency metrics
        try:
            # 检查调度是否完成
            job_status = self.env.job_status
            n_jobs = self.env.n_jobs
            n_machines = self.env.n_machines

            # 计算完成度：所有作业的所有工序都完成
            total_ops = n_jobs * n_machines
            completed_ops = int(np.sum(job_status))
            completion_ratio = completed_ops / total_ops if total_ops > 0 else 0.0

            # 只有调度完成时才计算效率
            if completion_ratio < 1.0:
                # 调度未完成，标记为无效
                log(f"\n=== Trajectory Metrics ===")
                log(f"Completion ratio: {completion_ratio:.3f} ❌ (Not completed)")
                log(f"Number of steps: {len(temp_actions)}")

                trajectory['metrics'] = {
                    'optimal_makespan': 0,
                    'actual_makespan': self.env.get_makespan() if hasattr(self.env, 'get_makespan') else 0,
                    'efficiency_ratio': 0.0,
                    'relative_gap': None,
                    'efficiency_meets_target': False,
                    'target_efficiency': 0.85,
                    'n_steps': len(temp_actions),
                    'n_jobs': self.env.n_jobs,
                    'n_machines': self.env.n_machines,
                    'completion_ratio': completion_ratio,
                    'completed': False
                }
            else:
                # 调度已完成，计算效率
                optimal_makespan, _ = solve_jssp(self.env.current_instance)
                actual_makespan = self.env.get_makespan()
                # 确保actual_makespan是整数类型
                if isinstance(actual_makespan, str):
                    actual_makespan = int(float(actual_makespan))
                elif not isinstance(actual_makespan, (int, float)):
                    actual_makespan = int(actual_makespan)
                efficiency_ratio = optimal_makespan / (actual_makespan + 1e-8)
                relative_gap = (actual_makespan - optimal_makespan) / (optimal_makespan + 1e-8)

                # Check if efficiency ratio meets the target threshold
                # 恢复原始效率阈值，确保训练和推理数据分布一致
                target_efficiency = 0.85
                efficiency_meets_target = efficiency_ratio >= target_efficiency

                log(f"\n=== Trajectory Metrics ===")
                log(f"Optimal makespan: {optimal_makespan}")
                log(f"Actual makespan: {actual_makespan}")
                log(f"Efficiency ratio: {efficiency_ratio:.3f} {'✅' if efficiency_meets_target else '❌'}")
                log(f"Target efficiency: {target_efficiency} - {'MET' if efficiency_meets_target else 'NOT MET'}")
                log(f"Number of steps: {len(temp_actions)}")

                # Add detailed metrics to trajectory
                trajectory['metrics'] = {
                    'optimal_makespan': optimal_makespan,
                    'actual_makespan': actual_makespan,
                    'efficiency_ratio': efficiency_ratio,
                    'relative_gap': float(relative_gap),
                    'efficiency_meets_target': efficiency_meets_target,
                    'target_efficiency': target_efficiency,
                    'n_steps': len(temp_actions),
                    'n_jobs': self.env.n_jobs,
                    'n_machines': self.env.n_machines,
                    'completion_ratio': completion_ratio,
                    'completed': True
                }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            trajectory['metrics'] = {
                'optimal_makespan': 0,
                'actual_makespan': self.env.get_makespan() if hasattr(self.env, 'get_makespan') else 0,
                'efficiency_ratio': 0.0,
                'relative_gap': None,
                'efficiency_meets_target': False,
                'target_efficiency': 0.85,
                'n_steps': len(temp_actions),
                'n_jobs': getattr(self.env, 'n_jobs', 0),
                'n_machines': getattr(self.env, 'n_machines', 0)
            }

        metrics = trajectory.get('metrics', {})
        metrics['epsilon'] = float(getattr(agent, 'epsilon', 0.0) or 0.0)
        metrics['random_action_count'] = int(getattr(agent, 'random_action_count', 0) or 0)
        metrics['processing_time_mutation_pct'] = float(getattr(self.env, "processing_time_mutation_pct", 0.0) or 0.0)
        trajectory['metrics'] = metrics

        states_len = len(trajectory.get('states', []))
        if states_len > 0:
            score = float(metrics.get('efficiency_ratio', 0.0) or 0.0)
            trajectory['metrics']['trajectory_quality_score'] = score
            trajectory['metrics']['rtg_mode'] = 'future_return'

        # Add to collection
        if len(self.trajectories) < self.max_trajectories:
            self.trajectories.append(trajectory)
        else:
            # Replace oldest trajectory
            self.trajectories.pop(0)
            self.trajectories.append(trajectory)

        return trajectory

    def collect_ortools_trajectories(self, num_trajectories=100, max_steps=100, output_path=None):
        """
        Collect trajectories using OR-Tools optimized action selection, filtering for efficiency ratio ≥ 0.85.
        添加多维度质量过滤：步数、稳定性、多样性。
        添加分层采样：按效率分层，确保数据分布均匀。

        Args:
            num_trajectories: Number of valid trajectories to collect (efficiency ratio ≥ 0.85)
            max_steps: Maximum steps per trajectory
            output_path: Optional path to save trajectories. If provided, trajectories will be saved incrementally.
        """
        # 定义效率分层和目标比例
        efficiency_bins = {
            'high': {'range': (0.95, 1.00), 'target': 0.2, 'collected': 0},    # 高效率：20%
            'medium': {'range': (0.85, 0.95), 'target': 0.6, 'collected': 0},  # 中效率：60%
            'low': {'range': (0.75, 0.85), 'target': 0.2, 'collected': 0}     # 低效率：20%
        }

        # 计算每个分层的目标数量
        for level, config in efficiency_bins.items():
            config['target_count'] = max(1, int(num_trajectories * config['target']))  # 确保每个分层至少有1个目标

        collected_count = 0
        attempts = 0
        max_attempts = num_trajectories * 15  # 增加最大尝试次数，因为分层采样更严格

        # 与最终指标更一致的过滤阈值
        max_steps_ratio = 2.0
        max_relative_gap = 0.35

        # 跟踪实例哈希，确保多样性
        instance_hashes = set()

        # 判断是否所有分层都已完成
        def all_bins_completed():
            return all(config['collected'] >= config['target_count'] for config in efficiency_bins.values())

        while attempts < max_attempts:
            # 检查是否满足停止条件
            total_collected = sum(config['collected'] for config in efficiency_bins.values())
            if total_collected >= num_trajectories or all_bins_completed():
                break

            attempts += 1

            # 打印收集进度
            if attempts % 100 == 0 or total_collected % 10 == 0:
                print(f"\n📊 Overall Progress: {total_collected}/{num_trajectories} trajectories collected")
                print(f"   Bin Progress: High={efficiency_bins['high']['collected']}/{efficiency_bins['high']['target_count']}, "
                      f"Medium={efficiency_bins['medium']['collected']}/{efficiency_bins['medium']['target_count']}, "
                      f"Low={efficiency_bins['low']['collected']}/{efficiency_bins['low']['target_count']}")
                print(f"   Attempts: {attempts}/{max_attempts}, Success rate: {total_collected/attempts:.3f}")

            # --- 关键修改：为每个轨迹创建全新的环境 ---
            # 假设 self.env 有 n_job_range, n_mac_range 属性
            new_env = type(self.env)(
                n_job_range=self.env.n_job_range,
                n_mac_range=self.env.n_mac_range,
                seed=None  # 确保随机性
            )
            # --- 修改结束 ---

            # 创建新的 collector 和 agent
            new_collector = JSSPTrajectoryCollector(new_env, max_trajectories=1)
            agent = ORToolsAgent(new_env)
            agent.reset()

            # 收集轨迹
            trajectory = new_collector.collect_trajectory(agent, max_steps=max_steps)

            # 多维度质量过滤
            metrics = trajectory['metrics']
            efficiency_ratio = metrics['efficiency_ratio']
            optimal_makespan = metrics['optimal_makespan']
            actual_makespan = metrics['actual_makespan']
            n_steps = metrics['n_steps']
            completed_ok = bool(metrics.get('completed', False))
            relative_gap = metrics.get('relative_gap', None)
            if relative_gap is None and optimal_makespan and float(optimal_makespan) > 0:
                relative_gap = (float(actual_makespan) - float(optimal_makespan)) / (float(optimal_makespan) + 1e-8)

            # 1. 完成度过滤：只保留完整调度轨迹
            completion_ok = completed_ok

            # 2. 效率过滤（保持）
            efficiency_ok = efficiency_ratio >= 0.75

            # 3. 与论文主指标对齐：relative gap 过滤
            gap_ok = (relative_gap is not None) and (float(relative_gap) <= max_relative_gap)

            # 4. 步数过滤：限制过长/无效轨迹
            steps_ok = bool(optimal_makespan) and (n_steps <= float(optimal_makespan) * max_steps_ratio)

            # 5. 多样性过滤：避免重复实例
            instance_hash = hash(new_env.current_instance.tobytes()) if hasattr(new_env, 'current_instance') else 0
            diversity_ok = instance_hash not in instance_hashes

            # 6. 分层采样：根据效率决定属于哪个分层
            bin_assigned = None
            for level, config in efficiency_bins.items():
                min_eff, max_eff = config['range']
                if min_eff <= efficiency_ratio < max_eff:
                    bin_assigned = level
                    break

            # 检查该分层是否还需要轨迹
            bin_needs_more = (bin_assigned is not None and
                             efficiency_bins[bin_assigned]['collected'] < efficiency_bins[bin_assigned]['target_count'])

            all_conditions_met = (completion_ok and efficiency_ok and gap_ok and steps_ok and diversity_ok and bin_needs_more)

            if all_conditions_met:
                instance_hashes.add(instance_hash)  # 记录已使用的实例哈希
                efficiency_bins[bin_assigned]['collected'] += 1  # 更新该分层的计数
                total_collected = sum(config['collected'] for config in efficiency_bins.values())

                print(f"✅ Collected valid OR-Tools trajectory {total_collected}/{num_trajectories} (Attempt: {attempts})")
                print(f"   Bin: {bin_assigned}, Efficiency: {efficiency_ratio:.3f}, Gap: {float(relative_gap)*100.0:.2f}%, Steps: {n_steps}/{optimal_makespan*max_steps_ratio:.0f}")
                print(f"   Bin Progress: High={efficiency_bins['high']['collected']}/{efficiency_bins['high']['target_count']}, "
                      f"Medium={efficiency_bins['medium']['collected']}/{efficiency_bins['medium']['target_count']}, "
                      f"Low={efficiency_bins['low']['collected']}/{efficiency_bins['low']['target_count']}")

                # Add to collection
                if len(self.trajectories) >= self.max_trajectories:
                    # Replace oldest trajectory if max reached
                    self.trajectories.pop(0)

                optimized_trajectory = trajectory

                # 应用轨迹质量评估（方案6）：计算综合评分
                quality_score, quality_details = self.evaluate_trajectory_quality(optimized_trajectory)
                optimized_trajectory['metrics']['quality_score'] = quality_score
                optimized_trajectory['metrics']['quality_details'] = quality_details

                print(f"🔧 Quality Score: {quality_score:.3f} (Eff: {quality_details['efficiency_score']:.2f}, "
                      f"Stab: {quality_details['stability_score']:.2f}, Div: {quality_details['diversity_score']:.2f}, "
                      f"Comp: {quality_details['completeness_score']:.2f})")

                self.trajectories.append(optimized_trajectory)

                # Save incrementally if output path is provided
                if output_path:
                    self.save_trajectories(output_path, append=True)
                    print(f"✅ Saved valid trajectory {total_collected} to {output_path}")
            else:
                # 打印过滤原因
                reasons = []
                if not completion_ok:
                    reasons.append("Not completed")
                if not efficiency_ok:
                    reasons.append(f"Efficiency {efficiency_ratio:.3f} < 0.75")
                if not gap_ok:
                    if relative_gap is None:
                        reasons.append("Relative gap unavailable")
                    else:
                        reasons.append(f"Relative gap {float(relative_gap)*100.0:.2f}% > {max_relative_gap*100.0:.1f}%")
                if not steps_ok:
                    reasons.append(f"Steps {n_steps} > {optimal_makespan*max_steps_ratio:.0f}")
                if not diversity_ok:
                    reasons.append("Duplicate instance")
                if bin_assigned is None:
                    reasons.append(f"Efficiency {efficiency_ratio:.3f} not in any bin")
                elif not bin_needs_more:
                    reasons.append(f"Bin {bin_assigned} full ({efficiency_bins[bin_assigned]['collected']}/{efficiency_bins[bin_assigned]['target_count']})")

                print(f"❌ Discarded trajectory (Attempt: {attempts}, Reasons: {', '.join(reasons)})")

        total_collected = sum(config['collected'] for config in efficiency_bins.values())
        if total_collected < num_trajectories:
            print(f"⚠️ Warning: Only collected {total_collected}/{num_trajectories} valid trajectories after {attempts} attempts")
            print(f"   Final Bin Progress: High={efficiency_bins['high']['collected']}/{efficiency_bins['high']['target_count']}, "
                  f"Medium={efficiency_bins['medium']['collected']}/{efficiency_bins['medium']['target_count']}, "
                  f"Low={efficiency_bins['low']['collected']}/{efficiency_bins['low']['target_count']}")
        else:
            print(f"🎉 Successfully collected {total_collected}/{num_trajectories} valid trajectories")
            print(f"   Final Bin Distribution: High={efficiency_bins['high']['collected']}, "
                  f"Medium={efficiency_bins['medium']['collected']}, Low={efficiency_bins['low']['collected']}")

    def augment_trajectory(self, trajectory, augmentation_ratio=0.3):
        """
        对轨迹进行数据增强，增加数据多样性。

        Args:
            trajectory: 原始轨迹
            augmentation_ratio: 增强比例（0.3表示30%的轨迹会被增强）

        Returns:
            augmented_trajectory: 增强后的轨迹
        """
        return trajectory

        # 复制轨迹，避免修改原始数据
        augmented = trajectory.copy()

        # 选择增强方式
        augmentation_type = np.random.choice([
            'state_noise',      # 状态噪声
            'action_perturb',   # 动作扰动
            'reward_scale',      # 奖励缩放
            'rtg_scale'         # RTG缩放
        ])

        if augmentation_type == 'state_noise':
            # 1. 状态噪声增强：对状态添加小幅高斯噪声
            augmented['states'] = []
            for state in trajectory['states']:
                if isinstance(state, dict):
                    # 如果状态是字典，对数值字段添加噪声
                    noisy_state = {}
                    for key, value in state.items():
                        if isinstance(value, (int, float)):
                            noise = np.random.normal(0, 0.05 * abs(value) if value != 0 else 0.01)
                            noisy_state[key] = value + noise
                        else:
                            noisy_state[key] = value
                    augmented['states'].append(noisy_state)
                else:
                    # 如果状态是数组，添加噪声
                    noise = np.random.normal(0, 0.05, size=state.shape)
                    noisy_state = state + noise
                    augmented['states'].append(noisy_state)

            print(f"🔧 Applied state noise augmentation")

        elif augmentation_type == 'action_perturb':
            # 2. 动作扰动增强：随机扰动部分动作
            augmented['actions'] = []
            for i, action in enumerate(trajectory['actions']):
                if np.random.random() < 0.1:  # 10%的概率扰动
                    # 动作是(job_idx, machine_idx)格式
                    if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
                        job_idx, machine_idx = action
                        # 扰动机器索引（±1）
                        if np.random.random() < 0.5:
                            machine_idx = max(0, min(self.env.n_machines - 1, machine_idx + np.random.choice([-1, 1])))
                        perturbed_action = [job_idx, machine_idx]
                        augmented['actions'].append(perturbed_action)
                    else:
                        augmented['actions'].append(action)
                else:
                    augmented['actions'].append(action)

            print(f"🔧 Applied action perturbation augmentation")

        elif augmentation_type == 'reward_scale':
            # 3. 奖励缩放增强：轻微缩放奖励
            scale_factor = np.random.uniform(0.95, 1.05)
            augmented['rewards'] = [r * scale_factor for r in trajectory['rewards']]
            augmented['returns_to_go'] = [rtg * scale_factor for rtg in trajectory['returns_to_go']]

            print(f"🔧 Applied reward scaling augmentation (factor: {scale_factor:.3f})")

        elif augmentation_type == 'rtg_scale':
            # 4. RTG缩放增强：轻微缩放returns-to-go
            scale_factor = np.random.uniform(0.98, 1.02)
            augmented['returns_to_go'] = [rtg * scale_factor for rtg in trajectory['returns_to_go']]

            print(f"🔧 Applied RTG scaling augmentation (factor: {scale_factor:.3f})")

        # 更新轨迹指标
        if 'metrics' in augmented:
            augmented['metrics'] = augmented['metrics'].copy()
            augmented['metrics']['augmented'] = True
            augmented['metrics']['augmentation_type'] = augmentation_type

        return augmented

    def optimize_trajectory_length(self, trajectory, optimization_ratio=0.5):
        """
        优化轨迹长度：截断、压缩，减少冗余信息。

        Args:
            trajectory: 原始轨迹
            optimization_ratio: 优化比例（0.5表示50%的轨迹会被优化）

        Returns:
            optimized_trajectory: 优化后的轨迹
        """
        # 随机决定是否优化
        if np.random.random() > optimization_ratio:
            return trajectory

        # 复制轨迹，避免修改原始数据
        optimized = trajectory.copy()

        # 选择优化方式
        optimization_type = np.random.choice([
            'truncate',      # 轨迹截断
            'compress'        # 轨迹压缩
        ])

        if optimization_type == 'truncate':
            # 1. 轨迹截断：截断过长轨迹
            optimal_makespan = trajectory['metrics'].get('optimal_makespan', 50)
            actual_makespan = trajectory['metrics'].get('actual_makespan', 50)
            n_steps = len(trajectory['actions'])

            # 如果轨迹长度超过最优解的1.2倍，截断到1.2倍
            max_length = int(optimal_makespan * 1.2)
            if n_steps > max_length:
                # 截断轨迹
                truncate_idx = max_length

                optimized['states'] = trajectory['states'][:truncate_idx + 1]
                optimized['actions'] = trajectory['actions'][:truncate_idx]
                optimized['rewards'] = trajectory['rewards'][:truncate_idx]
                optimized['returns_to_go'] = trajectory['returns_to_go'][:truncate_idx + 1]
                optimized['timesteps'] = trajectory['timesteps'][:truncate_idx + 1]

                # 更新轨迹指标
                optimized['metrics'] = optimized['metrics'].copy()
                optimized['metrics']['optimized'] = True
                optimized['metrics']['optimization_type'] = 'truncate'
                optimized['metrics']['original_length'] = n_steps
                optimized['metrics']['optimized_length'] = truncate_idx

                print(f"🔧 Truncated trajectory from {n_steps} to {truncate_idx} steps")
            else:
                # 轨迹长度合理，不截断
                optimized = trajectory

        elif optimization_type == 'compress':
            # 2. 轨迹压缩：合并连续的相同动作
            original_actions = trajectory['actions']
            compressed_actions = []

            # 压缩动作序列
            for action in original_actions:
                if not compressed_actions or action != compressed_actions[-1]:
                    compressed_actions.append(action)

            # 如果压缩后长度减少，重新计算轨迹
            if len(compressed_actions) < len(original_actions):
                # 计算压缩比例
                compression_ratio = len(compressed_actions) / len(original_actions)

                # 重新计算轨迹（简化处理：只压缩动作）
                optimized['actions'] = compressed_actions

                # 按照压缩比例截断其他序列
                compressed_length = len(compressed_actions)
                optimized['states'] = trajectory['states'][:compressed_length + 1]
                optimized['rewards'] = trajectory['rewards'][:compressed_length]
                optimized['returns_to_go'] = trajectory['returns_to_go'][:compressed_length + 1]
                optimized['timesteps'] = trajectory['timesteps'][:compressed_length + 1]

                # 更新轨迹指标
                optimized['metrics'] = optimized['metrics'].copy()
                optimized['metrics']['optimized'] = True
                optimized['metrics']['optimization_type'] = 'compress'
                optimized['metrics']['original_length'] = len(original_actions)
                optimized['metrics']['optimized_length'] = compressed_length
                optimized['metrics']['compression_ratio'] = compression_ratio

                print(f"🔧 Compressed trajectory from {len(original_actions)} to {compressed_length} steps (ratio: {compression_ratio:.2f})")
            else:
                # 动作已经是最优的，不压缩
                optimized = trajectory

        return optimized

    def evaluate_trajectory_quality(self, trajectory):
        """
        多维度评估轨迹质量，计算综合评分。

        Args:
            trajectory: 轨迹

        Returns:
            quality_score: 综合质量评分（0-1之间）
            quality_details: 各维度评分详情
        """
        metrics = trajectory.get('metrics', {})

        # 1. 效率评分（30%权重）
        efficiency_ratio = metrics.get('efficiency_ratio', 0.0)
        efficiency_score = min(1.0, efficiency_ratio)

        # 2. 相对误差评分（40%权重）：越接近最优越高
        relative_gap = metrics.get('relative_gap', None)
        if relative_gap is None:
            gap_score = 0.0
        else:
            gap_score = max(0.0, 1.0 - float(relative_gap))

        # 3. 稳定性评分（10%权重）：奖励方差小，表示轨迹稳定
        rewards = trajectory.get('rewards', [])
        if len(rewards) > 1:
            reward_variance = np.var(rewards)
            stability_score = 1.0 / (1.0 + reward_variance / 10.0)
        else:
            stability_score = 1.0

        # 4. 多样性评分（10%权重）：动作多样性
        actions = trajectory.get('actions', [])
        if len(actions) > 0:
            unique_actions = len(set(str(a) for a in actions))
            diversity_score = unique_actions / len(actions)
        else:
            diversity_score = 0.0

        # 5. 完整性评分（10%权重）：轨迹是否完整
        n_steps = len(actions)
        n_jobs = metrics.get('n_jobs', 5)
        n_machines = metrics.get('n_machines', 5)
        expected_steps = n_jobs * n_machines
        completeness_score = min(1.0, n_steps / expected_steps)

        # 6. 综合评分
        overall_score = (
            0.3 * efficiency_score +
            0.4 * gap_score +
            0.1 * stability_score +
            0.1 * diversity_score +
            0.1 * completeness_score
        )

        # 质量详情
        quality_details = {
            'efficiency_score': efficiency_score,
            'gap_score': gap_score,
            'stability_score': stability_score,
            'diversity_score': diversity_score,
            'completeness_score': completeness_score,
            'overall_score': overall_score
        }

        return overall_score, quality_details

    def normalize_trajectory(self, trajectory, normalization_ratio=0.5):
        """
        归一化轨迹数据：仅对状态进行归一化，提高BC训练效果。

        Args:
            trajectory: 原始轨迹
            normalization_ratio: 归一化比例（0.5表示50%的轨迹会被归一化）

        Returns:
            normalized_trajectory: 归一化后的轨迹
        """
        # 随机决定是否归一化
        if np.random.random() > normalization_ratio:
            return trajectory

        # 复制轨迹，避免修改原始数据
        normalized = trajectory.copy()

        # 仅对状态进行归一化
        states = trajectory.get('states', [])
        if states and len(states) > 0:
            normalized_states = []

            # 对每个时间步的状态进行归一化
            for state in states:
                if not isinstance(state, dict):
                    # 如果状态不是字典类型，跳过归一化
                    normalized_states.append(state)
                    continue

                normalized_state = {}

                # 对状态中的每个组件进行归一化
                for key, value in state.items():
                    if isinstance(value, np.ndarray) and value.ndim > 0:
                        # 避免对已经归一化的组件再次归一化
                        if 'normalized' in key:
                            normalized_state[key] = value
                            continue

                        # 使用Z-score归一化
                        mean_val = np.mean(value)
                        std_val = np.std(value)

                        # 避免除零
                        std_val = max(std_val, 1e-8)

                        # 应用归一化
                        normalized_value = (value - mean_val) / std_val
                        normalized_state[key] = normalized_value
                    else:
                        # 非数组类型保持不变
                        normalized_state[key] = value

                normalized_states.append(normalized_state)

            # 更新归一化后的状态
            normalized['states'] = normalized_states

            # 更新轨迹指标
            normalized['metrics'] = normalized['metrics'].copy()
            normalized['metrics']['normalized'] = True
            normalized['metrics']['normalization_type'] = 'state_zscore'

            print(f"🔧 Applied state normalization to trajectory")

        return normalized

    def _convert_to_serializable(self, obj):
        """
        Recursively convert numpy arrays to lists for JSON serialization.

        Args:
            obj: Object to convert (can be nested dict, list, np.ndarray, etc.)

        Returns:
            Serializable object with all numpy arrays converted to lists
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        else:
            # For any other type, try to convert to string representation
            return str(obj)

    def save_trajectories(self, filepath, append=False):
        """
        Save collected trajectories to a file.

        Args:
            filepath: Path to save the trajectories
            append: If True, append trajectories to existing file; if False, overwrite existing file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if append:
            serializable_traj = self._convert_to_serializable(self.trajectories[-1])
            self.append_trajectory_to_file(filepath, serializable_traj)
        else:
            # Overwrite mode: save all trajectories
            if str(filepath).lower().endswith(".jsonl"):
                with open(filepath, 'w', encoding='utf-8') as f:
                    for traj in self.trajectories:
                        serializable_traj = self._convert_to_serializable(traj)
                        f.write(json.dumps(serializable_traj, ensure_ascii=False) + "\n")
                print(f"Saved {len(self.trajectories)} trajectories to {filepath}")
            else:
                serializable_trajectories = []
                for traj in self.trajectories:
                    serializable_traj = self._convert_to_serializable(traj)
                    serializable_trajectories.append(serializable_traj)

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(serializable_trajectories, f, ensure_ascii=False)

                print(f"Saved {len(self.trajectories)} trajectories to {filepath}")

    def load_trajectories(self, filepath):
        """
        Load trajectories from a file.

        Args:
            filepath: Path to load the trajectories from
        """
        self.trajectories = self.load_trajectories_from_file(filepath)
        print(f"Loaded {len(self.trajectories)} trajectories from {filepath}")

    def load_trajectories_from_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            first_non_ws = ''
            while True:
                ch = f.read(1)
                if not ch:
                    break
                if not ch.isspace():
                    first_non_ws = ch
                    break
            f.seek(0)
            if first_non_ws == '[':
                return json.load(f)
            out = []
            for line in f:
                s = line.strip()
                if not s:
                    continue
                out.append(json.loads(s))
            return out

    def append_trajectory_to_file(self, filepath, trajectory):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_non_ws = ''
                    while True:
                        ch = f.read(1)
                        if not ch:
                            break
                        if not ch.isspace():
                            first_non_ws = ch
                            break
                if first_non_ws == '[':
                    existing = self.load_trajectories_from_file(filepath)
                    with open(filepath, 'w', encoding='utf-8') as wf:
                        for item in existing:
                            wf.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception:
                pass

        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
        print(f"Appended trajectory to {filepath}")

class ORToolsAgent:
    """Agent that uses OR-Tools CP-SAT solver to make optimal or near-optimal scheduling decisions."""

    def __init__(self, env):
        self.env = env
        # Import solve_jssp function with flexible path handling
        try:
            try:
                from Envs.JSSPEnv.JSSPEnv import solve_jssp
            except ImportError:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from Envs.JSSPEnv.JSSPEnv import solve_jssp
            self.solve_jssp = solve_jssp
        except Exception as e:
            print(f"Warning: Failed to import solve_jssp: {e}")
            self.solve_jssp = None

        # State cache
        self.optimal_job_order = None # 这是一个按时间排序的作业列表 [job_id, job_id, ...]
        self.epsilon = 0.0
        self.random_action_count = 0

    def reset(self):
        """Reset the agent's state to prepare for a new trajectory."""
        self.optimal_job_order = None # 清空执行列表
        self.epsilon = 0.0
        self.random_action_count = 0

    def extract_instance_from_state(self, state):
        """Extract JSSP instance data from the environment state."""
        # 优先使用 current_instance (原始未归一化的实例)
        if hasattr(self.env, 'current_instance') and self.env.current_instance is not None:
            return self.env.current_instance
        
        # 其次尝试 instance
        if hasattr(self.env, 'instance'):
            return self.env.instance
            
        raise ValueError("Could not extract valid JSSP instance from environment.")

    def _generate_precise_job_order(self, instance, machine_schedule):
        """
        将 OR-Tools 的机器调度表转换为按时间顺序排列的作业执行列表。
        这确保了 Agent 严格按照时间线执行，而不是随机选择空闲机器。
        """
        n, two_m = instance.shape
        m = two_m // 2
        
        # 追踪状态
        job_next = [0] * n
        job_finish = [0] * n
        machine_finish = [0] * m
        machine_ptr = [0] * m
        
        execution_order = [] # 这是我们要生成的列表
        
        scheduled_ops = 0
        total_ops = n * m
        
        # 模拟整个调度过程
        while scheduled_ops < total_ops:
            available = [] # (start_time, machine, job)
            
            # 检查每台机器的下一个任务是否这就绪
            for machine in range(m):
                if machine_ptr[machine] < len(machine_schedule[machine]):
                    job = machine_schedule[machine][machine_ptr[machine]]
                    op_idx = job_next[job]
                    
                    # 确保作业的下一道工序确实是在这台机器上
                    if op_idx < m and int(instance[job, 2 * op_idx]) == machine:
                        # 计算最早开始时间：max(作业前置工序完成时间, 机器空闲时间)
                        start_time = max(job_finish[job], machine_finish[machine])
                        available.append((start_time, machine, job))
            
            if not available:
                break # 应该不会发生，除非调度不可行
                
            # 关键步骤：按开始时间排序！优先执行开始时间最早的任务
            # 如果开始时间相同，machine index小的优先（作为打破平局的规则）
            available.sort(key=lambda x: (x[0], x[1]))
            
            start_time, machine, job = available[0]
            
            # 加入执行列表
            execution_order.append(job)
            
            # 更新状态
            op_idx = job_next[job]
            proc_time = instance[job, 2 * op_idx + 1]
            finish_time = start_time + proc_time
            
            job_finish[job] = finish_time
            machine_finish[machine] = finish_time
            job_next[job] += 1
            machine_ptr[machine] += 1
            scheduled_ops += 1
            
        return execution_order

    def get_available_jobs(self, state):
        """Get available jobs from environment."""
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        return self.env.action_space if isinstance(self.env.action_space, list) else []

    def select_action(self, state, epsilon=None):
        """
        Select action using the pre-calculated time-ordered list.
        修正核心逻辑：
        1. epsilon-greedy 随机选择后，同步更新 optimal_job_order
        2. 如果目标作业不可用，选择其他可用作业（而不是返回 -1 等待）
        """
        available_jobs = self.get_available_jobs(state)
        if not available_jobs:
            return -1

        eps = self.epsilon if epsilon is None else float(epsilon)
        if eps > 0 and np.random.random() < eps:
            self.random_action_count += 1
            chosen_job = np.random.choice(available_jobs)
            # ✅ 关键修复：随机选择后，从 optimal_job_order 中移除该作业的所有出现
            # 这样可以保持调度的一致性
            if self.optimal_job_order is not None:
                # 移除第一次出现（对应当前要执行的操作）
                if chosen_job in self.optimal_job_order:
                    self.optimal_job_order.remove(chosen_job)
            return chosen_job

        if self.solve_jssp is None:
            return available_jobs[0]

        try:
            if self.optimal_job_order is None:
                current_instance = self.extract_instance_from_state(state)
                optimal_makespan, machine_schedule = self.solve_jssp(current_instance)
                
                if machine_schedule:
                    self.optimal_job_order = self._generate_precise_job_order(current_instance, machine_schedule)
                else:
                    return available_jobs[0]

            if not self.optimal_job_order:
                return available_jobs[0]

            target_job = self.optimal_job_order[0]
            
            if target_job in available_jobs:
                self.optimal_job_order.pop(0)
                return target_job
            else:
                # ✅ 关键修复：如果目标作业不可用，但有其他作业可用
                # 选择 optimal_job_order 中最早出现的可用作业
                # 这样可以尽量保持接近最优调度，而不是白白等待
                for job in self.optimal_job_order:
                    if job in available_jobs:
                        self.optimal_job_order.remove(job)
                        return job
                
                # 如果 optimal_job_order 中没有可用作业，选择任意可用作业
                # 这种情况理论上不应该发生，但作为保险
                chosen_job = available_jobs[0]
                return chosen_job

        except Exception as e:
            print(f"Error in ORToolsAgent: {e}")
            import traceback
            traceback.print_exc()
            return available_jobs[0]



class RandomAgent:
    """Simple agent that selects actions randomly."""

    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        if isinstance(self.env.action_space, list):
            # For real JSSPEnv, action_space is a list of available job indices
            return np.random.choice(self.env.action_space)
        else:
            # Fallback for environments with sample() method
            return self.env.action_space.sample()


class FixedScheduleAgent:
    """
    固定调度Agent - 严格按照预计算的调度执行。
    
    这是最简单、最可靠的方法！
    直接使用 OR-Tools 的解（最优或次优），按该调度执行。
    
    关键：调度在初始化时就确定，执行时严格遵循。
    """
    
    def __init__(self, env, schedule=None, job_order=None):
        """
        Args:
            env: JSSP环境
            schedule: OR-Tools 返回的机器调度表（可选）
            job_order: 预计算的作业执行顺序（可选）
        """
        self.env = env
        self.schedule = schedule
        self.job_order = list(job_order) if job_order else None
        self.step_count = 0
    
    def reset(self):
        """重置Agent状态"""
        self.step_count = 0
        # 不重置 job_order，因为它是初始化时设定的
    
    def set_job_order(self, job_order):
        """设置作业执行顺序"""
        self.job_order = list(job_order)
    
    def get_available_jobs(self, state):
        """获取当前可用的作业"""
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        return self.env.action_space if isinstance(self.env.action_space, list) else []
    
    def select_action(self, state, epsilon=None):
        """严格按照预计算的顺序执行"""
        available_jobs = self.get_available_jobs(state)
        if not available_jobs:
            return -1
        
        if not self.job_order:
            return available_jobs[0]
        
        self.step_count += 1
        
        # 找到序列中下一个可用的作业
        for i, job in enumerate(self.job_order):
            if job in available_jobs:
                self.job_order.pop(i)
                return job
        
        # 如果预计算序列中没有可用作业，返回-1等待
        return -1


class MultiSolutionAgent:
    """
    多解Agent - 使用 OR-Tools 的多个解来生成不同质量的轨迹。
    
    这是用户建议的方法：
    1. 让 OR-Tools 在搜索过程中收集多个解（从次优到最优）
    2. 选择其中一个解来执行
    3. 不同的解对应不同的效率等级
    
    优点：
    - 所有解都是 OR-Tools 认可的"可行解"
    - 效率差异来自搜索过程的自然变化，不是人工加噪声
    - 更符合实际的"次优决策"场景
    """
    
    def __init__(self, env, solution_index=0, solutions=None):
        """
        Args:
            env: JSSP环境
            solution_index: 使用第几个解（0=最优，1=次优，...）
            solutions: 预先获取的多个解列表
        """
        self.env = env
        self.solution_index = solution_index
        self.solutions = solutions
        self.job_order = None
        self.step_count = 0
        
        try:
            from Envs.JSSPEnv.JSSPEnv import solve_jssp, solve_jssp_multiple
            self.solve_jssp = solve_jssp
            self.solve_jssp_multiple = solve_jssp_multiple
        except Exception as e:
            print(f"Warning: Failed to import solve_jssp: {e}")
            self.solve_jssp = None
            self.solve_jssp_multiple = None
    
    def reset(self):
        """重置Agent状态"""
        self.job_order = None
        self.step_count = 0
    
    def _generate_job_order_from_schedule(self, instance, machine_schedule):
        """将机器调度转换为作业执行顺序（与 ORToolsAgent 相同）"""
        n, two_m = instance.shape
        m = two_m // 2
        
        job_next = [0] * n
        job_finish = [0] * n
        machine_finish = [0] * m
        machine_ptr = [0] * m
        
        execution_order = []
        scheduled_ops = 0
        total_ops = n * m
        
        while scheduled_ops < total_ops:
            available = []
            
            for machine in range(m):
                if machine_ptr[machine] < len(machine_schedule[machine]):
                    job = machine_schedule[machine][machine_ptr[machine]]
                    op_idx = job_next[job]
                    
                    if op_idx < m and int(instance[job, 2 * op_idx]) == machine:
                        start_time = max(job_finish[job], machine_finish[machine])
                        available.append((start_time, machine, job))
            
            if not available:
                break
            
            available.sort(key=lambda x: (x[0], x[1]))
            start_time, machine, job = available[0]
            execution_order.append(job)
            
            op_idx = job_next[job]
            proc_time = instance[job, 2 * op_idx + 1]
            finish_time = start_time + proc_time
            
            job_finish[job] = finish_time
            machine_finish[machine] = finish_time
            job_next[job] += 1
            machine_ptr[machine] += 1
            scheduled_ops += 1
        
        return execution_order
    
    def get_available_jobs(self, state):
        """获取当前可用的作业"""
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        return self.env.action_space if isinstance(self.env.action_space, list) else []
    
    def select_action(self, state, epsilon=None):
        """选择动作"""
        available_jobs = self.get_available_jobs(state)
        if not available_jobs:
            return -1
        
        self.step_count += 1
        
        # 首次调用时，获取解并生成执行顺序
        if self.job_order is None:
            try:
                instance = self.env.current_instance
                
                if self.solutions:
                    # 使用预先获取的解
                    idx = min(self.solution_index, len(self.solutions) - 1)
                    _, schedule = self.solutions[idx]
                elif self.solve_jssp_multiple:
                    # 获取多个解
                    solutions = self.solve_jssp_multiple(instance, max_solutions=5)
                    if solutions:
                        idx = min(self.solution_index, len(solutions) - 1)
                        _, schedule = solutions[idx]
                    else:
                        # 降级到单解
                        _, schedule = self.solve_jssp(instance)
                else:
                    _, schedule = self.solve_jssp(instance)
                
                self.job_order = self._generate_job_order_from_schedule(instance, schedule)
                
            except Exception as e:
                print(f"Error getting solution: {e}")
                return available_jobs[0]
        
        if not self.job_order:
            return available_jobs[0]
        
        # 按顺序执行
        target_job = self.job_order[0]
        
        if target_job in available_jobs:
            self.job_order.pop(0)
            return target_job
        else:
            # 找序列中最早的可用作业
            for i, job in enumerate(self.job_order):
                if job in available_jobs:
                    self.job_order.pop(i)
                    return job
            return -1


class AdaptiveReplanningAgent:
    """
    [已废弃] 自适应重新规划Agent。
    
    警告：这个方法的 _get_optimal_action_ranking 逻辑有问题，
    会导致效率严重下降。推荐使用 FixedScheduleAgent 或 MultiSolutionAgent。
    """
    
    def __init__(self, env, noise_ratio=0.0, noise_type='second_best'):
        print("⚠️ 警告: AdaptiveReplanningAgent 已废弃，推荐使用 FixedScheduleAgent")
        self.env = env
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type
        
        try:
            from Envs.JSSPEnv.JSSPEnv import solve_jssp
            self.solve_jssp = solve_jssp
        except Exception as e:
            self.solve_jssp = None
        
        self.noise_count = 0
        self.total_steps = 0
        
        # 使用 ORToolsAgent 的逻辑
        self.optimal_job_order = None
    
    def reset(self):
        self.noise_count = 0
        self.total_steps = 0
        self.optimal_job_order = None
    
    def _generate_precise_job_order(self, instance, machine_schedule):
        """与 ORToolsAgent 相同的逻辑"""
        n, two_m = instance.shape
        m = two_m // 2
        
        job_next = [0] * n
        job_finish = [0] * n
        machine_finish = [0] * m
        machine_ptr = [0] * m
        
        execution_order = []
        scheduled_ops = 0
        total_ops = n * m
        
        while scheduled_ops < total_ops:
            available = []
            
            for machine in range(m):
                if machine_ptr[machine] < len(machine_schedule[machine]):
                    job = machine_schedule[machine][machine_ptr[machine]]
                    op_idx = job_next[job]
                    
                    if op_idx < m and int(instance[job, 2 * op_idx]) == machine:
                        start_time = max(job_finish[job], machine_finish[machine])
                        available.append((start_time, machine, job))
            
            if not available:
                break
            
            available.sort(key=lambda x: (x[0], x[1]))
            start_time, machine, job = available[0]
            execution_order.append(job)
            
            op_idx = job_next[job]
            proc_time = instance[job, 2 * op_idx + 1]
            finish_time = start_time + proc_time
            
            job_finish[job] = finish_time
            machine_finish[machine] = finish_time
            job_next[job] += 1
            machine_ptr[machine] += 1
            scheduled_ops += 1
        
        return execution_order
    
    def get_available_jobs(self, state):
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        return self.env.action_space if isinstance(self.env.action_space, list) else []
    
    def select_action(self, state, epsilon=None):
        available_jobs = self.get_available_jobs(state)
        if not available_jobs:
            return -1
        
        self.total_steps += 1
        
        # 首次调用时计算最优序列
        if self.optimal_job_order is None:
            try:
                instance = self.env.current_instance
                _, schedule = self.solve_jssp(instance)
                if schedule:
                    self.optimal_job_order = self._generate_precise_job_order(instance, schedule)
            except Exception as e:
                return available_jobs[0]
        
        if not self.optimal_job_order:
            return available_jobs[0]
        
        target_job = self.optimal_job_order[0]
        
        # 噪声：有概率选择非最优
        if self.noise_ratio > 0 and len(available_jobs) > 1 and np.random.random() < self.noise_ratio:
            self.noise_count += 1
            non_target = [j for j in available_jobs if j != target_job]
            if non_target:
                chosen = np.random.choice(non_target)
                # 同步更新序列
                if chosen in self.optimal_job_order:
                    self.optimal_job_order.remove(chosen)
                return chosen
        
        if target_job in available_jobs:
            self.optimal_job_order.pop(0)
            return target_job
        else:
            for i, job in enumerate(self.optimal_job_order):
                if job in available_jobs:
                    self.optimal_job_order.pop(i)
                    return job
            return -1


class SequenceSwapAgent:
    """
    [已废弃] 基于序列交换的次优解生成Agent。
    
    警告：这个方法在JSSP问题上效果很差，因为JSSP是高度耦合的问题。
    即使只交换2次，也可能导致效率暴跌到20-30%。
    
    推荐使用 AdaptiveReplanningAgent 代替。
    """
    
    def __init__(self, env, n_swaps=0, swap_distance=1):
        print("⚠️ 警告: SequenceSwapAgent 已废弃，推荐使用 AdaptiveReplanningAgent")
        self.env = env
        self.n_swaps = n_swaps
        self.swap_distance = swap_distance
        
        try:
            from Envs.JSSPEnv.JSSPEnv import solve_jssp
            self.solve_jssp = solve_jssp
        except Exception as e:
            self.solve_jssp = None
        
        self.job_order = None
        self.swaps_made = []
    
    def reset(self):
        self.job_order = None
        self.swaps_made = []
    
    def _generate_optimal_job_order(self, instance, machine_schedule):
        n, two_m = instance.shape
        m = two_m // 2
        
        job_next = [0] * n
        job_finish = [0] * n
        machine_finish = [0] * m
        machine_ptr = [0] * m
        
        execution_order = []
        scheduled_ops = 0
        total_ops = n * m
        
        while scheduled_ops < total_ops:
            available = []
            
            for machine in range(m):
                if machine_ptr[machine] < len(machine_schedule[machine]):
                    job = machine_schedule[machine][machine_ptr[machine]]
                    op_idx = job_next[job]
                    
                    if op_idx < m and int(instance[job, 2 * op_idx]) == machine:
                        start_time = max(job_finish[job], machine_finish[machine])
                        available.append((start_time, machine, job))
            
            if not available:
                break
            
            available.sort(key=lambda x: (x[0], x[1]))
            start_time, machine, job = available[0]
            execution_order.append(job)
            
            op_idx = job_next[job]
            proc_time = instance[job, 2 * op_idx + 1]
            finish_time = start_time + proc_time
            
            job_finish[job] = finish_time
            machine_finish[machine] = finish_time
            job_next[job] += 1
            machine_ptr[machine] += 1
            scheduled_ops += 1
        
        return execution_order
    
    def get_available_jobs(self, state):
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        return self.env.action_space if isinstance(self.env.action_space, list) else []
    
    def select_action(self, state, epsilon=None):
        available_jobs = self.get_available_jobs(state)
        if not available_jobs:
            return -1
        
        # 直接使用 OR-Tools 最优（不再交换，因为交换会导致效率暴跌）
        if self.job_order is None:
            if self.solve_jssp is None:
                return available_jobs[0]
            
            try:
                if hasattr(self.env, 'current_instance') and self.env.current_instance is not None:
                    instance = self.env.current_instance
                elif hasattr(self.env, 'instance'):
                    instance = self.env.instance
                else:
                    return available_jobs[0]
                
                optimal_makespan, machine_schedule = self.solve_jssp(instance)
                
                if machine_schedule:
                    self.job_order = self._generate_optimal_job_order(instance, machine_schedule)
                else:
                    return available_jobs[0]
            except Exception as e:
                print(f"Error in SequenceSwapAgent: {e}")
                return available_jobs[0]
        
        if not self.job_order:
            return available_jobs[0]
        
        target_job = self.job_order[0]
        
        if target_job in available_jobs:
            self.job_order.pop(0)
            return target_job
        else:
            for job in self.job_order:
                if job in available_jobs:
                    self.job_order.remove(job)
                    return job
            return -1


class ReplanningAgent:
    """
    重新规划Agent：每次执行随机动作后，重新调用OR-Tools求解。
    
    这是最稳健的次优解生成方法：
    - 在关键决策点引入噪声（选择随机动作）
    - 但随后立即重新规划，确保后续仍然是（局部）最优
    
    优点：即使引入噪声，效率也不会暴跌
    缺点：计算开销大（多次调用OR-Tools）
    """
    
    def __init__(self, env, noise_probability=0.1, noise_positions=None):
        """
        Args:
            env: JSSP环境
            noise_probability: 在每个决策点引入噪声的概率（如果noise_positions为None）
            noise_positions: 固定的噪声位置列表（第几步引入噪声）
        """
        self.env = env
        self.noise_probability = noise_probability
        self.noise_positions = set(noise_positions) if noise_positions else None
        
        try:
            from Envs.JSSPEnv.JSSPEnv import solve_jssp
            self.solve_jssp = solve_jssp
        except Exception as e:
            print(f"Warning: Failed to import solve_jssp: {e}")
            self.solve_jssp = None
        
        self.step_count = 0
        self.noise_introduced = []
    
    def reset(self):
        """重置Agent状态"""
        self.step_count = 0
        self.noise_introduced = []
    
    def get_available_jobs(self, state):
        """获取当前可用的作业"""
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        return self.env.action_space if isinstance(self.env.action_space, list) else []
    
    def _should_introduce_noise(self):
        """判断当前步是否应该引入噪声"""
        if self.noise_positions is not None:
            return self.step_count in self.noise_positions
        else:
            return np.random.random() < self.noise_probability
    
    def _solve_current_state(self):
        """
        基于当前部分调度，求解剩余最优。
        注意：这是一个简化实现，假设环境有current_instance属性。
        """
        if self.solve_jssp is None:
            return None
        
        try:
            if hasattr(self.env, 'current_instance') and self.env.current_instance is not None:
                instance = self.env.current_instance
            else:
                return None
            
            # 调用OR-Tools求解
            _, machine_schedule = self.solve_jssp(instance)
            
            if machine_schedule:
                # 生成作业执行顺序
                return self._generate_job_order_from_schedule(instance, machine_schedule)
            return None
        except Exception as e:
            print(f"Error solving current state: {e}")
            return None
    
    def _generate_job_order_from_schedule(self, instance, machine_schedule):
        """将机器调度转换为作业执行顺序"""
        n, two_m = instance.shape
        m = two_m // 2
        
        job_next = [0] * n
        job_finish = [0] * n
        machine_finish = [0] * m
        machine_ptr = [0] * m
        
        execution_order = []
        scheduled_ops = 0
        total_ops = n * m
        
        while scheduled_ops < total_ops:
            available = []
            
            for machine in range(m):
                if machine_ptr[machine] < len(machine_schedule[machine]):
                    job = machine_schedule[machine][machine_ptr[machine]]
                    op_idx = job_next[job]
                    
                    if op_idx < m and int(instance[job, 2 * op_idx]) == machine:
                        start_time = max(job_finish[job], machine_finish[machine])
                        available.append((start_time, machine, job))
            
            if not available:
                break
            
            available.sort(key=lambda x: (x[0], x[1]))
            start_time, machine, job = available[0]
            execution_order.append(job)
            
            op_idx = job_next[job]
            proc_time = instance[job, 2 * op_idx + 1]
            finish_time = start_time + proc_time
            
            job_finish[job] = finish_time
            machine_finish[machine] = finish_time
            job_next[job] += 1
            machine_ptr[machine] += 1
            scheduled_ops += 1
        
        return execution_order
    
    def select_action(self, state, epsilon=None):
        """选择动作：可能引入噪声，但随后重新规划"""
        available_jobs = self.get_available_jobs(state)
        if not available_jobs:
            return -1
        
        self.step_count += 1
        
        # 判断是否引入噪声
        if self._should_introduce_noise() and len(available_jobs) > 1:
            # 随机选择一个非最优动作
            optimal_order = self._solve_current_state()
            
            if optimal_order and optimal_order[0] in available_jobs:
                optimal_action = optimal_order[0]
                # 选择一个不是最优的动作
                non_optimal = [j for j in available_jobs if j != optimal_action]
                if non_optimal:
                    chosen = np.random.choice(non_optimal)
                    self.noise_introduced.append(self.step_count)
                    return chosen
        
        # 正常情况：选择当前最优动作
        optimal_order = self._solve_current_state()
        
        if optimal_order:
            # 跳过已完成的作业（基于当前环境状态）
            for job in optimal_order:
                if job in available_jobs:
                    return job
        
        # 如果求解失败，选择第一个可用作业
        return available_jobs[0]


def _trajectory_gap(trajectory):
    metrics = trajectory.get("metrics", {}) if isinstance(trajectory, dict) else {}
    gap = metrics.get("relative_gap", None)
    if gap is not None:
        try:
            return float(gap)
        except Exception:
            return None
    opt = metrics.get("optimal_makespan", None)
    actual = metrics.get("actual_makespan", None)
    try:
        opt = float(opt)
        actual = float(actual)
    except Exception:
        return None
    if opt <= 0:
        return None
    return (actual - opt) / (opt + 1e-8)


def _trajectory_decision_field_error(trajectory):
    required_state_keys = {
        "job_next_op",
        "job_completion_mask",
        "remaining_processing_times",
        "normalized_machine_availability",
        "machine_busy_mask",
        "overall_info",
        "machine_sequence",
        "scheduled_machine_adj",
        "legal_action_mask",
        "current_time",
    }
    states = trajectory.get("states", []) if isinstance(trajectory, dict) else []
    actions = trajectory.get("actions", []) if isinstance(trajectory, dict) else []
    rewards = trajectory.get("rewards", []) if isinstance(trajectory, dict) else []
    returns_to_go = trajectory.get("returns_to_go", []) if isinstance(trajectory, dict) else []
    timesteps = trajectory.get("timesteps", []) if isinstance(trajectory, dict) else []
    if not (len(states) == len(actions) + 1 == len(returns_to_go) == len(timesteps)):
        return (
            f"length mismatch: states={len(states)}, actions={len(actions)}, "
            f"rtg={len(returns_to_go)}, timesteps={len(timesteps)}"
        )
    if len(rewards) != len(actions):
        return f"reward/action length mismatch: rewards={len(rewards)}, actions={len(actions)}"
    expected_steps = 0
    metrics = trajectory.get("metrics", {})
    try:
        expected_steps = int(metrics.get("n_jobs", 0)) * int(metrics.get("n_machines", 0))
    except Exception:
        expected_steps = 0
    if expected_steps > 0:
        final_state = states[-1] if states else {}
        final_ops = final_state.get("job_next_op", final_state.get("job_status", []))
        final_ops_arr = np.asarray([] if final_ops is None else final_ops).reshape(-1)
        final_ops_sample = final_ops_arr[: min(10, final_ops_arr.size)].tolist()
        try:
            n_machines = int(metrics.get("n_machines", 0))
            final_complete = final_ops_arr.size > 0 and bool(np.all(final_ops_arr.astype(float) >= n_machines))
        except Exception:
            final_complete = False
        if not final_complete:
            try:
                op_min = float(np.min(final_ops_arr)) if final_ops_arr.size else None
                op_max = float(np.max(final_ops_arr)) if final_ops_arr.size else None
            except Exception:
                op_min = op_max = None
            final_keys = sorted(final_state.keys()) if isinstance(final_state, dict) else []
            return (
                f"final state incomplete: n_machines={metrics.get('n_machines')}, "
                f"op_min={op_min}, op_max={op_max}, op_count={final_ops_arr.size}, "
                f"op_shape={list(np.asarray([] if final_ops is None else final_ops).shape)}, "
                f"op_sample={final_ops_sample}, final_keys={final_keys}"
            )
        # Some older environments counted only completed jobs in ``completed_operations``.
        # The final job_next_op vector is the authoritative completion check.
        if len(actions) > expected_steps:
            return f"too many actions: actions={len(actions)}, expected={expected_steps}"
    for idx, state in enumerate(states):
        if not isinstance(state, dict) or not required_state_keys.issubset(state.keys()):
            missing = sorted(required_state_keys - set(state.keys())) if isinstance(state, dict) else sorted(required_state_keys)
            return f"state {idx} missing keys: {missing}"
    return None


def _trajectory_has_required_decision_fields(trajectory):
    return _trajectory_decision_field_error(trajectory) is None


def _sample_tier_epsilon(config):
    if "epsilon" in config:
        return float(config.get("epsilon", 0.0))
    eps_min = float(config.get("epsilon_min", 0.0) or 0.0)
    eps_max = float(config.get("epsilon_max", eps_min) or eps_min)
    if eps_max <= eps_min:
        return eps_min
    return float(np.random.uniform(eps_min, eps_max))


def collect_tiered_trajectories(
    env,
    num_trajectories=100,
    max_steps=100,
    output_path=None,
    tier_distribution=None,
    processing_time_mutation_ratio=0.0,
    processing_time_mutation_pct=0.0,
    save_light_trajectories=False,
    validate_light_trajectories=False,
):
    """
    分层轨迹收集函数。

    每条轨迹先由 ORToolsAgent 或 epsilon 扰动版本生成，再按实际 relative_gap
    落入对应质量层，避免只靠 epsilon 猜测轨迹质量。
    
    Args:
        env: JSSP环境模板
        num_trajectories: 总轨迹数
        max_steps: 每条轨迹最大步数
        output_path: 保存路径
        tier_distribution: 分层分布配置
        processing_time_mutation_ratio: 处理时间变异的轨迹比例
        processing_time_mutation_pct: 处理时间变异幅度
        
    Returns:
        收集的轨迹列表
    """
    
    if tier_distribution is None:
        tier_distribution = {
            "expert": {
                "ratio": 0.70,
                "epsilon": 0.0,
                "gap_range": (0.0, 0.05),
                "mutation": False,
            },
            "near_optimal": {
                "ratio": 0.20,
                "epsilon_min": 0.05,
                "epsilon_max": 0.15,
                "gap_range": (0.05, 0.10),
                "mutation": True,
            },
            "medium": {
                "ratio": 0.10,
                "epsilon_min": 0.15,
                "epsilon_max": 0.35,
                "gap_range": (0.10, 0.25),
                "mutation": True,
            },
        }
    
    # 计算每层目标数量
    tier_targets = {}
    remaining = num_trajectories
    for i, (tier_name, config) in enumerate(tier_distribution.items()):
        if i == len(tier_distribution) - 1:
            tier_targets[tier_name] = remaining
        else:
            count = int(num_trajectories * config['ratio'])
            tier_targets[tier_name] = count
            remaining -= count
    
    print(f"\n{'='*60}")
    print(f"分层轨迹收集（按实际 gap 过滤）")
    print(f"{'='*60}")
    for tier_name, count in tier_targets.items():
        config = tier_distribution[tier_name]
        eps = config.get('epsilon', None)
        if eps is None:
            eps = f"{float(config.get('epsilon_min', 0.0)):.2f}-{float(config.get('epsilon_max', 0.0)):.2f}"
        else:
            eps = f"{float(eps):.2f}"
        mut = config.get('mutation', False)
        gap_range = config.get("gap_range", (0.0, 1.0))
        print(f"  {tier_name}: {count} 条轨迹 (epsilon={eps}, gap=[{gap_range[0]:.0%},{gap_range[1]:.0%}), mutation={mut})")
    print(f"{'='*60}\n")
    
    trajectories = []
    tier_collected = {tier: 0 for tier in tier_distribution}
    tier_efficiency_sum = {tier: 0.0 for tier in tier_distribution}
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            pass
    
    total_collected = 0
    max_attempts = max(100, num_trajectories * 60)
    attempts = 0
    seen_instances = set()
    
    while total_collected < num_trajectories and attempts < max_attempts:
        attempts += 1
        
        # 选择还需要轨迹的层级
        available_tiers = [
            tier for tier, count in tier_collected.items()
            if count < tier_targets[tier]
        ]
        
        if not available_tiers:
            break
        
        # 随机选择一个层级（按剩余需求加权）
        weights = [tier_targets[t] - tier_collected[t] for t in available_tiers]
        total_weight = sum(weights)
        if total_weight == 0:
            break
        probs = [w / total_weight for w in weights]
        tier_name = np.random.choice(available_tiers, p=probs)
        config = tier_distribution[tier_name]
        
        # 创建新环境
        new_env = type(env)(
            n_job_range=env.n_job_range,
            n_mac_range=env.n_mac_range,
            seed=None
        )
        
        inherited_max_steps = int(getattr(env, "max_steps", 0) or 0)
        env_n_jobs = int(getattr(new_env, "n_jobs", 0) or max(getattr(new_env, "n_job_range", [0])))
        env_n_machines = int(getattr(new_env, "n_machines", 0) or max(getattr(new_env, "n_mac_range", [0])))
        new_env.max_steps = max(int(max_steps), inherited_max_steps, env_n_jobs * env_n_machines)
        
        use_mutation = bool(config.get('mutation', False))
        if use_mutation or (processing_time_mutation_ratio > 0 and np.random.random() < processing_time_mutation_ratio):
            new_env.processing_time_mutation_pct = float(config.get("mutation_pct", processing_time_mutation_pct) or 0.0)
        else:
            new_env.processing_time_mutation_pct = 0.0

        epsilon = _sample_tier_epsilon(config)
        agent = ORToolsAgent(new_env)
        agent.epsilon = epsilon
        agent.reset()
        
        # 收集轨迹
        collector = JSSPTrajectoryCollector(new_env)
        trajectory = collector.collect_trajectory(agent, max_steps=max_steps, verbose=False)
        
        # 检查轨迹质量
        metrics = trajectory.get('metrics', {})
        efficiency = metrics.get('efficiency_ratio', 0.0)
        completed = metrics.get('completed', False)
        gap = _trajectory_gap(trajectory)

        if not completed:
            if attempts % 25 == 0:
                print(f"  跳过未完成轨迹 (attempt={attempts})")
            continue
        light_to_save = None
        if save_light_trajectories:
            try:
                light_to_save = make_light_jssp_trajectory(trajectory, env=new_env)
                if validate_light_trajectories:
                    validate_light_replay_complete(light_to_save)
            except Exception as e:
                print(f"  跳过light replay验证失败的轨迹 (attempt={attempts}) 原因: {e}")
                continue
        else:
            field_error = _trajectory_decision_field_error(trajectory)
            if field_error is not None:
                repaired_candidate = repair_jssp_decision_trajectory(trajectory, rtg_mode="future_return")
                repaired_error = _trajectory_decision_field_error(repaired_candidate)
                if repaired_error is None:
                    trajectory = repaired_candidate
                    metrics = trajectory.get('metrics', {})
                    efficiency = metrics.get('efficiency_ratio', efficiency)
                    completed = metrics.get('completed', completed)
                    gap = _trajectory_gap(trajectory)
                else:
                    print(
                        f"  跳过字段不完整或长度不一致的轨迹 (attempt={attempts}) "
                        f"原因: {field_error}; repair后: {repaired_error}"
                    )
                    continue
        if gap is None:
            print(f"  跳过无法计算 gap 的轨迹 (attempt={attempts})")
            continue

        gap_min, gap_max = config.get("gap_range", (0.0, 1.0))
        if not (float(gap_min) <= gap < float(gap_max)):
            if attempts % 50 == 0:
                print(
                    f"  [{tier_name}] gap={gap*100:.2f}% 不在目标区间 "
                    f"[{float(gap_min)*100:.1f}%, {float(gap_max)*100:.1f}%)"
                )
            continue

        if hasattr(new_env, "current_instance") and new_env.current_instance is not None:
            instance_hash = hash(new_env.current_instance.tobytes())
            if instance_hash in seen_instances:
                continue
            seen_instances.add(instance_hash)

        # 添加层级信息
        trajectory['tier'] = tier_name
        trajectory['epsilon'] = epsilon
        trajectory['mutation'] = use_mutation
        trajectory['metrics']['tier'] = tier_name
        trajectory['metrics']['epsilon'] = float(epsilon)
        trajectory['metrics']['mutation'] = bool(use_mutation)
        trajectory['metrics']['quality_bin'] = tier_name
        trajectory['metrics']['processing_time_mutation_pct'] = float(getattr(new_env, "processing_time_mutation_pct", 0.0) or 0.0)
        if save_light_trajectories and light_to_save is not None:
            light_to_save['tier'] = tier_name
            light_to_save['epsilon'] = epsilon
            light_to_save['mutation'] = use_mutation
            light_to_save['total_return'] = trajectory.get('total_return', light_to_save.get('total_return', 0.0))
            light_to_save['metrics'] = dict(trajectory['metrics'])
        
        tier_collected[tier_name] += 1
        tier_efficiency_sum[tier_name] += efficiency
        total_collected += 1
        
        print(f"✅ [{tier_name}] 收集轨迹 {total_collected}/{num_trajectories} "
              f"(gap: {gap*100:.2f}%, 效率: {efficiency:.3f}, epsilon: {epsilon:.2f}, mutation: {use_mutation})")
        
        if output_path:
            try:
                to_save = trajectory
                if save_light_trajectories:
                    to_save = light_to_save if light_to_save is not None else make_light_jssp_trajectory(trajectory, env=new_env)
                serializable_traj = collector._convert_to_serializable(to_save)
                collector.append_trajectory_to_file(output_path, serializable_traj)
            except Exception as e:
                if save_light_trajectories:
                    raise
                print(f"  ❌ 保存失败: {e}")

        # In light-trajectory generation runs the JSONL file is the source of
        # truth. Keeping full per-step states in memory makes large 20x15/20x20
        # collection runs grow until the OS kills the process, even though the
        # saved training data is compact. Avoid retaining full trajectories when
        # we are streaming compact records to disk.
        if save_light_trajectories and output_path:
            continue

        trajectories.append(trajectory)
    
    # 打印收集结果
    print(f"\n{'='*60}")
    print(f"分层轨迹收集完成（按实际 gap 过滤）")
    print(f"{'='*60}")
    for tier_name, count in tier_collected.items():
        target = tier_targets[tier_name]
        status = "✅" if count >= target else "⚠️"
        avg_eff = tier_efficiency_sum[tier_name] / count if count > 0 else 0
        print(f"  {status} {tier_name}: {count}/{target} (平均效率: {avg_eff:.1%})")
    print(f"  总计: {total_collected}/{num_trajectories}")
    print(f"{'='*60}\n")
    
    return trajectories


def collect_ortools_trajectories(
    env,
    num_trajectories=100,
    max_steps=100,
    output_path=None,
    pure_expert_ratio=0.7,
    epsilon_min=0.1,
    epsilon_max=0.3,
    processing_time_mutation_ratio=0.5,
    processing_time_mutation_pct=0.1,
    use_sequence_swap=True,
    save_light_trajectories=False,
    validate_light_trajectories=False,
):
    """
    收集 OR-Tools 引导的分层轨迹。

    默认比例约为 70% expert、20% near_optimal、10% medium。每层会用实际
    relative_gap 做过滤，并通过不同实例、可选 processing time mutation 和
    epsilon 扰动增加多样性。
    
    Args:
        env: JSSP environment (used as a template)
        num_trajectories: Number of trajectories to collect
        max_steps: Maximum steps per trajectory
        output_path: Optional path to save trajectories
        pure_expert_ratio: expert 层目标比例
        epsilon_min/max: 非 expert 层的扰动范围
        processing_time_mutation_ratio: 处理时间变异比例
        processing_time_mutation_pct: 处理时间变异幅度
        use_sequence_swap: 保留兼容，不再单独使用

    Returns:
        List of collected trajectories
    """
    
    print("\n🔧 使用分层 OR-Tools 轨迹收集")
    print("   expert/near_optimal/medium 会按实际 gap 过滤")

    expert_ratio = min(0.9, max(0.1, float(pure_expert_ratio)))
    remaining = max(0.0, 1.0 - expert_ratio)
    near_ratio = min(0.20, remaining)
    medium_ratio = max(0.0, remaining - near_ratio)
    if medium_ratio <= 1e-8:
        near_ratio = remaining * 0.67
        medium_ratio = remaining - near_ratio

    eps_min = float(epsilon_min)
    eps_max = float(epsilon_max)
    eps_mid = eps_min + 0.45 * max(0.0, eps_max - eps_min)

    tier_distribution = {
        'expert': {
            'ratio': expert_ratio,
            'epsilon': 0.0,
            'gap_range': (0.0, 0.05),
            'mutation': False,
        },
        'near_optimal': {
            'ratio': near_ratio,
            'epsilon_min': max(0.0, eps_min * 0.5),
            'epsilon_max': max(eps_min, eps_mid),
            'gap_range': (0.05, 0.10),
            'mutation': True,
            'mutation_pct': processing_time_mutation_pct,
        },
        'medium': {
            'ratio': medium_ratio,
            'epsilon_min': max(eps_min, eps_mid),
            'epsilon_max': eps_max,
            'gap_range': (0.10, 0.25),
            'mutation': True,
            'mutation_pct': processing_time_mutation_pct,
        },
    }
    
    return collect_tiered_trajectories(
        env=env,
        num_trajectories=num_trajectories,
        max_steps=max_steps,
        output_path=output_path,
        tier_distribution=tier_distribution,
        processing_time_mutation_ratio=processing_time_mutation_ratio,
        processing_time_mutation_pct=processing_time_mutation_pct,
        save_light_trajectories=save_light_trajectories,
        validate_light_trajectories=validate_light_trajectories,
    )


def collect_random_trajectories(env, num_trajectories=100, max_steps=100, output_path=None):
    """
    Collect trajectories using random action selection.

    Args:
        env: JSSP environment
        num_trajectories: Number of trajectories to collect
        max_steps: Maximum steps per trajectory
        output_path: Optional path to save trajectories. If provided, trajectories will be saved incrementally.

    Returns:
        List of collected trajectories
    """
    collector = JSSPTrajectoryCollector(env)
    agent = RandomAgent(env)

    for i in range(num_trajectories):
        trajectory = collector.collect_trajectory(agent, max_steps=max_steps)
        print(f"Collected random trajectory {i+1}/{num_trajectories}")

        # Save incrementally if output path is provided
        if output_path:
            collector.save_trajectories(output_path, append=True)
            print(f"Saved trajectory {i+1} to {output_path}")

    return collector.trajectories


def repair_jssp_decision_trajectory(trajectory, rtg_mode="normalized_budget"):
    """Keep only true decision transitions and realign actions with state deltas.

    Older collected data can contain time-advance states in ``states`` while
    ``actions`` only stores real scheduling decisions. Pairing by index then
    shifts labels after the first wait event and often pads the tail with job 0.
    This repair derives the action from ``job_next_op`` changes:
        states[i] -> states[i + 1] where exactly one job advances by one op.
    """
    if not isinstance(trajectory, dict):
        return trajectory
    if is_light_jssp_trajectory(trajectory):
        trajectory = replay_light_jssp_trajectory(trajectory)
    states = trajectory.get("states", [])
    if not (isinstance(states, list) and len(states) >= 2 and isinstance(states[0], dict)):
        return trajectory

    repaired_states = []
    repaired_actions = []
    repaired_rewards = []

    raw_rewards = trajectory.get("rewards", [])
    reward_ptr = 0

    for i in range(len(states) - 1):
        cur = states[i]
        nxt = states[i + 1]
        cur_ops = cur.get("job_next_op", cur.get("job_status", None))
        nxt_ops = nxt.get("job_next_op", nxt.get("job_status", None))
        if cur_ops is None or nxt_ops is None:
            continue

        diffs = []
        for j, (a, b) in enumerate(zip(cur_ops, nxt_ops)):
            try:
                da = int(float(a))
                db = int(float(b))
            except Exception:
                da = db = 0
            if db > da:
                diffs.append((j, db - da))

        if len(diffs) != 1 or diffs[0][1] != 1:
            continue

        repaired_states.append(cur)
        repaired_actions.append(int(diffs[0][0]))
        if reward_ptr < len(raw_rewards):
            try:
                repaired_rewards.append(float(raw_rewards[reward_ptr]))
            except Exception:
                repaired_rewards.append(0.0)
        else:
            repaired_rewards.append(0.0)
        reward_ptr += 1

    if not repaired_actions:
        return trajectory

    repaired_states.append(states[-1])

    first_state = repaired_states[0]
    try:
        n_jobs = int(len(first_state.get("job_next_op", first_state.get("job_status", []))))
        n_machines = int(len(first_state.get("machine_sequence", [[]])[0]))
    except Exception:
        n_jobs = n_machines = 0
    if n_jobs > 0 and n_machines > 0 and "machine_sequence" in first_state:
        node_count = n_jobs * n_machines
        adj = np.zeros((node_count, node_count), dtype=float)
        last_on_machine = {}
        job_ready = np.zeros(n_jobs, dtype=float)
        machine_ready = np.zeros(n_machines, dtype=float)
        states_with_adj = []
        for t, state in enumerate(repaired_states):
            state_copy = dict(state)
            cur_t = float(state.get("current_time", 0.0) or 0.0)
            chosen_job = int(repaired_actions[t]) if t < len(repaired_actions) else None

            def _action_is_ready(job, at_time):
                if job is None or job < 0 or job >= n_jobs:
                    return False
                try:
                    op_j = int(float(state.get("job_next_op", state.get("job_status", []))[job]))
                    if op_j >= n_machines:
                        return False
                    m_j = int(state["machine_sequence"][job][op_j])
                except Exception:
                    return False
                return (
                    0 <= m_j < n_machines
                    and job_ready[job] <= at_time + 1e-8
                    and machine_ready[m_j] <= at_time + 1e-8
                )

            if chosen_job is not None and not _action_is_ready(chosen_job, cur_t) and t + 1 < len(repaired_states):
                next_state = repaired_states[t + 1]
                next_t = float(next_state.get("current_time", cur_t) or cur_t)
                if next_t > cur_t and _action_is_ready(chosen_job, next_t):
                    cur_t = next_t
                    state_copy["current_time"] = cur_t
                    if "overall_info" in state_copy:
                        overall = list(state_copy["overall_info"])
                        if len(overall) > 0:
                            try:
                                overall[0] = float(next_state.get("overall_info", overall)[0])
                            except Exception:
                                scale = float(state.get("max_expected_time", 0.0) or 0.0)
                                overall[0] = cur_t / max(1.0, scale)
                        state_copy["overall_info"] = overall
                    if "machine_busy_mask" in state_copy and "normalized_machine_availability" in state_copy:
                        try:
                            cur_norm = float(state_copy.get("overall_info", [0.0])[0])
                            state_copy["machine_busy_mask"] = [
                                1.0 if float(v) > cur_norm + 1e-8 else 0.0
                                for v in state_copy["normalized_machine_availability"]
                            ]
                        except Exception:
                            pass

            state_copy["scheduled_machine_adj"] = adj.copy().tolist()
            legal_mask = np.zeros(n_jobs, dtype=float)
            job_ops = state.get("job_next_op", state.get("job_status", []))
            for j in range(n_jobs):
                try:
                    op_j = int(float(job_ops[j]))
                    if op_j >= n_machines:
                        continue
                    m_j = int(state["machine_sequence"][j][op_j])
                except Exception:
                    continue
                if (
                    0 <= m_j < n_machines
                    and job_ready[j] <= cur_t + 1e-8
                    and machine_ready[m_j] <= cur_t + 1e-8
                ):
                    legal_mask[j] = 1.0
            if chosen_job is not None and 0 <= chosen_job < n_jobs:
                legal_mask[chosen_job] = 1.0
            state_copy["legal_action_mask"] = legal_mask.tolist()
            states_with_adj.append(state_copy)
            if t >= len(repaired_actions):
                continue
            job = int(repaired_actions[t])
            try:
                op = int(float(state.get("job_next_op", state.get("job_status"))[job]))
                machine = int(state["machine_sequence"][job][op])
            except Exception:
                continue
            node = job * n_machines + op
            prev = last_on_machine.get(machine)
            if prev is not None and 0 <= prev < node_count and 0 <= node < node_count:
                adj[prev, node] = 1.0
            last_on_machine[machine] = node
            try:
                if "processing_times" in state:
                    duration = float(state["processing_times"][job][op])
                else:
                    duration = float(state.get("remaining_processing_times", [[0.0]])[job][op])
            except Exception:
                duration = 0.0
            finish = max(cur_t, job_ready[job], machine_ready[machine]) + max(0.0, duration)
            job_ready[job] = finish
            machine_ready[machine] = finish
        repaired_states = states_with_adj

    metrics = trajectory.get("metrics", {}) if isinstance(trajectory.get("metrics", {}), dict) else {}
    rtg_mode = str(rtg_mode or "original").lower().strip()
    if rtg_mode == "normalized_budget":
        budget = metrics.get("actual_makespan", None) or metrics.get("makespan", None) or metrics.get("optimal_makespan", None)
        try:
            budget = float(budget)
        except Exception:
            budget = 0.0
        if budget <= 0:
            budget = max(float(s.get("current_time", 0.0) or 0.0) for s in repaired_states)
        budget = max(1.0, float(budget))
        returns_to_go = []
        for s in repaired_states:
            cur_t = float(s.get("current_time", 0.0) or 0.0)
            returns_to_go.append(-max(0.0, budget - cur_t) / budget)
    else:
        returns_to_go = []
        future_return = 0.0
        for r in reversed(repaired_rewards):
            future_return += float(r)
            returns_to_go.insert(0, future_return)
        returns_to_go.append(0.0)

    repaired = dict(trajectory)
    repaired["states"] = repaired_states
    repaired["actions"] = repaired_actions
    repaired["rewards"] = repaired_rewards
    repaired["returns_to_go"] = returns_to_go
    repaired["timesteps"] = list(range(len(repaired_states)))
    repaired["trajectory_length"] = len(repaired_actions)
    if metrics:
        repaired["metrics"] = dict(metrics)
        repaired["metrics"]["n_decision_steps_repaired"] = len(repaired_actions)
        repaired["metrics"]["rtg_mode"] = rtg_mode
    return repaired


class JSSPTrajectoryConverter:
    """Converter for JSSP trajectories to Decision Transformer format."""

    def __init__(self, state_dim, act_dim):
        """
        Initialize the trajectory converter.

        Args:
            state_dim: Dimension of the state embeddings
            act_dim: Dimension of the action embeddings
        """
        self.state_dim = state_dim
        self.act_dim = act_dim

    def convert_trajectory(self, trajectory, max_length=128):
        """
        Convert a single trajectory to Decision Transformer format.

        Args:
            trajectory: Trajectory dictionary
            max_length: Maximum sequence length (used to truncate if needed)

        Returns:
            Dictionary with tensors for Decision Transformer training
        """
        # Get trajectory components
        states = trajectory['states']
        actions = np.array(trajectory['actions'])
        expert_actions = None
        if 'expert_actions' in trajectory and trajectory['expert_actions'] is not None:
            expert_actions = np.array(trajectory['expert_actions'])
        returns_to_go = np.array(trajectory['returns_to_go'])
        timesteps = np.array(trajectory['timesteps'])

        # Add NaN value checking and handling
        if not (isinstance(states, list) and states and isinstance(states[0], dict)):
            states = np.array(states)
            if np.isnan(states).any():
                print(f"Warning: Trajectory contains NaN values in states. Replacing with 0.")
                states = np.nan_to_num(states, nan=0.0)

        # Check and handle NaNs in actions
        if np.isnan(actions).any():
            print(f"Warning: Trajectory contains NaN values in actions. Replacing with 0.")
            actions = np.nan_to_num(actions, nan=0.0)
        if expert_actions is not None and np.isnan(expert_actions).any():
            print(f"Warning: Trajectory contains NaN values in expert_actions. Replacing with 0.")
            expert_actions = np.nan_to_num(expert_actions, nan=0.0)

        # Check and handle NaNs in returns_to_go
        if np.isnan(returns_to_go).any():
            print(f"Warning: Trajectory contains NaN values in returns_to_go. Replacing with 0.")
            returns_to_go = np.nan_to_num(returns_to_go, nan=0.0)

        # Check and handle NaNs in timesteps
        if np.isnan(timesteps).any():
            print(f"Warning: Trajectory contains NaN values in timesteps. Replacing with 0.")
            timesteps = np.nan_to_num(timesteps, nan=0.0)

        # Keep state/action rows at the same decision index. The training loop
        # builds previous-action inputs as [dummy, A0, A1, ...] and supervises
        # current actions [A0, A1, A2, ...], matching rollout.

        # Original sequence lengths
        T = len(states)
        A = len(actions)
        EA = len(expert_actions) if expert_actions is not None else A
        R = len(returns_to_go)

        # For DT, we need at least 1 time step to have valid alignment
        if T < 1:
            # Handle empty trajectory case
            return {
                'states': torch.empty((0,), dtype=torch.float32),
                'actions': torch.empty((0,), dtype=torch.long),
                'returns_to_go': torch.empty((0, 1), dtype=torch.float32),
                'timesteps': torch.empty((0,), dtype=torch.long),
                'mask': torch.empty((0,), dtype=torch.float32),
                'trajectory_length': 0
            }

        # Keep the full decision trajectory here. ``max_length`` is applied in
        # the Dataset via a random contiguous crop, which is the DT context
        # window behavior we want during training.
        dt_seq_length = min(T, A, R)
        if expert_actions is not None:
            dt_seq_length = min(dt_seq_length, EA)

        if dt_seq_length < 1:
            # Not enough data for DT
            dt_seq_length = 0

        # Create tensors with proper DT alignment
        if dt_seq_length > 0:
            # States: S_0 to S_{dt_seq_length-1} (original)
            # These will be used as input states [S0, S1, ..., S_{dt_seq_length-1}]
            if isinstance(states, list) and states and isinstance(states[0], dict):
                states_tensor = states[:dt_seq_length]
            else:
                states_tensor = torch.tensor(states[:dt_seq_length], dtype=torch.float32)

            # Actions: A_0 to A_{dt_seq_length-1} (original actions)
            actions_tensor = torch.tensor(actions[:dt_seq_length], dtype=torch.long)
            expert_actions_tensor = None
            if expert_actions is not None:
                expert_actions_tensor = torch.tensor(expert_actions[:dt_seq_length], dtype=torch.long)

            # Returns-to-go: G_0 to G_{dt_seq_length-1} (original)
            # These match the states
            # 确保returns_to_go是数值类型
            returns_slice = returns_to_go[:dt_seq_length]

            # 检查并转换returns_to_go的类型
            if isinstance(returns_slice, np.ndarray):
                # 如果是numpy数组，确保是数值类型
                if returns_slice.dtype.kind in {'U', 'S'}:  # 字符串类型
                    returns_slice = returns_slice.astype(np.float32)
            elif isinstance(returns_slice, list):
                # 如果是列表，检查每个元素的类型
                for i, val in enumerate(returns_slice):
                    if isinstance(val, str):
                        # 字符串类型转换为float
                        returns_slice[i] = float(val)
                    elif isinstance(val, (int, float)):
                        # 数值类型保持不变
                        continue
                    elif hasattr(val, '__float__'):
                        # 其他可转换为float的类型
                        returns_slice[i] = float(val)
                # 转换为numpy数组
                returns_slice = np.array(returns_slice, dtype=np.float32)

            # 确保是正确的形状
            if len(returns_slice.shape) == 1:
                returns_slice = returns_slice[:, None]  # 添加第二个维度

            returns_tensor = torch.tensor(returns_slice, dtype=torch.float32)

            # Timesteps: t_0 to t_{dt_seq_length-1} (original)
            timesteps_tensor = torch.tensor(timesteps[:dt_seq_length], dtype=torch.long)

            # Mask for valid steps
            mask_tensor = torch.ones((dt_seq_length,), dtype=torch.float32)

            # Ensure proper alignment
            assert len(states_tensor) == dt_seq_length
            assert len(actions_tensor) == dt_seq_length
            if expert_actions_tensor is not None:
                assert len(expert_actions_tensor) == dt_seq_length
            assert len(returns_tensor) == dt_seq_length
            assert len(timesteps_tensor) == dt_seq_length
        else:
            # Edge case with no valid DT steps
            states_tensor = torch.empty((0,), dtype=torch.float32)
            actions_tensor = torch.empty((0,), dtype=torch.long)
            expert_actions_tensor = None
            returns_tensor = torch.empty((0, 1), dtype=torch.float32)
            timesteps_tensor = torch.empty((0,), dtype=torch.long)
            mask_tensor = torch.empty((0,), dtype=torch.float32)

        out = {
            'states': states_tensor,
            'actions': actions_tensor,
            'returns_to_go': returns_tensor,
            'timesteps': timesteps_tensor,
            'mask': mask_tensor,
            'trajectory_length': dt_seq_length
        }
        if 'metrics' in trajectory and isinstance(trajectory['metrics'], dict):
            out['metrics'] = dict(trajectory['metrics'])
        if expert_actions_tensor is not None:
            out['expert_actions'] = expert_actions_tensor
        return out

    def convert_multiple_trajectories(self, trajectories, max_length=128):
        """
        Convert multiple trajectories to Decision Transformer format.

        Args:
            trajectories: List of trajectory dictionaries
            max_length: Maximum sequence length

        Returns:
            List of converted trajectories
        """
        converted_trajectories = []
        for traj in trajectories:
            converted = self.convert_trajectory(traj, max_length)
            converted_trajectories.append(converted)
        return converted_trajectories


class JSSPDecisionTransformerDataset(Dataset):
    """Dataset for Decision Transformer training on JSSP data."""

    def __init__(
        self,
        trajectories,
        max_length=128,
        p_sample=1.0,
        max_jobs=None,
        max_machines=None,
        enable_augmentation=False,
        augmentation_ratio=0.0,
        state_noise_std=0.0,
        rtg_scale=1.0,
        use_gap_weighted_loss=False,
        gap_weight_alpha=6.0,
        gap_weight_power=2.0,
        quality_weight_mix=0.25,
        min_sample_weight=0.35,
        max_sample_weight=3.0,
        state_tensorization_mode=LEGACY_TRAIN_COMPATIBLE,
    ):
        """
        Initialize the dataset.

        Args:
            trajectories: List of trajectories
            max_length: Maximum sequence length
            p_sample: Probability to sample a trajectory (for balancing)
        """
        self.trajectories = trajectories
        self.max_length = max_length
        self.p_sample = p_sample
        self.max_jobs = max_jobs
        self.max_machines = max_machines
        self.enable_augmentation = bool(enable_augmentation)
        self.augmentation_ratio = float(augmentation_ratio)
        self.state_noise_std = float(state_noise_std)
        self.rtg_scale = float(rtg_scale)
        self.use_gap_weighted_loss = bool(use_gap_weighted_loss)
        self.gap_weight_alpha = float(gap_weight_alpha)
        self.gap_weight_power = float(gap_weight_power)
        self.quality_weight_mix = float(quality_weight_mix)
        self.min_sample_weight = float(min_sample_weight)
        self.max_sample_weight = float(max_sample_weight)
        self.state_tensorization_mode = validate_state_tensorization_mode(
            state_tensorization_mode
        )

        if self.max_jobs is None or self.max_machines is None:
            inferred_jobs = 0
            inferred_machines = 0
            for traj in self.trajectories:
                if not isinstance(traj, dict):
                    continue
                states = traj.get("states", None)
                if not (isinstance(states, list) and states and isinstance(states[0], dict)):
                    continue
                s0 = states[0]
                try:
                    if "job_next_op" in s0:
                        inferred_jobs = max(inferred_jobs, int(len(s0["job_next_op"])))
                    elif "job_status" in s0:
                        inferred_jobs = max(inferred_jobs, int(len(s0["job_status"])))
                except Exception:
                    pass
                try:
                    if "normalized_machine_availability" in s0:
                        inferred_machines = max(inferred_machines, int(len(s0["normalized_machine_availability"])))
                    elif "machine_status" in s0:
                        inferred_machines = max(inferred_machines, int(len(s0["machine_status"])))
                    elif "job_completion_mask" in s0 and len(s0["job_completion_mask"]) > 0:
                        inferred_machines = max(inferred_machines, int(len(s0["job_completion_mask"][0])))
                    elif "remaining_processing_times" in s0 and len(s0["remaining_processing_times"]) > 0:
                        inferred_machines = max(inferred_machines, int(len(s0["remaining_processing_times"][0])))
                except Exception:
                    pass
            if self.max_jobs is None:
                self.max_jobs = inferred_jobs if inferred_jobs > 0 else 10
            if self.max_machines is None:
                self.max_machines = inferred_machines if inferred_machines > 0 else 10

        self.sample_weights = self._build_sample_weights()

    def _build_sample_weights(self):
        weights = []
        mix = min(1.0, max(0.0, self.quality_weight_mix))
        for traj in self.trajectories:
            metrics = traj.get("metrics", {}) if isinstance(traj, dict) else {}
            relative_gap = metrics.get("relative_gap", None)
            quality_score = metrics.get("quality_score", metrics.get("trajectory_quality_score", None))

            weight = 1.0
            if self.use_gap_weighted_loss:
                gap_weight = 1.0
                if relative_gap is not None:
                    try:
                        gap_val = max(0.0, float(relative_gap))
                        gap_weight = 1.0 / ((1.0 + self.gap_weight_alpha * gap_val) ** self.gap_weight_power)
                    except Exception:
                        gap_weight = 1.0

                quality_weight = 1.0
                if quality_score is not None:
                    try:
                        quality_weight = max(0.2, float(quality_score))
                    except Exception:
                        quality_weight = 1.0

                weight = (1.0 - mix) * gap_weight + mix * quality_weight
                weight = min(self.max_sample_weight, max(self.min_sample_weight, weight))

            weights.append(float(weight))

        if not weights:
            return []

        mean_w = float(np.mean(weights))
        if mean_w <= 0:
            return [1.0 for _ in weights]
        return [float(w / mean_w) for w in weights]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the trajectory

        Returns:
            Dictionary with tensors for training:
                - states: State tensors (padded with 0s)
                - actions: Action tensors (padded with -100 to avoid confusion with job 0)
                - returns_to_go: Returns-to-go tensors (padded with -1e8 to avoid misleading model)
                - timesteps: Timestep tensors (padded with 0s)
                - mask: Mask tensor (1 for valid steps, 0 for padded steps)

        Note:
            Downstream models must use the mask to ignore padded positions when computing loss.
            Padded positions:
            - actions: -100 (to distinguish from actual job 0)
            - returns_to_go: -1e8 (to avoid suggesting zero future returns)
        """
        traj = self.trajectories[idx]
        traj_length = traj['trajectory_length']

        # Randomly choose a subsequence of length up to max_length
        if traj_length > self.max_length:
            start_idx = np.random.randint(0, traj_length - self.max_length + 1)
            end_idx = start_idx + self.max_length
        else:
            start_idx = 0
            end_idx = traj_length

        # Extract subsequence
        states = traj['states'][start_idx:end_idx]
        actions = traj['actions'][start_idx:end_idx]
        expert_actions = traj.get('expert_actions', None)
        if expert_actions is not None:
            expert_actions = expert_actions[start_idx:end_idx]
        returns_to_go = traj['returns_to_go'][start_idx:end_idx]
        timesteps = traj['timesteps'][start_idx:end_idx]
        mask = traj['mask'][start_idx:end_idx]

        # ── RTG 归一化：按单条轨迹首 RTG 的绝对值归一化 ────────────────
        # 恢复旧版稳定口径：让模型学习“相对剩余回报”，训练分布通常落在 [-1, 0]。
        if torch.is_tensor(returns_to_go) and returns_to_go.dtype.is_floating_point:
            full_rtg = traj['returns_to_go']
            rtg0 = full_rtg[0] if torch.is_tensor(full_rtg) else torch.tensor(full_rtg[0], dtype=torch.float32)
            rtg0_abs = float(rtg0.abs().item()) if torch.is_tensor(rtg0) else abs(float(rtg0))
            if rtg0_abs > 1.0:
                returns_to_go = returns_to_go / rtg0_abs

        do_aug = self.enable_augmentation and self.augmentation_ratio > 0.0 and np.random.random() < self.augmentation_ratio
        if do_aug:
            if self.rtg_scale is not None and self.rtg_scale != 1.0:
                if torch.is_tensor(returns_to_go) and returns_to_go.dtype.is_floating_point:
                    returns_to_go = returns_to_go.clone() * self.rtg_scale

            if self.state_noise_std > 0.0 and not (isinstance(states, list) and states and isinstance(states[0], dict)):
                if torch.is_tensor(states) and states.dtype.is_floating_point:
                    states = states.clone()
                    states = states + torch.randn_like(states) * self.state_noise_std

        pad_length = self.max_length - len(actions)
        if isinstance(states, list) and states and isinstance(states[0], dict):
            allowed_keys = {
                'job_next_op',
                'job_completion_mask',
                'remaining_processing_times',
                'normalized_machine_availability',
                'machine_busy_mask',
                'overall_info',
                'machine_sequence',
                'scheduled_machine_adj',
                'legal_action_mask',
            }
            keys = [k for k in states[0].keys() if k in allowed_keys]
            states_dict = {}
            noise_keys = {"remaining_processing_times", "normalized_machine_availability", "overall_info"}
            first_state = states[0]
            try:
                actual_n_jobs = int(len(first_state.get("job_next_op", first_state.get("job_status", []))))
            except Exception:
                actual_n_jobs = 0
            try:
                if "normalized_machine_availability" in first_state:
                    actual_n_machines = int(len(first_state["normalized_machine_availability"]))
                elif "machine_status" in first_state:
                    actual_n_machines = int(len(first_state["machine_status"]))
                elif "job_completion_mask" in first_state and len(first_state["job_completion_mask"]) > 0:
                    actual_n_machines = int(len(first_state["job_completion_mask"][0]))
                elif "remaining_processing_times" in first_state and len(first_state["remaining_processing_times"]) > 0:
                    actual_n_machines = int(len(first_state["remaining_processing_times"][0]))
                else:
                    actual_n_machines = 0
            except Exception:
                actual_n_machines = 0
            if actual_n_jobs <= 0:
                actual_n_jobs = int(self.max_jobs or 0)
            if actual_n_machines <= 0:
                actual_n_machines = int(self.max_machines or 0)
            max_jobs = int(self.max_jobs or actual_n_jobs or 0)
            max_machines = int(self.max_machines or actual_n_machines or 0)

            tensorized_states = [
                tensorize_graph_state(
                    state,
                    actual_n_jobs=actual_n_jobs,
                    actual_n_machines=actual_n_machines,
                    max_jobs=max_jobs,
                    max_machines=max_machines,
                    keys=keys,
                    mode=self.state_tensorization_mode,
                )
                for state in states
            ]
            for key in keys:
                comp = torch.stack(
                    [tensorized[key] for tensorized in tensorized_states], dim=0
                )
                if do_aug and self.state_noise_std > 0.0 and key in noise_keys:
                    comp = comp + torch.randn_like(comp) * self.state_noise_std
                if pad_length > 0:
                    pad = torch.zeros((pad_length,) + tuple(comp.shape[1:]), dtype=comp.dtype)
                    comp = torch.cat([comp, pad], dim=0)
                states_dict[key] = comp
            pn_jobs = torch.full((len(states), 1), float(actual_n_jobs), dtype=torch.float32)
            pn_machines = torch.full((len(states), 1), float(actual_n_machines), dtype=torch.float32)
            if pad_length > 0:
                pn_jobs = torch.cat([pn_jobs, torch.zeros((pad_length, 1), dtype=torch.float32)], dim=0)
                pn_machines = torch.cat([pn_machines, torch.zeros((pad_length, 1), dtype=torch.float32)], dim=0)
            states_dict["problem_n_jobs"] = pn_jobs
            states_dict["problem_n_machines"] = pn_machines
            states = states_dict
        else:
            if pad_length > 0:
                states = torch.cat([states, torch.zeros((pad_length, states.shape[-1]))], dim=0)

        if pad_length > 0:
            actions = torch.cat([actions, torch.full((pad_length,), -100, dtype=torch.long)], dim=0)
            if expert_actions is not None:
                expert_actions = torch.cat([expert_actions, torch.full((pad_length,), -100, dtype=torch.long)], dim=0)
            returns_to_go = torch.cat([returns_to_go, torch.full((pad_length, 1), -1e4, dtype=returns_to_go.dtype)], dim=0)
            timesteps = torch.cat([timesteps, torch.zeros((pad_length,), dtype=torch.long)], dim=0)
            mask = torch.cat([mask, torch.zeros((pad_length,))], dim=0)

        out = {
            'states': states,
            'actions': actions,
            'returns_to_go': returns_to_go,
            'timesteps': timesteps,
            'mask': mask,
            'sample_weight': torch.tensor(float(self.sample_weights[idx]) if self.sample_weights else 1.0, dtype=torch.float32),
        }
        if expert_actions is not None:
            out['expert_actions'] = expert_actions
        return out


def build_jsonl_offsets(filepath):
    """Build byte offsets for a jsonl trajectory file without parsing it."""
    offsets = []
    with open(filepath, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    return offsets


class LazyJSSPDecisionTransformerDataset(Dataset):
    """Memory-safe jsonl dataset that parses one trajectory per sample."""

    def __init__(
        self,
        filepath,
        offsets,
        state_dim=128,
        act_dim=64,
        max_length=128,
        max_jobs=None,
        max_machines=None,
        enable_augmentation=False,
        augmentation_ratio=0.0,
        state_noise_std=0.0,
        rtg_scale=1.0,
        use_gap_weighted_loss=False,
        gap_weight_alpha=6.0,
        gap_weight_power=2.0,
        quality_weight_mix=0.25,
        min_sample_weight=0.35,
        max_sample_weight=3.0,
        rtg_mode="normalized_budget",
        state_tensorization_mode=LEGACY_TRAIN_COMPATIBLE,
    ):
        self.filepath = str(filepath)
        self.offsets = list(offsets)
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.max_jobs = max_jobs
        self.max_machines = max_machines
        self.enable_augmentation = bool(enable_augmentation)
        self.augmentation_ratio = float(augmentation_ratio)
        self.state_noise_std = float(state_noise_std)
        self.rtg_scale = float(rtg_scale)
        self.use_gap_weighted_loss = bool(use_gap_weighted_loss)
        self.gap_weight_alpha = float(gap_weight_alpha)
        self.gap_weight_power = float(gap_weight_power)
        self.quality_weight_mix = float(quality_weight_mix)
        self.min_sample_weight = float(min_sample_weight)
        self.max_sample_weight = float(max_sample_weight)
        self.rtg_mode = rtg_mode
        self.state_tensorization_mode = validate_state_tensorization_mode(
            state_tensorization_mode
        )
        self.converter = JSSPTrajectoryConverter(state_dim, act_dim)
        self._fh = None
        self.sample_weights = []

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fh"] = None
        return state

    def _file(self):
        if self._fh is None or self._fh.closed:
            self._fh = open(self.filepath, "r", encoding="utf-8")
        return self._fh

    def _sample_weight_from_metrics(self, metrics):
        if not self.use_gap_weighted_loss:
            return 1.0
        mix = min(1.0, max(0.0, self.quality_weight_mix))
        gap_weight = 1.0
        relative_gap = metrics.get("relative_gap", None) if isinstance(metrics, dict) else None
        if relative_gap is not None:
            try:
                gap_val = max(0.0, float(relative_gap))
                gap_weight = 1.0 / ((1.0 + self.gap_weight_alpha * gap_val) ** self.gap_weight_power)
            except Exception:
                gap_weight = 1.0

        quality_weight = 1.0
        quality_score = metrics.get("quality_score", metrics.get("trajectory_quality_score", None)) if isinstance(metrics, dict) else None
        if quality_score is not None:
            try:
                quality_weight = max(0.2, float(quality_score))
            except Exception:
                quality_weight = 1.0

        weight = (1.0 - mix) * gap_weight + mix * quality_weight
        return float(min(self.max_sample_weight, max(self.min_sample_weight, weight)))

    def _load_trajectory(self, idx):
        fh = self._file()
        fh.seek(self.offsets[idx])
        return json.loads(fh.readline())

    def __getitem__(self, idx):
        raw = self._load_trajectory(idx)
        raw = repair_jssp_decision_trajectory(raw, rtg_mode=self.rtg_mode)
        processed = {
            "states": raw["states"],
            "actions": raw["actions"],
            "rewards": raw["rewards"],
            "returns_to_go": raw.get("returns_to_go", []),
            "timesteps": raw["timesteps"],
        }
        if "metrics" in raw and isinstance(raw["metrics"], dict):
            processed["metrics"] = dict(raw["metrics"])
        if "expert_actions" in raw:
            processed["expert_actions"] = raw["expert_actions"]

        converted = self.converter.convert_trajectory(processed, max_length=self.max_length)
        tmp = JSSPDecisionTransformerDataset(
            [converted],
            max_length=self.max_length,
            max_jobs=self.max_jobs,
            max_machines=self.max_machines,
            enable_augmentation=self.enable_augmentation,
            augmentation_ratio=self.augmentation_ratio,
            state_noise_std=self.state_noise_std,
            rtg_scale=self.rtg_scale,
            use_gap_weighted_loss=False,
            state_tensorization_mode=self.state_tensorization_mode,
        )
        item = tmp[0]
        item["sample_weight"] = torch.tensor(
            self._sample_weight_from_metrics(processed.get("metrics", {})),
            dtype=torch.float32,
        )
        return item


def collate_fn(batch):
    """
    Collate function to handle trajectories of varying lengths.

    Args:
        batch: List of trajectory dictionaries from JSSPDecisionTransformerDataset

    Returns:
        Dictionary with padded tensors for states, actions, returns_to_go, timesteps, and mask.
        The mask is used to ignore padded positions during loss calculation.
    """
    max_len = max(item['actions'].shape[0] for item in batch)

    # Initialize lists to hold padded tensors
    states_padded = []
    actions_padded = []
    expert_actions_padded = []
    rtg_padded = []
    timesteps_padded = []
    masks = []
    sample_weights = []
    has_expert_actions = any(isinstance(item, dict) and 'expert_actions' in item for item in batch)

    # Process each item in the batch
    for item in batch:
        L = item['actions'].shape[0]

        # Padding length needed
        pad_len = max_len - L

        batch_mask = item.get('mask', None)
        if batch_mask is None:
            batch_mask = torch.ones((L,), dtype=torch.float32)

        # Extract trajectory components
        states = item['states']
        actions = item['actions']
        expert_actions = item.get('expert_actions', None) if has_expert_actions else None
        returns_to_go = item['returns_to_go']
        timesteps = item['timesteps']
        sample_weight = item.get('sample_weight', torch.tensor(1.0, dtype=torch.float32))

        if isinstance(states, dict):
            if pad_len > 0:
                for key, comp in list(states.items()):
                    if comp.shape[0] < max_len:
                        pad = torch.zeros((pad_len,) + tuple(comp.shape[1:]), dtype=comp.dtype)
                        states[key] = torch.cat([comp, pad], dim=0)
            states_padded.append(states)
        else:
            if pad_len > 0:
                states = F.pad(states, (0, 0, 0, pad_len), value=0.0)  # (L, D) -> (max_len, D)
            states_padded.append(states)

        # Pad actions (assuming actions has shape [L])
        # Use -100 as padding value to distinguish from actual job 0
        if pad_len > 0:
            actions = F.pad(actions, (0, pad_len), value=-100)  # (L) -> (max_len)
        actions_padded.append(actions)

        if has_expert_actions:
            if expert_actions is None:
                expert_actions = torch.full((L,), -100, dtype=torch.long)
            if pad_len > 0:
                expert_actions = F.pad(expert_actions, (0, pad_len), value=-100)
            expert_actions_padded.append(expert_actions)

        # Pad returns_to_go (assuming returns_to_go has shape [L, 1])
        # Use -1e4 as padding value to avoid misleading the model
        if pad_len > 0:
            returns_to_go = F.pad(returns_to_go, (0, 0, 0, pad_len), value=-1e4)  # (L, 1) -> (max_len, 1)
        rtg_padded.append(returns_to_go)

        # Pad timesteps (assuming timesteps has shape [L])
        # Pad with zeros
        if pad_len > 0:
            timesteps = F.pad(timesteps, (0, pad_len), value=0)  # (L) -> (max_len)
        timesteps_padded.append(timesteps)

        if pad_len > 0:
            batch_mask = F.pad(batch_mask, (0, pad_len), value=0.0)
        masks.append(batch_mask)
        sample_weights.append(sample_weight.to(dtype=torch.float32).view(()))

    if isinstance(states_padded[0], dict):
        keys = list(states_padded[0].keys())
        batch_states = {}
        for k in keys:
            comps = [s[k] for s in states_padded]
            max_feat = max(int(c.shape[-1]) for c in comps)
            padded = []
            for c in comps:
                if int(c.shape[-1]) < max_feat:
                    c = F.pad(c, (0, max_feat - int(c.shape[-1])), value=0.0)
                elif int(c.shape[-1]) > max_feat:
                    c = c[..., :max_feat]
                padded.append(c)
            batch_states[k] = torch.stack(padded, dim=0)
    else:
        batch_states = torch.stack(states_padded)

    out = {
        'states': batch_states,
        'actions': torch.stack(actions_padded),
        'returns_to_go': torch.stack(rtg_padded),
        'timesteps': torch.stack(timesteps_padded),
        'mask': torch.stack(masks),
        'sample_weight': torch.stack(sample_weights),
    }
    if has_expert_actions:
        out['expert_actions'] = torch.stack(expert_actions_padded)
    return out


def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a data loader for the Decision Transformer dataset.

    Args:
        dataset: JSSPDecisionTransformerDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )


def create_jssp_state_encoder(state_dim=128, global_proc_time_max=None):
    """
    Create an operation-based state encoder for JSSP states that captures rich information about each operation's status.

    Args:
        state_dim: Output dimension of the encoder
        global_proc_time_max: Global maximum processing time across all instances (for consistent normalization)

    Returns:
        Callable encoder function
    """
    def encode_state(state):
        """Encode a JSSP state into a fixed-size vector using operation-based features."""
        # Check if state is in the new enhanced format (dictionary with specific keys)
        if isinstance(state, dict):
            # Get job and machine counts from state or infer from existing data
            if 'job_next_op' in state:
                n_jobs = len(state['job_next_op'])
            elif 'job_completion_mask' in state:
                n_jobs = len(state['job_completion_mask'])
            else:
                n_jobs = 5  # Default if not specified

            if 'job_completion_mask' in state:
                n_machines = len(state['job_completion_mask'][0])
            elif 'remaining_processing_times' in state:
                n_machines = len(state['remaining_processing_times'][0])
            else:
                n_machines = 5  # Default if not specified

            N = n_jobs * n_machines  # Total number of operations

            # Extract required information from state
            job_next_op = state.get('job_next_op', np.zeros(n_jobs))
            # 确保job_next_op是数值类型数组
            try:
                job_next_op = np.array(job_next_op, dtype=np.float64)
                # 如果job_next_op包含字符串，尝试转换
                if job_next_op.dtype.kind == 'U':  # 字符串类型
                    job_next_op = np.array([float(x) for x in job_next_op], dtype=np.float64)
            except (ValueError, TypeError):
                job_next_op = np.zeros(n_jobs, dtype=np.float64)

            # Ensure processing_times is correctly shaped and numeric
            processing_times = state.get('processing_times', np.zeros((n_jobs, n_machines)))
            try:
                # 尝试将处理时间转换为数值数组
                processing_times = np.array(processing_times, dtype=np.float64)
                # 确保形状正确
                if processing_times.shape != (n_jobs, n_machines):
                    processing_times = np.zeros((n_jobs, n_machines), dtype=np.float64)
            except (ValueError, TypeError):
                # 如果转换失败，使用默认值
                processing_times = np.zeros((n_jobs, n_machines), dtype=np.float64)

            # Ensure machine_sequence is correctly shaped and numeric
            machine_sequence = state.get('machine_sequence', np.zeros((n_jobs, n_machines), dtype=int))
            try:
                # 尝试将机器序列转换为整数数组
                machine_sequence = np.array(machine_sequence, dtype=np.int64)
                # 确保形状正确
                if machine_sequence.shape != (n_jobs, n_machines):
                    machine_sequence = np.zeros((n_jobs, n_machines), dtype=np.int64)
            except (ValueError, TypeError):
                # 如果转换失败，使用默认值
                machine_sequence = np.zeros((n_jobs, n_machines), dtype=np.int64)

            current_time = state.get('current_time', 0)
            # 确保current_time是数值类型
            current_time = float(current_time) if isinstance(current_time, (int, float)) else 0.0

            # Ensure machine_status is numeric and correctly shaped
            machine_status = state.get('machine_status', np.zeros(n_machines))
            try:
                # 尝试将machine_status转换为数值数组
                machine_status = np.array(machine_status, dtype=np.float64)
                # 确保形状正确
                if machine_status.shape != (n_machines,):
                    machine_status = np.zeros(n_machines, dtype=np.float64)
            except (ValueError, TypeError):
                # 如果转换失败，使用默认值
                machine_status = np.zeros(n_machines, dtype=np.float64)

            completed_operations = state.get('completed_operations', 0)
            # 确保completed_operations是数值类型
            try:
                completed_operations = float(completed_operations)
            except (ValueError, TypeError):
                completed_operations = 0.0

            # 1. 初始化默认值（关键！确保所有路径都有定义）
            makespan_estimate = current_time  # 使用current_time作为默认值

            # 2. 检查是否有overall_info字段（可能包含字符串数值）
            if 'overall_info' in state:
                overall_info = state['overall_info']
                # 确保overall_info中的数值都是浮点数类型
                try:
                    # 处理overall_info中的各个字段
                    if isinstance(overall_info, (list, np.ndarray)) and len(overall_info) >= 3:
                        # overall_info[2]通常是makespan_estimate
                        raw_val = overall_info[2]
                        if isinstance(raw_val, str):
                            makespan_estimate = float(raw_val)
                        elif isinstance(raw_val, (int, float)):
                            makespan_estimate = raw_val
                except (ValueError, TypeError) as e:
                    pass
            else:
                # 从常规字段获取makespan_estimate
                raw_val = state.get('makespan_estimate', current_time)
                if isinstance(raw_val, str):
                    try:
                        makespan_estimate = float(raw_val)
                    except (ValueError, TypeError):
                        pass  # 保持默认值
                elif isinstance(raw_val, (int, float)):
                    makespan_estimate = raw_val

            # 3. 最后确保makespan_estimate是数值类型
            try:
                makespan_estimate = float(makespan_estimate)
            except (ValueError, TypeError):
                makespan_estimate = current_time

            # Calculate total processing time as a robust upper bound
            # 防御性检查：确保processing_times是数值类型
            try:
                # 确保所有处理时间都是数值类型
                processing_times_numeric = np.asarray(processing_times, dtype=np.float64)
                total_proc_time = np.sum(processing_times_numeric)
                # 确保total_proc_time和makespan_estimate都是数值类型
                total_proc_time = float(total_proc_time)
                makespan_estimate = float(makespan_estimate)
                max_expected_time = max(makespan_estimate, total_proc_time)
            except (ValueError, TypeError, np.core._exceptions._UFuncNoLoopError):
                # 如果计算失败，使用默认值
                max_expected_time = float(makespan_estimate) if isinstance(makespan_estimate, (int, float)) else 0.0

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

                        # 防御性检查：确保m是有效索引
                        try:
                            m = int(m)  # 确保是整数
                            if 0 <= m < n_machines:
                                # 有效索引：假设从当前时间开始（或 machine 可用时间）
                                start_est = max(current_time, machine_status[m])
                                est_complete[op_idx] = start_est + pt
                            else:
                                # 无效索引：使用默认值
                                start_est = current_time
                                est_complete[op_idx] = start_est + pt
                        except (ValueError, TypeError, IndexError):
                            # 类型错误或索引错误：使用默认值
                            start_est = current_time
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

        # If state is already a numpy array (old format compatibility)
        if isinstance(state, np.ndarray):
            # Flatten and normalize using global maximum if available
            state_flat = state.flatten()
            if global_proc_time_max and global_proc_time_max > 0:
                state_flat = state_flat / global_proc_time_max
            elif np.max(state_flat) > 0:
                state_flat = state_flat / np.max(state_flat)  # Fallback to instance-specific max
            # Pad or truncate
            if len(state_flat) > state_dim:
                return state_flat[:state_dim]
            elif len(state_flat) < state_dim:
                return np.pad(state_flat, (0, state_dim - len(state_flat)))
            return state_flat

        # Default: return zeros
        return np.zeros(state_dim)

    return encode_state


# Note: Action encoder removed. Decision Transformer typically uses integer actions directly with nn.Embedding.


def process_jssp_trajectories_for_dt(
    raw_trajectories,
    state_dim=128,
    act_dim=64,
    max_length=128,
    enable_augmentation=False,
    augmentation_ratio=0.0,
    state_noise_std=0.0,
    rtg_scale=1.0,
    max_jobs=None,
    max_machines=None,
    use_gap_weighted_loss=False,
    gap_weight_alpha=6.0,
    gap_weight_power=2.0,
    quality_weight_mix=0.25,
    min_sample_weight=0.35,
    max_sample_weight=3.0,
    rtg_mode="normalized_budget",
    state_tensorization_mode=LEGACY_TRAIN_COMPATIBLE,
):
    """
    Process raw JSSP trajectories for Decision Transformer training.

    Args:
        raw_trajectories: List of raw trajectories
        state_dim: State embedding dimension
        act_dim: Action embedding dimension
        max_length: Maximum sequence length

    Returns:
        Processed dataset ready for training
    """
    # Process trajectories
    processed_trajectories = []

    repaired_count = 0

    for traj in raw_trajectories:
        original_actions = len(traj.get('actions', [])) if isinstance(traj, dict) else 0
        traj = repair_jssp_decision_trajectory(traj, rtg_mode=rtg_mode)
        if isinstance(traj, dict) and len(traj.get('actions', [])) != original_actions:
            repaired_count += 1
        raw_rtg = traj.get('returns_to_go', [])

        processed_traj = {
            'states': traj['states'],
            'actions': traj['actions'],  # Raw integer actions
            'rewards': traj['rewards'],
            'returns_to_go': raw_rtg,
            'timesteps': traj['timesteps']
        }
        if 'metrics' in traj and isinstance(traj['metrics'], dict):
            processed_traj['metrics'] = dict(traj['metrics'])
        if 'expert_actions' in traj:
            processed_traj['expert_actions'] = traj['expert_actions']

        processed_trajectories.append(processed_traj)

    if repaired_count > 0:
        print(f"[TrajectoryRepair] realigned {repaired_count}/{len(raw_trajectories)} trajectories using job_next_op deltas; rtg_mode={rtg_mode}")

    # Convert to Decision Transformer format
    converter = JSSPTrajectoryConverter(state_dim, act_dim)
    converted_trajectories = converter.convert_multiple_trajectories(
        processed_trajectories,
        max_length=max_length
    )

    # Create dataset
    dataset = JSSPDecisionTransformerDataset(
        converted_trajectories,
        max_length=max_length,
        max_jobs=max_jobs,
        max_machines=max_machines,
        enable_augmentation=enable_augmentation,
        augmentation_ratio=augmentation_ratio,
        state_noise_std=state_noise_std,
        rtg_scale=rtg_scale,
        use_gap_weighted_loss=use_gap_weighted_loss,
        gap_weight_alpha=gap_weight_alpha,
        gap_weight_power=gap_weight_power,
        quality_weight_mix=quality_weight_mix,
        min_sample_weight=min_sample_weight,
        max_sample_weight=max_sample_weight,
        state_tensorization_mode=state_tensorization_mode,
    )

    return dataset


def generate_synthetic_jssp_trajectories(n_trajectories=100, n_jobs=10, n_machines=10, max_steps=100, output_path=None):
    """
    Generate synthetic JSSP trajectories for testing.

    Args:
        n_trajectories: Number of trajectories to generate
        n_jobs: Number of jobs
        n_machines: Number of machines
        max_steps: Maximum steps per trajectory
        output_path: Optional path to save trajectories. If provided, trajectories will be saved incrementally.

    Returns:
        List of synthetic trajectories
    """
    trajectories = []

    # Initialize output file if provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            pass

    for i in range(n_trajectories):
        print(f"Generating synthetic trajectory {i+1}/{n_trajectories}")
        # Generate random processing times
        processing_times = np.random.randint(1, 10, size=(n_jobs, n_machines))

        # Initialize trajectory
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'returns_to_go': [],
            'timesteps': []
        }

        # Generate initial state
        initial_state = np.random.rand(n_jobs * n_machines)
        trajectory['states'].append(initial_state)
        trajectory['timesteps'].append(0)  # Initial timestep is 0

        # Generate trajectory steps - note that we now have T steps (actions) but T+1 states
        for t in range(1, max_steps + 1):  # Now we start from 1 to max_steps
            # Random action (job-machine assignment)
            action = np.random.randint(0, n_jobs * n_machines)

            # Random reward
            reward = np.random.uniform(-1.0, 1.0)

            # Random next state (simulating job and machine status after action)
            next_state = np.random.rand(n_jobs * n_machines)

            # Add to trajectory
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['states'].append(next_state)  # Add next state for t
            trajectory['timesteps'].append(t)  # Timestep for state at t

        # Calculate returns-to-go with the correct length and structure
        returns_to_go = []
        future_return = 0
        for reward in reversed(trajectory['rewards']):
            future_return += reward
            returns_to_go.insert(0, future_return)

        # Add the final state's RTG (which should be 0.0)
        returns_to_go.append(0.0)

        # Ensure returns_to_go has the correct length (T+1)
        T = len(trajectory['actions'])
        assert len(returns_to_go) == T + 1, f"Returns_to_go length ({len(returns_to_go)}) must be T + 1 ({T + 1})"

        trajectory['returns_to_go'] = returns_to_go

        trajectories.append(trajectory)

        # Save incrementally if output path is provided
        if output_path:
            serializable_traj = {}
            for key, value in trajectory.items():
                if isinstance(value, np.ndarray):
                    serializable_traj[key] = value.tolist()
                elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    serializable_traj[key] = [item.tolist() for item in value]
                else:
                    serializable_traj[key] = value
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(serializable_traj, ensure_ascii=False) + "\n")
            print(f"Saved trajectory {i+1} to {output_path}")

    return trajectories


def collect_mixed_scale_trajectories(
    env,
    num_trajectories=100,
    max_steps=100,
    output_path=None,
    pure_expert_ratio=0.3,
    epsilon_min=0.1,
    epsilon_max=0.3,
    processing_time_mutation_ratio=0.5,
    processing_time_mutation_pct=0.1,
    mixed_scale_config=None,
    save_light_trajectories=False,
    validate_light_trajectories=False,
):
    """
    收集混合规模的专家轨迹，提高泛化能力。

    Args:
        env: JSSP环境（作为模板）
        num_trajectories: 总轨迹数量
        max_steps: 最大步数
        output_path: 输出路径

    Returns:
        List of collected valid trajectories
    """
    trajectories = []

    # 初始化输出文件（如果提供了output_path）
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            pass

    if mixed_scale_config is None:
        scale_distribution = {
            '5x5': {'count': 0.4, 'n_jobs': 5, 'n_machines': 5},
            '5x10': {'count': 0.3, 'n_jobs': 5, 'n_machines': 10},
            '10x10': {'count': 0.3, 'n_jobs': 10, 'n_machines': 10}
        }
    else:
        scale_distribution = {}
        entries = [seg.strip() for seg in str(mixed_scale_config).split(",") if str(seg).strip()]
        for entry in entries:
            if ":" not in entry or "x" not in entry:
                continue
            scale_name, ratio_part = entry.split(":", 1)
            scale_name = scale_name.strip()
            ratio = float(ratio_part.strip())
            jm = scale_name.lower().split("x")
            if len(jm) != 2:
                continue
            n_jobs = int(jm[0].strip())
            n_machines = int(jm[1].strip())
            if n_jobs <= 0 or n_machines <= 0 or ratio <= 0:
                continue
            scale_distribution[scale_name] = {
                "count": ratio,
                "n_jobs": n_jobs,
                "n_machines": n_machines
            }
        if not scale_distribution:
            raise ValueError(f"Invalid mixed_scale_config: {mixed_scale_config}")
        total_ratio = sum(v["count"] for v in scale_distribution.values())
        for k in scale_distribution.keys():
            scale_distribution[k]["count"] = float(scale_distribution[k]["count"]) / float(max(total_ratio, 1e-8))

    # 为每个规模收集轨迹
    for scale, config in scale_distribution.items():
        target_count = int(num_trajectories * config['count'])

        # 临时修改环境参数以生成指定规模的实例
        original_n_job_range = env.n_job_range.copy()
        original_n_mac_range = env.n_mac_range.copy()

        env.n_job_range = [config['n_jobs'], config['n_jobs']]
        env.n_mac_range = [config['n_machines'], config['n_machines']]

        # 收集该规模的轨迹
        collected_count = 0
        attempts = 0
        max_attempts = target_count * 20  # 增加尝试次数，从10倍改为20倍

        while collected_count < target_count and attempts < max_attempts:
            attempts += 1

            # 创建一个新的环境实例
            new_env = type(env)(
                n_job_range=env.n_job_range,
                n_mac_range=env.n_mac_range,
                seed=None
            )
            new_env.processing_time_mutation_pct = processing_time_mutation_pct if np.random.random() < processing_time_mutation_ratio else 0.0

            # 设置max_steps
            inherited_max_steps = int(getattr(env, "max_steps", 0) or 0)
            env_n_jobs = int(getattr(new_env, "n_jobs", 0) or max(getattr(new_env, "n_job_range", [0])))
            env_n_machines = int(getattr(new_env, "n_machines", 0) or max(getattr(new_env, "n_mac_range", [0])))
            new_env.max_steps = max(int(max_steps), inherited_max_steps, env_n_jobs * env_n_machines)

            # 创建collector
            collector = JSSPTrajectoryCollector(new_env)

            # 使用OR-Tools收集轨迹
            agent = ORToolsAgent(new_env)
            agent.epsilon = 0.0 if np.random.random() < pure_expert_ratio else float(np.random.uniform(epsilon_min, epsilon_max))
            agent.reset()
            trajectory = collector.collect_trajectory(agent, max_steps=max_steps)

            # 检查轨迹质量
            metrics = trajectory['metrics']
            efficiency_ratio = metrics['efficiency_ratio']

            # 多维度过滤（降低阈值以提高通过率）
            # 添加完成度检查：只有调度完成的轨迹才有效
            completed_ok = metrics.get('completed', False)
            efficiency_ok = efficiency_ratio >= 0.60  # 从0.75降低到0.60
            steps_ok = len(trajectory['actions']) <= metrics['optimal_makespan'] * 2.0  # 从1.5倍放宽到2.0倍
            # 移除稳定性检查，因为奖励方差可能很大是正常的

            if completed_ok and efficiency_ok and steps_ok:
                collected_count += 1

                # 添加规模信息到轨迹
                trajectory['metrics']['scale'] = scale
                trajectory['metrics']['n_jobs'] = config['n_jobs']
                trajectory['metrics']['n_machines'] = config['n_machines']

                print(f"✅ Collected {scale} trajectory {collected_count}/{target_count} "
                      f"(Efficiency: {efficiency_ratio:.3f})")

                trajectories.append(trajectory)

                # 保存轨迹
                if output_path:
                    try:
                        to_save = trajectory
                        if save_light_trajectories:
                            to_save = make_light_jssp_trajectory(trajectory, env=new_env)
                            if validate_light_trajectories:
                                validate_light_replay_equivalence(trajectory, to_save)
                        serializable_traj = collector._convert_to_serializable(to_save)
                        collector.append_trajectory_to_file(output_path, serializable_traj)
                    except Exception as e:
                        if save_light_trajectories:
                            raise
                        print(f"❌ Error saving trajectory: {e}")
            else:
                # 打印过滤原因
                reasons = []
                if not completed_ok:
                    reasons.append(f"Not completed (completion_ratio: {metrics.get('completion_ratio', 0.0):.2f})")
                if not efficiency_ok:
                    reasons.append(f"Efficiency {efficiency_ratio:.3f} < 0.60")
                if not steps_ok:
                    reasons.append(f"Steps {len(trajectory['actions'])} > {metrics['optimal_makespan']*2.0:.0f}")

                print(f"❌ Discarded {scale} trajectory (Attempt: {attempts}, Reasons: {', '.join(reasons)})")

        # 恢复原始环境参数
        env.n_job_range = original_n_job_range
        env.n_mac_range = original_n_mac_range

    print(f"🎉 Successfully collected {len(trajectories)}/{num_trajectories} valid trajectories")
    dist_parts = []
    for scale_name, cfg in scale_distribution.items():
        dist_parts.append(f"{scale_name}={int(num_trajectories * cfg['count'])}")
    print("   Scale distribution: " + ", ".join(dist_parts))
    
    return trajectories
