"""
数据集优化模块 - 用于降低 Decision Transformer 的 Gap

基于以下论文和方法：
1. Decision Transformer (Chen et al., 2021)
2. Trajectory Transformer (Janner et al., 2021)
3. Offline RL with Suboptimal Demonstrations
4. JSSP 领域的神经网络方法

优化策略：
1. 数据增强：状态噪声、轨迹裁剪
2. RTG 优化：目标回报调整
3. 多尺度训练：混合不同规模问题
4. 课程学习：从简单到复杂
"""

import numpy as np
import json
import copy
from typing import List, Dict, Tuple, Optional


class DatasetOptimizer:
    """
    数据集优化器 - 通过多种技术提升轨迹质量和多样性
    """
    
    def __init__(self, trajectories: List[Dict]):
        """
        Args:
            trajectories: 原始轨迹列表
        """
        self.original_trajectories = trajectories
        self.optimized_trajectories = []
    
    def augment_state_noise(self, trajectory: Dict, noise_std: float = 0.01) -> Dict:
        """
        策略1：状态噪声增强
        
        对状态添加微小高斯噪声，增加数据多样性，提升泛化能力。
        论文依据：Dropout 和噪声注入在深度学习中的正则化效果
        
        Args:
            trajectory: 原始轨迹
            noise_std: 噪声标准差（建议 0.005-0.02）
            
        Returns:
            增强后的轨迹
        """
        augmented = copy.deepcopy(trajectory)
        states = augmented.get('states', [])
        
        if not states:
            return augmented
        
        augmented_states = []
        for state in states:
            if isinstance(state, dict):
                noisy_state = {}
                for key, value in state.items():
                    if isinstance(value, (list, np.ndarray)):
                        arr = np.array(value, dtype=np.float32)
                        # 只对数值型数据添加噪声
                        if arr.dtype in [np.float32, np.float64, np.int32, np.int64]:
                            noise = np.random.normal(0, noise_std, arr.shape).astype(arr.dtype)
                            noisy_state[key] = arr + noise
                        else:
                            noisy_state[key] = value
                    elif isinstance(value, (int, float)):
                        noise = np.random.normal(0, noise_std)
                        noisy_state[key] = value + noise
                    else:
                        noisy_state[key] = value
                augmented_states.append(noisy_state)
            else:
                augmented_states.append(state)
        
        augmented['states'] = augmented_states
        augmented['augmentation'] = 'state_noise'
        return augmented
    
    def augment_trajectory_crop(self, trajectory: Dict, crop_ratio: float = 0.1) -> Dict:
        """
        策略2：轨迹裁剪增强
        
        随机裁剪轨迹的开头或结尾，让模型学习从不同起点/终点处理问题。
        
        Args:
            trajectory: 原始轨迹
            crop_ratio: 裁剪比例（建议 0.05-0.15）
            
        Returns:
            裁剪后的轨迹
        """
        augmented = copy.deepcopy(trajectory)
        states = augmented.get('states', [])
        actions = augmented.get('actions', [])
        rewards = augmented.get('rewards', [])
        rtg = augmented.get('returns_to_go', [])
        timesteps = augmented.get('timesteps', [])
        
        if len(actions) < 10:  # 太短不裁剪
            return augmented
        
        n = len(actions)
        crop_n = int(n * crop_ratio)
        
        if crop_n < 1:
            return augmented
        
        # 随机选择裁剪开头还是结尾
        if np.random.random() < 0.5:
            # 裁剪开头
            start = crop_n
            augmented['states'] = states[start:]
            augmented['actions'] = actions[start:]
            augmented['rewards'] = rewards[start:]
            augmented['returns_to_go'] = rtg[start:]
            augmented['timesteps'] = list(range(len(augmented['states'])))
        else:
            # 裁剪结尾
            end = n - crop_n
            augmented['states'] = states[:end+1]
            augmented['actions'] = actions[:end]
            augmented['rewards'] = rewards[:end]
            augmented['returns_to_go'] = rtg[:end+1]
            augmented['timesteps'] = list(range(len(augmented['states'])))
        
        augmented['augmentation'] = 'trajectory_crop'
        return augmented
    
    def optimize_rtg(self, trajectory: Dict, rtg_scale: float = 1.1) -> Dict:
        """
        策略3：RTG (Returns-to-Go) 优化
        
        Decision Transformer 的核心是用 RTG 来引导行为。
        通过放大 RTG，可以让模型追求更好的目标。
        
        论文依据：DT 在测试时使用更高的 RTG 可以获得更好的性能
        
        Args:
            trajectory: 原始轨迹
            rtg_scale: RTG 缩放因子（建议 1.05-1.2）
            
        Returns:
            RTG 优化后的轨迹
        """
        augmented = copy.deepcopy(trajectory)
        rtg = augmented.get('returns_to_go', [])
        
        if not rtg:
            return augmented
        
        # 缩放 RTG
        scaled_rtg = [r * rtg_scale for r in rtg]
        augmented['returns_to_go'] = scaled_rtg
        augmented['rtg_scaled'] = True
        augmented['rtg_scale'] = rtg_scale
        
        return augmented
    
    def create_curriculum_batches(self, trajectories: List[Dict], 
                                   n_stages: int = 3) -> List[List[Dict]]:
        """
        策略4：课程学习
        
        将轨迹按难度分组，先训练简单的再训练困难的。
        难度定义：步数越少越简单，效率越高越简单
        
        Args:
            trajectories: 轨迹列表
            n_stages: 课程阶段数
            
        Returns:
            按难度分组的轨迹列表
        """
        # 计算每条轨迹的难度分数
        scored_trajectories = []
        for traj in trajectories:
            metrics = traj.get('metrics', {})
            n_steps = len(traj.get('actions', []))
            efficiency = metrics.get('efficiency_ratio', 0.5)
            
            # 难度分数：步数多 + 效率低 = 更难
            difficulty = n_steps * (2 - efficiency)
            scored_trajectories.append((difficulty, traj))
        
        # 按难度排序
        scored_trajectories.sort(key=lambda x: x[0])
        
        # 分成 n_stages 组
        n = len(scored_trajectories)
        stage_size = n // n_stages
        
        stages = []
        for i in range(n_stages):
            start = i * stage_size
            end = (i + 1) * stage_size if i < n_stages - 1 else n
            stage_trajectories = [t[1] for t in scored_trajectories[start:end]]
            stages.append(stage_trajectories)
        
        return stages
    
    def apply_all_augmentations(self, augmentation_ratio: float = 0.3,
                                 noise_std: float = 0.01,
                                 crop_ratio: float = 0.1,
                                 rtg_scale: float = 1.1) -> List[Dict]:
        """
        应用所有增强策略
        
        Args:
            augmentation_ratio: 增强轨迹的比例
            noise_std: 状态噪声标准差
            crop_ratio: 轨迹裁剪比例
            rtg_scale: RTG 缩放因子
            
        Returns:
            增强后的轨迹列表（包含原始轨迹）
        """
        all_trajectories = list(self.original_trajectories)
        
        for traj in self.original_trajectories:
            if np.random.random() < augmentation_ratio:
                # 随机选择一种增强方法
                aug_type = np.random.choice(['noise', 'crop', 'rtg'])
                
                if aug_type == 'noise':
                    augmented = self.augment_state_noise(traj, noise_std)
                elif aug_type == 'crop':
                    augmented = self.augment_trajectory_crop(traj, crop_ratio)
                else:  # rtg
                    augmented = self.optimize_rtg(traj, rtg_scale)
                
                all_trajectories.append(augmented)
        
        self.optimized_trajectories = all_trajectories
        return all_trajectories
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        trajectories = self.optimized_trajectories or self.original_trajectories
        
        efficiencies = []
        n_steps_list = []
        
        for traj in trajectories:
            metrics = traj.get('metrics', {})
            efficiencies.append(metrics.get('efficiency_ratio', 0))
            n_steps_list.append(len(traj.get('actions', [])))
        
        return {
            'n_trajectories': len(trajectories),
            'avg_efficiency': np.mean(efficiencies) if efficiencies else 0,
            'std_efficiency': np.std(efficiencies) if efficiencies else 0,
            'min_efficiency': np.min(efficiencies) if efficiencies else 0,
            'max_efficiency': np.max(efficiencies) if efficiencies else 0,
            'avg_steps': np.mean(n_steps_list) if n_steps_list else 0,
            'augmentation_count': len(trajectories) - len(self.original_trajectories)
        }


class MultiScaleTrainer:
    """
    多尺度训练器 - 混合不同规模的问题进行训练
    
    论文依据：多尺度训练可以提升模型的泛化能力
    """
    
    def __init__(self, scale_configs: List[Dict] = None):
        """
        Args:
            scale_configs: 规模配置列表，例如:
                [
                    {'n_jobs': 5, 'n_machines': 5, 'ratio': 0.3},
                    {'n_jobs': 5, 'n_machines': 10, 'ratio': 0.4},
                    {'n_jobs': 10, 'n_machines': 10, 'ratio': 0.3}
                ]
        """
        if scale_configs is None:
            self.scale_configs = [
                {'n_jobs': 5, 'n_machines': 5, 'ratio': 0.2, 'name': '5x5'},
                {'n_jobs': 5, 'n_machines': 10, 'ratio': 0.5, 'name': '5x10'},
                {'n_jobs': 10, 'n_machines': 10, 'ratio': 0.3, 'name': '10x10'}
            ]
        else:
            self.scale_configs = scale_configs
    
    def collect_multi_scale_trajectories(self, env_class, total_trajectories: int,
                                          max_steps: int = 100,
                                          output_path: str = None) -> List[Dict]:
        """
        收集多尺度轨迹
        
        Args:
            env_class: 环境类
            total_trajectories: 总轨迹数
            max_steps: 每条轨迹最大步数
            output_path: 保存路径
            
        Returns:
            混合尺度的轨迹列表
        """
        from dt_data_processing import JSSPTrajectoryCollector, ORToolsAgent
        import json
        import os
        
        all_trajectories = []
        
        for config in self.scale_configs:
            n_jobs = config['n_jobs']
            n_machines = config['n_machines']
            ratio = config['ratio']
            name = config.get('name', f'{n_jobs}x{n_machines}')
            
            n_traj = int(total_trajectories * ratio)
            
            print(f"\n收集 {name} 规模轨迹: {n_traj} 条")
            
            for i in range(n_traj):
                env = env_class(
                    n_job_range=[n_jobs, n_jobs],
                    n_mac_range=[n_machines, n_machines],
                    seed=None
                )
                env.max_steps = max_steps
                
                agent = ORToolsAgent(env)
                agent.epsilon = 0.0
                agent.reset()
                
                collector = JSSPTrajectoryCollector(env)
                trajectory = collector.collect_trajectory(agent, max_steps=max_steps)
                
                trajectory['scale'] = name
                all_trajectories.append(trajectory)
                
                if (i + 1) % 100 == 0:
                    eff = trajectory.get('metrics', {}).get('efficiency_ratio', 0)
                    print(f"  [{name}] {i+1}/{n_traj} (效率: {eff:.3f})")
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # 序列化轨迹
            serializable = []
            for traj in all_trajectories:
                s_traj = {}
                for key, value in traj.items():
                    if isinstance(value, np.ndarray):
                        s_traj[key] = value.tolist()
                    elif isinstance(value, list):
                        s_traj[key] = [
                            v.tolist() if isinstance(v, np.ndarray) else 
                            {k: vv.tolist() if isinstance(vv, np.ndarray) else vv 
                             for k, vv in v.items()} if isinstance(v, dict) else v
                            for v in value
                        ]
                    else:
                        s_traj[key] = value
                serializable.append(s_traj)
            
            with open(output_path, 'w') as f:
                json.dump(serializable, f)
            print(f"\n✅ 保存 {len(all_trajectories)} 条多尺度轨迹到 {output_path}")
        
        return all_trajectories


class InferenceOptimizer:
    """
    推理优化器 - 提升测试时的性能
    """
    
    @staticmethod
    def beam_search_action(model, state, rtg, timestep, available_actions: List[int],
                           beam_width: int = 3, temperature: float = 1.0) -> int:
        """
        使用 Beam Search 选择动作
        
        比贪心选择更好，可以考虑多个候选路径。
        
        Args:
            model: Decision Transformer 模型
            state: 当前状态
            rtg: Returns-to-Go
            timestep: 时间步
            available_actions: 可用动作列表
            beam_width: Beam 宽度
            temperature: Softmax 温度
            
        Returns:
            选择的动作
        """
        import torch
        import torch.nn.functional as F
        
        # 获取动作概率分布
        with torch.no_grad():
            logits = model.get_action_logits(state, rtg, timestep)
            
            # 应用温度
            if temperature != 1.0:
                logits = logits / temperature
            
            probs = F.softmax(logits, dim=-1)
            
            # 只考虑可用动作
            mask = torch.zeros_like(probs)
            for a in available_actions:
                mask[a] = 1.0
            
            masked_probs = probs * mask
            masked_probs = masked_probs / masked_probs.sum()
            
            # 选择 top-k 动作
            top_probs, top_indices = torch.topk(masked_probs, min(beam_width, len(available_actions)))
            
            # 从 top-k 中采样
            selected_idx = torch.multinomial(top_probs, 1).item()
            action = top_indices[selected_idx].item()
        
        return action
    
    @staticmethod
    def get_optimal_rtg_target(problem_size: Tuple[int, int], 
                                scaling_factor: float = 1.1) -> float:
        """
        计算最优 RTG 目标
        
        Decision Transformer 在推理时使用更高的 RTG 可以获得更好性能。
        
        Args:
            problem_size: (n_jobs, n_machines)
            scaling_factor: 缩放因子
            
        Returns:
            推荐的 RTG 目标
        """
        n_jobs, n_machines = problem_size
        
        # 基于问题规模估算最优 RTG
        # 这里使用经验公式，可以根据实际情况调整
        base_rtg = -n_jobs * n_machines * 10  # 负数因为 JSSP 通常使用负 reward
        optimal_rtg = base_rtg * scaling_factor
        
        return optimal_rtg


def create_optimized_dataset(input_path: str, output_path: str,
                              augmentation_ratio: float = 0.3,
                              noise_std: float = 0.01,
                              rtg_scale: float = 1.1) -> Dict:
    """
    创建优化后的数据集
    
    Args:
        input_path: 原始轨迹文件路径
        output_path: 输出路径
        augmentation_ratio: 增强比例
        noise_std: 状态噪声标准差
        rtg_scale: RTG 缩放因子
        
    Returns:
        数据集统计信息
    """
    import json
    
    # 加载原始轨迹
    with open(input_path, 'r') as f:
        trajectories = json.load(f)
    
    print(f"加载 {len(trajectories)} 条原始轨迹")
    
    # 创建优化器
    optimizer = DatasetOptimizer(trajectories)
    
    # 应用增强
    optimized = optimizer.apply_all_augmentations(
        augmentation_ratio=augmentation_ratio,
        noise_std=noise_std,
        rtg_scale=rtg_scale
    )
    
    # 获取统计信息
    stats = optimizer.get_statistics()
    
    print(f"\n数据集统计:")
    print(f"  原始轨迹数: {len(trajectories)}")
    print(f"  增强后轨迹数: {stats['n_trajectories']}")
    print(f"  增强轨迹数: {stats['augmentation_count']}")
    print(f"  平均效率: {stats['avg_efficiency']:.3f}")
    print(f"  效率标准差: {stats['std_efficiency']:.3f}")
    
    # 保存优化后的数据集
    with open(output_path, 'w') as f:
        json.dump(optimized, f)
    
    print(f"\n✅ 保存优化后的数据集到 {output_path}")
    
    return stats


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集优化工具')
    parser.add_argument('--input', type=str, required=True, help='原始轨迹文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出路径')
    parser.add_argument('--augmentation_ratio', type=float, default=0.3, help='增强比例')
    parser.add_argument('--noise_std', type=float, default=0.01, help='状态噪声标准差')
    parser.add_argument('--rtg_scale', type=float, default=1.1, help='RTG缩放因子')
    
    args = parser.parse_args()
    
    stats = create_optimized_dataset(
        input_path=args.input,
        output_path=args.output,
        augmentation_ratio=args.augmentation_ratio,
        noise_std=args.noise_std,
        rtg_scale=args.rtg_scale
    )
