"""
一键收集多尺度轨迹数据

使用方法：
    python collect_multi_scale.py --total 10000 --output_dir ./data
"""

import argparse
import json
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def collect_multi_scale_trajectories(total_trajectories: int, 
                                      output_dir: str,
                                      max_steps: int = 100):
    """
    收集多尺度轨迹
    
    默认比例：
    - 5x5:  20% (简单问题，帮助模型学习基础)
    - 5x10: 50% (目标规模)
    - 10x10: 30% (困难问题，提升泛化)
    """
    from Envs.JSSPEnv.JSSPEnv import JSSPEnv
    from dt_data_processing import JSSPTrajectoryCollector, ORToolsAgent
    
    # 规模配置
    scale_configs = [
        {'n_jobs': 5, 'n_machines': 5, 'ratio': 0.20, 'name': '5x5'},
        {'n_jobs': 10, 'n_machines': 5, 'ratio': 0.50, 'name': '5x10'},
        {'n_jobs': 10, 'n_machines': 10, 'ratio': 0.30, 'name': '10x10'}
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_trajectories = []
    
    print(f"\n{'='*60}")
    print(f"多尺度轨迹收集")
    print(f"{'='*60}")
    print(f"总轨迹数: {total_trajectories}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    for config in scale_configs:
        n_jobs = config['n_jobs']
        n_machines = config['n_machines']
        ratio = config['ratio']
        name = config['name']
        
        n_traj = int(total_trajectories * ratio)
        
        print(f"\n📊 收集 {name} 规模轨迹: {n_traj} 条 ({ratio*100:.0f}%)")
        print("-" * 40)
        
        scale_trajectories = []
        
        for i in range(n_traj):
            # 创建环境
            env = JSSPEnv(
                n_job_range=[n_jobs, n_jobs],
                n_mac_range=[n_machines, n_machines],
                seed=None
            )
            env.max_steps = max_steps
            
            # 创建 Agent
            agent = ORToolsAgent(env)
            agent.epsilon = 0.0
            agent.reset()
            
            # 收集轨迹
            collector = JSSPTrajectoryCollector(env)
            trajectory = collector.collect_trajectory(agent, max_steps=max_steps)
            
            # 添加规模信息
            trajectory['scale'] = name
            trajectory['n_jobs'] = n_jobs
            trajectory['n_machines'] = n_machines
            
            scale_trajectories.append(trajectory)
            all_trajectories.append(trajectory)
            
            # 打印进度
            if (i + 1) % 100 == 0 or (i + 1) == n_traj:
                eff = trajectory.get('metrics', {}).get('efficiency_ratio', 0)
                print(f"  [{name}] {i+1}/{n_traj} 完成 (最新效率: {eff:.3f})")
        
        # 保存单独规模的文件
        scale_file = os.path.join(output_dir, f"ortools_{name}.json")
        save_trajectories(scale_trajectories, scale_file)
        print(f"  ✅ 保存到 {scale_file}")
    
    # 保存合并文件
    combined_file = os.path.join(output_dir, "ortools_multi_scale.json")
    save_trajectories(all_trajectories, combined_file)
    
    # 统计信息
    print(f"\n{'='*60}")
    print(f"收集完成!")
    print(f"{'='*60}")
    print(f"总轨迹数: {len(all_trajectories)}")
    
    for config in scale_configs:
        name = config['name']
        count = sum(1 for t in all_trajectories if t.get('scale') == name)
        avg_eff = sum(t.get('metrics', {}).get('efficiency_ratio', 0) 
                     for t in all_trajectories if t.get('scale') == name) / max(count, 1)
        print(f"  {name}: {count} 条, 平均效率: {avg_eff:.3f}")
    
    print(f"\n合并文件: {combined_file}")
    print(f"{'='*60}")
    
    return all_trajectories


def save_trajectories(trajectories, filepath):
    """保存轨迹到文件"""
    import numpy as np
    
    serializable = []
    for traj in trajectories:
        s_traj = {}
        for key, value in traj.items():
            if isinstance(value, np.ndarray):
                s_traj[key] = value.tolist()
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        new_list.append(item.tolist())
                    elif isinstance(item, dict):
                        new_dict = {}
                        for k, v in item.items():
                            if isinstance(v, np.ndarray):
                                new_dict[k] = v.tolist()
                            else:
                                new_dict[k] = v
                        new_list.append(new_dict)
                    else:
                        new_list.append(item)
                s_traj[key] = new_list
            else:
                s_traj[key] = value
        serializable.append(s_traj)
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='收集多尺度 JSSP 轨迹')
    parser.add_argument('--total', type=int, default=10000, 
                        help='总轨迹数量')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='输出目录')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='每条轨迹最大步数')
    
    args = parser.parse_args()
    
    collect_multi_scale_trajectories(
        total_trajectories=args.total,
        output_dir=args.output_dir,
        max_steps=args.max_steps
    )
