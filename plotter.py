# plotter.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import pandas as pd
import os
import json
import matplotlib

def load_data(filepath):
    """从 .npz 文件加载实验数据。"""
    try:
        with np.load(filepath, allow_pickle=True) as data:
            metadata = json.loads(data['metadata'][0])
            trajectories = data['trajectories']
            return metadata, trajectories
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {filepath}")
        return None, None

def plot_reward_comparison(data_dict, title="Reward Curve Comparison", window_size=20, save_path=None):
    """绘制多算法奖励对比图 (对应图5a)，支持单模型或多模型数据，调整为论文样式。"""
    plt.figure(figsize=(10, 6))
    if not data_dict:
        print("警告: 没有提供数据")
        return

    # 定义六个算法的颜色和线型，匹配自定义样式#3951A2
    model_styles = {
        'MTHA': {'color': "#666562", 'linestyle': '-', 'label': 'MTHA'},
        'HOA': {'color': "#3951A2", 'linestyle': '-', 'label': 'HOA'},
        'MTHA-B': {'color': "#DA383E", 'linestyle': '-', 'label': 'MTHA-B'},
        'MOA': {'color': "#3951A2", 'linestyle': '-', 'label': 'MOA'},
        'HTMA': {'color': "#666562", 'linestyle': '-', 'label': 'HTMA'},
        'HTMA-B': {'color': "#DA383E", 'linestyle': '-', 'label': 'HTMA-B'}
    }

    for label, metadata in data_dict.items():
        rewards = pd.Series(metadata['rewards']).astype(float)
        mean = rewards.rolling(window_size, min_periods=1).mean()
        std = rewards.rolling(window_size, min_periods=1).std().fillna(0)
        
        # 使用特定颜色和线型绘制均值曲线
        style = model_styles.get(label, {'color': (0, 0, 0), 'linestyle': '-', 'label': label})
        plt.plot(mean, **style)
        
        # 填充标准差区域，使用对应颜色
        x = np.array(mean.index, dtype=float)
        y1 = np.array(mean - std, dtype=float)
        y2 = np.array(mean + std, dtype=float)
        plt.fill_between(x, y1, y2, alpha=0.2, color=style['color'])

        # 绘制异常点，使用红色+
        is_outlier = (rewards - mean).abs() > 2 * std
        plt.plot(rewards[is_outlier], '+', markersize=5, color='red', label='_nolegend_' if label == list(data_dict.keys())[0] else "")

    # 设置坐标轴范围，匹配论文
    plt.ylim(-600, 400)
    plt.xlim(0, 100)
    # 设置标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)

    # 添加网格线，匹配论文的垂直网格样式
    plt.grid(True, which='both', axis='y', linestyle='-', alpha=0.1)
    plt.grid(True, which='both', axis='x', linestyle='-', alpha=0.1)

    # 调整图例位置到右上角，支持六个算法
    plt.legend(loc='upper right', fontsize=10)
    # 调整布局，避免裁剪
    plt.tight_layout()
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_successful_rewards(data_dict, title="Rewards of Successful Episodes", save_path=None):
    """绘制成功轮次的奖励散点图 (对应图5b)，支持单模型或多模型数据。"""
    plt.figure(figsize=(10, 6))
    if not data_dict:
        print("警告: 没有提供数据")
        return
    markers = ['v', 's', 'o', '^', '>', '<', 'p']  # 支持更多模型
    for marker, (label, metadata) in zip(markers, data_dict.items()):
        rewards = np.array(metadata['rewards'])
        successes = np.array(metadata['successes'])
        successful_rewards = rewards[successes]
        episode_indices = np.where(successes)[0]
        plt.scatter(episode_indices, successful_rewards, marker=marker, label=label)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_rate_comparison(data_dict, rate_type='successes', title="Success Rate Comparison", save_path=None):
    """绘制成功率或坠毁率对比图 (对应图5c, 5d)，支持单模型或多模型数据。"""
    plt.figure(figsize=(10, 6))
    if not data_dict:
        print("警告: 没有提供数据")
        return
    window_size = 20
    for label, metadata in data_dict.items():
        rate_data = pd.Series(metadata[rate_type]).astype(int)
        mean_rate = rate_data.rolling(window_size, min_periods=1).mean()
        std_rate = rate_data.rolling(window_size, min_periods=1).std().fillna(0)
        
        plt.plot(mean_rate, label=label)
        plt.fill_between(mean_rate.index, mean_rate - std_rate, mean_rate + std_rate, alpha=0.2)

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Rate", fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_action_percentage(metadata_dict, algo_names=None, title="Action Selection Percentage", save_path=None):
    """绘制动作百分比堆叠图 (对应图6)，支持单模型或多模型数据。"""
    plt.figure(figsize=(12, 6))
    if not metadata_dict:
        print("警告: 没有提供数据")
        return
    if algo_names is None:
        algo_names = list(metadata_dict.keys())
    
    for algo_name in algo_names:
        if algo_name not in metadata_dict:
            print(f"警告: {algo_name} 不在 metadata_dict 中")
            continue
        metadata = metadata_dict[algo_name]
        data = metadata['action_sources']
        df = pd.DataFrame(data)
        total = df.sum(axis=1); total[total == 0] = 1
        df_percent = df.div(total, axis=0)
        
        x_offset = algo_names.index(algo_name) * 0.2 - (len(algo_names) - 1) * 0.1
        df_percent.plot(kind='bar', stacked=True, width=0.8, position=x_offset, ax=plt.gca(),
                        label=f"{algo_name}", color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # 获取 action_sources 的长度
    total_episodes = len(next(iter(metadata_dict.values()))['action_sources'])
    
    # 设置横坐标刻度，每隔10个Episode显示一次数字,数字显示横向
    plt.xticks(rotation=0)
    tick_pos = np.arange(0, 110, 10) 
    plt.xticks(tick_pos, tick_pos)

    plt.title(title, fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Percent", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title="Action Source", loc='upper right')

    # 调整布局，避免裁剪
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_trajectories(trajectories_dict, metadata_dict=None, title="Flight Trajectories", save_path=None):
    """绘制飞行轨迹图，包含渐变色和起点/终点标记，支持单模型或多模型数据。"""
    plt.figure(figsize=(8, 6))
    if not trajectories_dict:
        print("警告: 没有提供轨迹数据")
        return
    if metadata_dict is None:
        metadata_dict = {key: None for key in trajectories_dict.keys()}
    
    # 计算总体成功率
    overall_success_rate = 0.0
    if metadata_dict:
        for metadata in metadata_dict.values():
            if metadata:
                success_rate = metadata['successes'].count(True) / len(metadata['successes'])
                overall_success_rate += success_rate
        overall_success_rate /= len(metadata_dict)
    title = f"{title} (success rate={overall_success_rate:.2f})"
    plt.title(title, fontsize=16)

    # 绘制每条轨迹
    for label, traj_list in trajectories_dict.items():
        for traj in traj_list:
            if len(traj) < 2:
                continue
            
            points = np.array(traj)
            segments = np.concatenate([points[:-1], points[1:]], axis=1).reshape(-1, 2, 2)
            
            norm = plt.Normalize(0, len(traj) - 1)
            colors = matplotlib.colormaps.get_cmap('gist_earth')(norm(np.arange(len(traj) - 1)))
            
            lc = LineCollection(segments, colors=colors, linewidth=1.5, alpha=0.8, label=label if traj is traj_list[0] else "")
            plt.gca().add_collection(lc)
            
            plt.scatter(traj[0, 0], traj[0, 1], c='red', s=50, zorder=5)
            plt.scatter(traj[-1, 0], traj[-1, 1], c="#002A6EBE", s=20, zorder=3)

    # 设置坐标轴范围和标签
    plt.xlabel("X-Coordinate", fontsize=12)
    plt.ylabel("Y-Coordinate", fontsize=12)
    plt.xlim(-1, 1)
    plt.ylim(-0.2, 1.2)
    plt.grid(True)
    # 绘制着陆平台
    plt.plot([-0.2, 0.2], [0, 0], color='red', linewidth=3, label='Landing Pad')
    # 添加图例
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_reward_boxplot(data_dict, title="Reward Distribution Comparison", save_path=None):
    """绘制奖励分布的箱线图，复现论文样式，调整为 MOA, HTMA, HTMA-B 样式。"""
    plt.figure(figsize=(10, 6))
    
    # 提取每个算法的奖励数据
    labels = []
    rewards_data = []
    for algo_name, metadata in data_dict.items():
        labels.append(algo_name)
        rewards_data.append(np.array(metadata['rewards']).astype(float))
    # 绘制箱线图
    box = plt.boxplot(rewards_data, patch_artist=True, labels=labels)
    # 设置箱体为蓝色轮廓，无填充或极浅填充
    for patch in box['boxes']:
        patch.set_facecolor('none')  # 移除填充
        patch.set_edgecolor('blue')  # 蓝色边框
        patch.set_linewidth(1.5)     # 边框宽度
    # 设置中位数线为红色
    for median in box['medians']:
        median.set_color('red')
        median.set_linewidth(1.5)    # 适当加粗中位数线
    # 设置胡须和顶点为黑色，较细，虚线
    for whisker in box['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)     # 减小线宽
        whisker.set_linestyle('--')  # 虚线
    for cap in box['caps']:
        cap.set_color('black')
        cap.set_linewidth(1)         # 减小线宽
    # 设置异常值样式
    for flier in box['fliers']:
        flier.set_marker('+')
        flier.set_color('red')
        flier.set_markersize(8)
    # 设置坐标轴范围和标签
    plt.ylim(-600, 600)
    plt.title(title, fontsize=16)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)  # 降低透明度
    # 调整布局
    plt.tight_layout()
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 加载数据
    moa_ddqn_mthab_meta, moa_ddqn_mthab_traj = load_data("results/MOA/moa_results.npz")
    hoa_ddqn_mthab_meta, hoa_ddqn_mthab_traj = load_data("results/HOA/hoa_results.npz")
    mtha_ddqn_mthab_meta, mtha_ddqn_mthab_traj = load_data("results/MTHA/mtha_results.npz")
    mtha_b_ddqn_mthab_meta, mtha_b_ddqn_mthab_traj = load_data("results/MTHA-B/mtha-b_results.npz")
    htma_b_ddqn_mthab_meta, htma_b_ddqn_mthab_traj = load_data("results/HTMA-B/htma-b_results.npz")
    htma_ddqn_mthab_meta, htma_ddqn_mthab_traj = load_data("results/HTMA/htma_results.npz")
    # 绘制MOA(Double DQN)轨迹图
    # plot_reward_comparison({'MOA': moa_ddqn_mthab_meta}, save_path="results/MOA/moa_Double_DQN_reward.png")
    # plot_trajectories({'MOA': moa_ddqn_mthab_traj}, {'MOA': moa_ddqn_mthab_meta}, save_path="results/MOA/moa_Double_DQN_trajectory.png")

    # 绘制HOA(Double DQN)轨迹图
    # plot_reward_comparison({'HOA': hoa_ddqn_mthab_meta}, save_path="results/HOA/hoa_Double_DQN_reward.png")
    # plot_trajectories({'HOA': hoa_ddqn_mthab_traj}, {'HOA': hoa_ddqn_mthab_meta}, save_path="results/HOA/hoa_Double_DQN_trajectory.png")
    
    #  绘制MTHA(Double DQN)轨迹图
    # plot_reward_comparison({'MTHA': mtha_ddqn_mthab_meta}, save_path="results/MTHA/mtha_Double_DQN_reward.png")
    # plot_trajectories({'MTHA': mtha_ddqn_mthab_traj}, {'MTHA': mtha_ddqn_mthab_meta}, save_path="results/MTHA/mtha_Double_DQN_trajectory.png")
    
    # 绘制MTHA-B(Double DQN)轨迹图
    # plot_reward_comparison({'MTHA-B': mtha_b_ddqn_mthab_meta}, save_path="results/MTHA-B/mtha_b_Double_DQN_reward.png")
    # plot_trajectories({'MTHA-B': mtha_b_ddqn_mthab_traj}, {'MTHA-B': mtha_b_ddqn_mthab_meta}, save_path="results/MTHA-B/mtha_b_Double_DQN_trajectory.png")
    
    # 绘制HTMA-B(Double DQN)轨迹图
    # plot_reward_comparison({'HTMA-B': htma_b_ddqn_mthab_meta}, save_path="results/HTMA-B/htma_b_Double_DQN_reward.png")
    # plot_trajectories({'HTMA-B': mtha_b_ddqn_mthab_traj}, {'HTMA-B': htma_b_ddqn_mthab_meta}, save_path="results/HTMA-B/htma_b_Double_DQN_trajectory.png")
    
    # 绘制HTMA(Double DQN)轨迹图
    # plot_reward_comparison({'HTMA': htma_ddqn_mthab_meta}, save_path="results/HTMA/htma_Double_DQN_reward.png")
    # plot_trajectories({'HTMA': htma_ddqn_mthab_traj}, {'HTMA': htma_ddqn_mthab_meta}, save_path="results/HTMA/htma_Double_DQN_trajectory.png")
    
    # 合并绘制 HOA 和 MTHA-B，MTHA奖励图
    combined_reward_data = {'HOA': hoa_ddqn_mthab_meta, 'MTHA': mtha_ddqn_mthab_meta, 'MTHA-B': mtha_b_ddqn_mthab_meta}
    plot_reward_comparison(combined_reward_data, title="Combined Reward Comparison of HOA, MTHA, MTHA-B", save_path="results/combined_hoa_mtha_mtha_b_reward.png")
    
    # 合并绘制 MOA 和 HTMA-B,HTMA奖励图
    combined_reward_data_moa_htma_b = {'MOA': moa_ddqn_mthab_meta, 'HTMA': htma_ddqn_mthab_meta, 'HTMA-B': htma_b_ddqn_mthab_meta,}
    plot_reward_comparison(combined_reward_data_moa_htma_b, title="Combined Reward Comparison of MOA, HTMA, HTMA-B", save_path="results/combined_moa_htma_b_htma_reward.png")
    
    # 绘制MTHA的动作占比图
    # plot_action_percentage({'MTHA': mtha_ddqn_mthab_meta}, algo_names=['MTHA'], title="Action Selection Percentage of MTHA", save_path="results/MTHA/mtha_action_percentage.png")
    # 绘制MTHA-B的动作占比图
    # plot_action_percentage({'MTHA-B': mtha_b_ddqn_mthab_meta}, algo_names=['MTHA-B'], title="Action Selection Percentage of MTHA-B", save_path="results/MTHA-B/mtha_b_action_percentage.png")
    # 绘制HTMA的动作占比图
    # plot_action_percentage({'HTMA': htma_ddqn_mthab_meta}, algo_names=['HTMA'], title="Action Selection Percentage of HTMA", save_path="results/HTMA/htma_action_percentage.png")
    # 绘制HTMA-B的动作占比图
    # plot_action_percentage({'HTMA-B': htma_b_ddqn_mthab_meta}, algo_names=['HTMA-B'], title="Action Selection Percentage of HTMA-B", save_path="results/HTMA-B/htma_b_action_percentage.png")
    
    # 合并绘制 HOA 和 MTHA-B，MTHA奖励箱型图
    # combined_boxplot_data = {'HOA': hoa_ddqn_mthab_meta,'MTHA-B': mtha_b_ddqn_mthab_meta,'MTHA': mtha_ddqn_mthab_meta}
    # plot_reward_boxplot(combined_boxplot_data, title="Combined Reward Boxplot of HOA, MTHA-B, MTHA", save_path="results/combined_hoa_mtha_b_mtha_reward_boxplot.png")
    # 合并绘制 MOA 和 HTMA-B,HTMA奖励箱型图
    # combined_boxplot_data_moa_htma_b = {'MOA': moa_ddqn_mthab_meta,'HTMA-B': htma_b_ddqn_mthab_meta,'HTMA': htma_ddqn_mthab_meta}
    # plot_reward_boxplot(combined_boxplot_data_moa_htma_b, title="Combined Reward Boxplot of MOA, HTMA-B, HTMA", save_path="results/combined_moa_htma_b_htma_reward_boxplot.png")
