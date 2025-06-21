# main_htma_b.py
import gymnasium as gym
import torch
import numpy as np
import argparse
import os
import pickle

import config
from agents import DoubleDQN, ReplayBuffer, ActionValueCredibility, KnnBoundary 
from human_controller import HumanController
from data_handler import DataHandler

def print_step_details(step, h_act, m_act, b_act, h_q, m_q, b_q, h_cred, m_cred, final_act, source, algorithm):
    """格式化并打印单步决策的详细信息"""
    action_map = {0: "无操作", 1: "喷左", 2: "主引擎", 3: "喷右", None: "无输入"}
    
    h_str = f"人: {action_map[h_act]:<6} (Q: {h_q:7.2f}, Cred: {h_cred:.2f})" if h_act is not None else "人: 无输入"
    m_str = f"机: {action_map[m_act]:<6} (Q: {m_q:7.2f}, Cred: {m_cred:.2f})" if m_act is not None else "机: 无动作"
    b_str = f"界: {action_map[b_act]:<6} (Q: {b_q:7.2f})" if b_act is not None else "界: 无动作"
    
    if source == 'human': h_str = f"\033[92m{h_str}\033[0m"
    if source == 'machine': m_str = f"\033[92m{m_str}\033[0m"
    if source == 'boundary': b_str = f"\033[92m{b_str}\033[0m"

    print(f"--- 算法: {algorithm}, 步数: {step} ---")
    print(h_str)
    print(m_str)
    if algorithm == 'htma-b':
        print(b_str)
    print(f"最终决策: {action_map[final_act]} (来自: {source})")
    print("-" * 30)

def arbitrator_htma_b(m_action, h_action, b_action, m_cred, h_cred, agent, state):
    """HTMA-B 仲裁器"""
    m_q = agent.get_q_value(state, m_action)
    b_q = agent.get_q_value(state, b_action)
    h_q = agent.get_q_value(state, h_action) if h_action is not None else float('-inf')
    
    q_values_dict = {'m_q': m_q, 'h_q': h_q, 'b_q': b_q}

    if h_action is not None and h_cred >= m_cred and h_q >= max(m_q, b_q):
        source = 'human'
        final_action = h_action
    elif m_cred > h_cred and b_q >= max(h_q, m_q):
        source = 'boundary'
        final_action = b_action
    else:
        source = 'machine'
        final_action = m_action
        
    return final_action, source, q_values_dict

def arbitrator_htma(m_action, h_action, m_cred, h_cred, agent, state):
    """HTMA 仲裁器"""
    with torch.no_grad():
        state_tensor = torch.tensor([state], dtype=torch.float).to(agent.device)
        q_values = agent.q_net(state_tensor).squeeze()
        m_q = q_values[m_action].item()
        h_q = q_values[h_action].item() if h_action is not None else float('-inf')

    q_values_dict = {'m_q': m_q, 'h_q': h_q, 'b_q': float('-inf')}

    if h_action is not None and h_cred >= m_cred and h_q >= m_q:
        source = 'human'
        final_action = h_action
    else:
        source = 'machine'
        final_action = m_action
        
    return final_action, source, q_values_dict

def train(algorithm, env, agent, boundary, args):
    """通用训练函数，根据算法参数控制功能"""
    print(f"\n--- 开始 {algorithm.upper()} 在线训练模式 ---")
    human_controller = HumanController() if algorithm in ['htma', 'htma-b'] else None
    replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
    human_cred_calculator = ActionValueCredibility(agent=agent, window_size=50) if algorithm in ['htma', 'htma-b'] else None
    
    for episode in range(config.NUM_EPISODES):
        state, _ = env.reset(seed=config.SEED + episode)
        episode_return = 0
        done = False
        step_count = 0

        while not done:
            step_count += 1
            human_action = human_controller.get_action() if human_controller else None
            if human_action == "QUIT":
                done = True
                continue
            
            human_credibility = 0.0
            if human_action is not None and human_cred_calculator:
                human_cred_calculator.add_human_action_and_state(state, human_action)
                human_credibility = human_cred_calculator.get_credibility(current_state=state)
            
            agent.epsilon = config.EPSILON 
            machine_action = agent.take_action(state)
            machine_credibility = agent.get_credibility(state)
            boundary_action = boundary.get_action(state, agent) if algorithm == 'htma-b' else None

            if algorithm == 'moa':
                final_action = machine_action
                source = 'machine'
                q_values = {'m_q': agent.get_q_value(state, machine_action), 'h_q': float('-inf'), 'b_q': float('-inf')}
            elif algorithm == 'htma':
                final_action, source, q_values = arbitrator_htma(
                    machine_action, human_action, machine_credibility, human_credibility, agent, state
                )
            elif algorithm == 'htma-b':
                final_action, source, q_values = arbitrator_htma_b(
                    machine_action, human_action, boundary_action, machine_credibility, human_credibility, agent, state
                )
            
            if args.verbose:
                print_step_details(step_count, human_action, machine_action, boundary_action, 
                                   q_values['h_q'], q_values['m_q'], q_values['b_q'],
                                   human_credibility, machine_credibility, final_action, source, algorithm.upper())

            next_state, reward, terminated, truncated, _ = env.step(final_action)
            done = terminated or truncated
            episode_return += reward
            
            replay_buffer.add(state, final_action, reward, next_state, done)
            if replay_buffer.size() > config.BATCH_SIZE:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(config.BATCH_SIZE)
                agent.update({'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns, 'dones': b_d})

            if algorithm == 'htma-b' and boundary:
                value_of_machine_action = q_values['m_q']
                value_of_boundary_action = q_values['b_q']
                if human_action is not None and q_values['h_q'] > value_of_boundary_action:
                    boundary.add(state, human_action)
                elif value_of_machine_action > value_of_boundary_action:
                    boundary.add(state, machine_action)
            
            state = next_state

        if human_action == "QUIT":
            break
        print(f"{algorithm.upper()} 训练 - 第 {episode + 1}/{config.NUM_EPISODES} 轮: 总奖励 = {episode_return:.2f}")

    model_save_path = {
        'moa': config.MOA_MODEL_SAVE_PATH,
        'htma': config.HTMA_MODEL_SAVE_PATH,
        'htma-b': config.HTMAB_AGENT_MODEL_SAVE_PATH
    }[algorithm]
    torch.save(agent.q_net.state_dict(), model_save_path)
    
    if algorithm == 'htma-b' and boundary:
        with open(config.BOUNDARY_KNN_PATH, "wb") as f:
            pickle.dump(boundary, f)
        print(f"\n边界数据已保存至: {config.BOUNDARY_KNN_PATH}")
    
    if human_controller:
        human_controller.close()
    print(f"\n训练结束。\nDQN模型已保存至: {model_save_path}")

def evaluate(algorithm, env, agent, boundary, args):
    """通用评估函数，根据算法参数控制功能"""
    print(f"\n--- 开始 {algorithm.upper()} 评估模式 ---")
    human_controller = HumanController() if algorithm in ['htma', 'htma-b'] else None
    human_cred_calculator = ActionValueCredibility(agent=agent, window_size=50) if algorithm in ['htma', 'htma-b'] else None
    data_handler = DataHandler(save_dir=f"results/{algorithm.upper()}")
    agent.epsilon = 0.0

    for episode in range(config.NUM_EPISODES):
        state, _ = env.reset(seed=config.SEED + episode)
        episode_return, done, last_reward = 0, False, 0
        action_counts = {'human': 0, 'machine': 0, 'boundary': 0} if algorithm == 'htma-b' else (
            {'human': 0, 'machine': 0} if algorithm == 'htma' else {'machine': 0}
        )
        trajectory = []
        step_count = 0

        while not done:
            step_count += 1
            trajectory.append(state[:2])
            human_action = human_controller.get_action() if human_controller else None
            if human_action == "QUIT":
                done = True
                continue
            
            human_credibility = 0.0
            if human_action is not None and human_cred_calculator:
                human_cred_calculator.add_human_action_and_state(state, human_action)
                human_credibility = human_cred_calculator.get_credibility(current_state=state)

            machine_action = agent.get_best_action(state)
            machine_credibility = agent.get_credibility(state)
            boundary_action = boundary.get_action(state, agent) if algorithm == 'htma-b' else None
            
            if algorithm == 'moa':
                final_action = machine_action
                source = 'machine'
                q_values = {'m_q': agent.get_q_value(state, machine_action), 'h_q': float('-inf'), 'b_q': float('-inf')}
            elif algorithm == 'htma':
                final_action, source, q_values = arbitrator_htma(
                    machine_action, human_action, machine_credibility, human_credibility, agent, state
                )
            elif algorithm == 'htma-b':
                final_action, source, q_values = arbitrator_htma_b(
                    machine_action, human_action, boundary_action, machine_credibility, human_credibility, agent, state
                )
            
            if args.verbose:
                print_step_details(step_count, human_action, machine_action, boundary_action, 
                                   q_values['h_q'], q_values['m_q'], q_values['b_q'],
                                   human_credibility, machine_credibility, final_action, source, algorithm.upper())

            action_counts[source] += 1
            
            next_state, reward, terminated, truncated, _ = env.step(final_action)
            done = terminated or truncated
            episode_return += reward
            last_reward = reward
            
            if algorithm == 'htma-b' and boundary:
                value_of_machine_action = q_values['m_q']
                value_of_boundary_action = q_values['b_q']
                if human_action is not None and q_values['h_q'] > value_of_boundary_action:
                    boundary.add(state, human_action)
                elif value_of_machine_action > value_of_boundary_action:
                    boundary.add(state, machine_action)
            
            state = next_state

        if human_action == "QUIT" and human_controller:
            break
        
        is_success = terminated and last_reward == 100
        is_crash = terminated and last_reward == -100
        data_handler.add_episode_data(reward=episode_return, success=is_success, crash=is_crash, 
                                      action_sources=action_counts, trajectory=trajectory)
        
        status_message = "超时" if truncated else ("成功着陆" if is_success else ("坠毁" if is_crash else "未知"))
        total_steps = sum(action_counts.values())
        if total_steps > 0:
            if algorithm == 'moa':
                m_p = (action_counts['machine'] / total_steps) * 100
                action_info = f"动作来源: 机 {m_p:.1f}%"
            elif algorithm == 'htma':
                h_p, m_p = (action_counts['human'] / total_steps) * 100, (action_counts['machine'] / total_steps) * 100
                action_info = f"动作来源: 人 {h_p:.1f}%, 机 {m_p:.1f}%"
            else:  # htma-b
                h_p = (action_counts['human'] / total_steps) * 100
                m_p = (action_counts['machine'] / total_steps) * 100
                b_p = (action_counts['boundary'] / total_steps) * 100
                action_info = f"动作来源: 人 {h_p:.1f}%, 机 {m_p:.1f}%, 界 {b_p:.1f}%"
        else:
            action_info = "无有效动作"
        print(f"{algorithm.upper()} 评估 - 第 {episode + 1}/{config.NUM_EPISODES} 轮 | 状态: {status_message} | 总奖励: {episode_return:.2f} | {action_info}")

    if algorithm == 'htma-b' and boundary:
        with open(config.BOUNDARY_KNN_PATH, "wb") as f:
            pickle.dump(boundary, f)
        print(f"\n边界数据已保存至: {config.BOUNDARY_KNN_PATH}")

    data_handler.save_all_data(f"{algorithm}_results.npz")
    if human_controller:
        human_controller.close()
    print(f"\n{algorithm.upper()} 评估结束。数据已保存。")

def main():
    parser = argparse.ArgumentParser(description="运行 MOA, HTMA 或 HTMA-B 算法")
    parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'], help="选择运行模式")
    parser.add_argument('--verbose', action='store_true', help="开启详细的单步决策信息打印")
    args = parser.parse_args()

    env = gym.make(config.ENV_NAME, render_mode="human")
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

    agent = DoubleDQN(state_dim, action_dim, config.DEVICE, use_mc_dropout=True)
    boundary = None
    
    if config.ALGORITHM == 'htma-b':
        boundary = KnnBoundary(capacity=config.BOUNDARY_CAPACITY)
        boundary_path = config.BOUNDARY_KNN_PATH
        try:
            with open(boundary_path, "rb") as f:
                boundary = pickle.load(f)
            print(f"成功加载已有边界数据: {boundary_path}")
        except FileNotFoundError:
            print("未找到边界数据，将使用新的边界。")
        
        model_path = config.MODEL_SAVE_PATH if args.mode == 'train' else (
            config.HTMAB_AGENT_MODEL_SAVE_PATH if os.path.exists(config.HTMAB_AGENT_MODEL_SAVE_PATH) else config.MODEL_SAVE_PATH
        )
    elif config.ALGORITHM == 'htma':
        model_path = config.MODEL_SAVE_PATH if args.mode == 'train' else (
            config.HTMA_MODEL_SAVE_PATH if os.path.exists(config.HTMA_MODEL_SAVE_PATH) else config.MODEL_SAVE_PATH
        )
    elif config.ALGORITHM == 'moa':
        model_path = config.MODEL_SAVE_PATH if args.mode == 'train' else (
            config.MOA_MODEL_SAVE_PATH if os.path.exists(config.MOA_MODEL_SAVE_PATH) else config.MODEL_SAVE_PATH
        )
    else:
        raise ValueError(f"不支持的算法: {config.ALGORITHM}")

    try:
        agent.q_net.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        agent.target_q_net.load_state_dict(agent.q_net.state_dict())
        print(f"成功加载DQN模型: {model_path}")
    except FileNotFoundError:
        print(f"警告: 找不到DQN模型 {model_path}。将使用随机初始化的DQN。")

    if args.mode == 'train':
        train(config.ALGORITHM, env, agent, boundary, args)
    else:
        evaluate(config.ALGORITHM, env, agent, boundary, args)

    env.close()

if __name__ == "__main__":
    main()