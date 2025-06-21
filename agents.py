# agents.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import config

class ReplayBuffer:
    """ 经验回放池 """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class Q_Net(nn.Module):
    """ Q 网络 """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if self.dropout_rate > 0:
            self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if self.dropout_rate > 0:
            self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.dropout_rate > 0:
            x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        if self.dropout_rate > 0:
            x = self.dropout2(x)
        return self.fc3(x)

class DoubleDQN:
    """ Double DQN 智能体 """
    def __init__(self, state_dim, action_dim, device, use_mc_dropout=False):
        self.action_dim = action_dim
        self.device = device
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON
        self.tau = config.TAU
        
        dropout_rate = config.DROPOUT_RATE if use_mc_dropout else 0.0

        self.q_net = Q_Net(state_dim, config.HIDDEN_DIM, action_dim, dropout_rate).to(self.device)
        self.target_q_net = Q_Net(state_dim, config.HIDDEN_DIM, action_dim, dropout_rate).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.LR)
        self.rng = np.random.RandomState(config.SEED)

    def take_action(self, state):
        """ 使用 epsilon-greedy 策略选择动作 """
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.action_dim)
        else:
            return self.get_best_action(state)

    def get_best_action(self, state):
        """ 获取当前策略下的最优动作 """
        state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
        original_mode = self.q_net.training
        try:
            self.q_net.eval()
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
        finally:
            self.q_net.train(original_mode)
        return action
    
    def get_q_value(self, state, action: int) -> float:
        """ 提供一个正式、清晰的Q值查询接口 """
        original_mode = self.q_net.training
        try:
            self.q_net.eval()
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
                q_values_all = self.q_net(state_tensor).squeeze()
                if q_values_all.dim() == 0:
                    q_values_all = q_values_all.unsqueeze(0)
                action_q_value = q_values_all[action].item()
        finally:
            self.q_net.train(original_mode)
        return action_q_value
    
    def get_credibility(self, state, num_samples=50):
        """ 计算机器行为可信度，基于Q值方差，与文章公式一致 """
        state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
        original_mode = self.q_net.training
        try:
            self.q_net.train()  # 保持 Dropout 开启
            with torch.no_grad():
                q_values_samples = [self.q_net(state_tensor).cpu().numpy() for _ in range(num_samples)]
        finally:
            self.q_net.train(original_mode)
        
        q_values_samples = np.array(q_values_samples).squeeze()  # 形状: (num_samples, action_dim)
        
        # 计算期望 E[a_m(t)]
        E_am = np.mean(q_values_samples, axis=0)  # 形状: (action_dim,)
        
        # 计算二阶矩 E[(a_m(t))^T(a_m(t))]
        action_dim = q_values_samples.shape[1]
        tau = config.HUMAN_NOISE_TAU
        E_am_T_am = tau**-1 * np.eye(action_dim)
        for sample in q_values_samples:
            E_am_T_am += np.outer(sample, sample) / num_samples
        
        # 计算方差 c_m(t)
        c_m = E_am_T_am - np.outer(E_am, E_am)
        c_m_trace = np.trace(c_m)  # 标量方差
        
        # 转换为可信度
        k = 0.15
        # credibility = np.exp(-k * c_m_trace)
        credibility = 1 / (1 +  0.1 * c_m_trace)  # 使用更平滑的转换
        # 调试信息
        # print(f"机器动作方差: {c_m_trace:.4f}, 可信度: {credibility:.4f}")
        
        return credibility
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()
        
        q_values = self.q_net(states).gather(dim=1, index=actions).squeeze()
        max_actions_index = self.q_net(next_states).max(axis=1)[1]
        max_next_q_values = self.target_q_net(next_states).gather(dim=1, index=max_actions_index.unsqueeze(1)).squeeze()
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        self.optimizer.zero_grad()                                                         
        dqn_loss.backward() 
        self.optimizer.step()
        
        for target_param, q_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

class ActionValueCredibility:
    """ 使用相对Q值或历史动作数据计算人类可信度 """
    def __init__(self, agent, window_size=100, min_samples=1):
        self.relative_q_values_history = deque(maxlen=window_size)
        self.agent = agent
        self.min_samples = min_samples
        self.history_buffer = deque(maxlen=config.HUMAN_HISTORY_CAPACITY)
        self.rng = np.random.RandomState(config.SEED)

    def add_human_action_and_state(self, state, human_action):
        """ 添加人类动作和状态到历史记录 """
        original_mode = self.agent.q_net.training
        try:
            self.agent.q_net.eval()
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float).to(self.agent.device)
                q_values = self.agent.q_net(state_tensor).squeeze()
                action_q_value = q_values[human_action].item()
        finally:
            self.agent.q_net.train(original_mode)
        
        self.relative_q_values_history.append(action_q_value)
        self.history_buffer.append({'state': np.array(state), 'action': human_action})

    def get_credibility(self, current_state=None):
        """ 基于历史动作的Q值方差计算可信度 """
        if len(self.history_buffer) < self.min_samples:
            return 0.5
        # 筛选相似状态的历史动作
        current_state = np.array(current_state)
        similar_actions = []
        for entry in self.history_buffer:
            state = entry['state']
            distance = np.linalg.norm(current_state - state)
            if distance < config.HUMAN_STATE_SIMILARITY_THRESHOLD:
                similar_actions.append(entry['action'])
        # 采样 T 次历史动作
        sampled_actions = self.rng.choice(similar_actions, size=config.HUMAN_SAMPLING_NUM, replace=True)
        # 计算每个采样动作的 Q 值
        q_values_samples = []
        original_mode = self.agent.q_net.training
        try:
            self.agent.q_net.eval()  # 使用评估模式，确保无 Dropout 影响
            with torch.no_grad():
                state_tensor = torch.tensor([current_state], dtype=torch.float).to(self.agent.device)
                q_values_all = self.agent.q_net(state_tensor).squeeze()  # 计算一次 Q 值向量
                for action in sampled_actions:
                    q_value = q_values_all[action].item()  # 提取对应动作的 Q 值
                    q_values_samples.append(q_value)
        finally:
            self.agent.q_net.train(original_mode)
        q_values_samples = np.array(q_values_samples)  # 形状: (num_samples,)
        # 计算期望 E[q_h(t)]
        E_qh = np.mean(q_values_samples)
        # 计算二阶矩 E[q_h(t)^2]
        tau = config.HUMAN_NOISE_TAU
        E_qh2 = tau**-1 + np.mean(q_values_samples**2)
        # 计算方差 c_h(t)
        c_h = E_qh2 - E_qh**2
        
        # 转换为可信度
        k = 0.05  
        # credibility = np.exp(-k * c_h)
        credibility = 1 / (1 +  0.3 * c_h)  # 使用更平滑的转换
        return credibility
    
class KnnBoundary:
    """ 基于k-NN的自主边界实现 """
    def __init__(self, capacity=config.BOUNDARY_CAPACITY):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
        self.default_action = config.ACTION_DO_NOTHING

    def add(self, state, action):
        """ add方法，添加 (状态, 动作) 对 """
        if len(self.buffer) == self.capacity:
            self.buffer.popleft()
        self.buffer.append({'state': state, 'action': action})

    def get_action(self, current_state, agent: DoubleDQN, k=50) -> int:
        """ 依赖外部agent进行实时价值评估 """
        if not self.buffer:
            return self.default_action
        
        k = min(k, len(self.buffer))
        
        stored_states = np.array([entry['state'] for entry in self.buffer])
        distances = np.linalg.norm(stored_states - current_state, axis=1)
        nearest_indices = np.argpartition(distances, k-1)[:k]
        
        candidate_actions = [self.buffer[idx]['action'] for idx in nearest_indices]
        
        best_action = self.default_action
        max_q_value = -float('inf')
        
        for action in set(candidate_actions):
            q_value = agent.get_q_value(current_state, action)
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action
                
        return best_action

    def __len__(self):
        return len(self.buffer)