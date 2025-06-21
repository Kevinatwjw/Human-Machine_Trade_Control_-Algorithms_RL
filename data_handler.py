# data_handler.py
import json
import numpy as np
import os

class DataHandler:
    """
    一个专门用于处理和保存实验数据的类。
    它负责收集每一轮的数据，并在实验结束后统一保存。
    """
    def __init__(self, save_dir="dataresults"):
        """
        初始化数据处理器。

        Args:
            save_dir (str): 保存结果的文件夹名称。
        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"创建文件夹: {self.save_dir}")
            
        # 初始化用于存储所有数据的列表
        self.all_rewards = []
        self.all_successes = []
        self.all_crashes = []
        self.all_action_sources = {'machine': [], 'human': [], 'boundary': []}
        self.all_trajectories = [] # 将存储多条轨迹

    def add_episode_data(self, reward, success, crash, action_sources, trajectory):
        """
        在每一轮结束后，添加该轮的数据。

        Args:
            reward (float): 本轮的总奖励。
            success (bool): 本轮是否成功。
            crash (bool): 本轮是否坠毁。
            action_sources (dict): 包含 'machine', 'human', 'boundary' 计数的字典。
            trajectory (list): 包含本轮所有(x, y)坐标的列表。
        """
        self.all_rewards.append(reward)
        self.all_successes.append(success)
        self.all_crashes.append(crash)
        self.all_trajectories.append(np.array(trajectory)) # 将轨迹转为numpy数组
        
        for key in self.all_action_sources:
            self.all_action_sources[key].append(action_sources.get(key, 0))

    def save_all_data(self, filename):
        """
        将收集到的所有数据保存到文件中。
        我们将使用 .npz 格式，它可以高效地保存多个numpy数组。

        Args:
            filename (str): 保存数据的文件名 (例如 'mtha_b_results.npz')。
        """
        filepath = os.path.join(self.save_dir, filename)
        
        # 将非数组数据保存为JSON字符串，以便与npz兼容
        # 我们需要一个字典来存储除了轨迹之外的所有数据
        metadata = {
            'rewards': self.all_rewards,
            'successes': self.all_successes,
            'crashes': self.all_crashes,
            'action_sources': self.all_action_sources
        }
        
        # 使用np.savez保存，它能处理不同长度的数组
        # 我们将轨迹和其他数据分开保存
        np.savez(filepath, 
                 trajectories=np.array(self.all_trajectories, dtype=object), # 保存轨迹
                 metadata=np.array([json.dumps(metadata)])) # 保存其他数据为json字符串

        print(f"所有实验数据已成功保存至: {filepath}")