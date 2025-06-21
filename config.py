# config.py
import torch
import pygame

# ==================================
# 算法选择
# ==================================
ALGORITHM = 'htma'  # 选择算法：'mtha-b', 'mtha', 'hoa', 'htma', 'htma-b', 'moa'

# ==================================
# 核心训练超参数
# ==================================
LR = 2e-3
NUM_EPISODES = 100
HIDDEN_DIM = 128
GAMMA = 0.98
EPSILON = 0.01
TAU = 0.005
BUFFER_SIZE = 50000
BATCH_SIZE = 64
DROPOUT_RATE = 0.001

# ==================================
# 神经网络边界相关配置
# ==================================
BOUNDARY_LEARNING_RATE = 1e-3
BOUNDARY_TRAINING_START_STEP = 10000
BOUNDARY_CAPACITY = 100000
BOUNDARY_UPDATE_FREQUENCY = 5
BOUNDARY_KNN_K = 10

# ==================================
# 人类行为可信度计算配置
# ==================================
HUMAN_HISTORY_CAPACITY = 10000
HUMAN_SAMPLING_NUM = 50
HUMAN_STATE_SIMILARITY_THRESHOLD = 0.8
HUMAN_CREDIBILITY_METHOD = 'history'
HUMAN_NOISE_TAU = 1.0

# ==================================
# 环境和设备配置
# ==================================
ENV_NAME = "LunarLander-v3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# ==================================
# HOA 配置
# ==================================
KEY_MAP = {
    pygame.K_LEFT: 1,
    pygame.K_DOWN: 2,
    pygame.K_RIGHT: 3,
}
ACTION_DO_NOTHING = 0

# ==================================
# 图像、模型保存路径
# ==================================
MODEL_SAVE_PATH = "model/dqn_lunarlander.pth"
MTHA_MODEL_SAVE_PATH = "model/dqn_lunarlander_mtha_finetuned.pth"
MOA_MODEL_SAVE_PATH = "model/dqn_lunarlander_moa_finetuned.pth"
MTHAB_MODEL_SAVE_PATH = "model/dqn_lunarlander_mthab_finetuned.pth"
HTMA_MODEL_SAVE_PATH = "model/dqn_lunarlander_htma_finetuned.pth"
HTMAB_AGENT_MODEL_SAVE_PATH = "model/dqn_lunarlander_htmab_agent_finetuned.pth"
BOUNDARY_NET_SAVE_PATH = "model/boundary_net.pth"
BOUNDARY_KNN_PATH = "model/knn_boundary.pkl"
MTHAB_BOUNDARY_PATH = "model/mthab_boundary.pkl" 
RESULTS_DIR = "plot_results"