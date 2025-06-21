本仓库基于"Traded Control of Human–Machine Systems for  Sequential Decision-Making Based on  Reinforcement Learning"论文综合实现HOA/MOA/HTMA/MTHA/HTMA-B/MTHA-B算法
好的，这是一个根据您提供的代码文件和项目背景编写的、结构完整且风格专业的 `README.md` 文件。

-----

# 基于强化学习的序贯决策人机切换控制

本项目是论文 **"Traded Control of Human–Machine Systems for Sequential Decision-Making Based on Reinforcement Learning" (Q. Zhang et al., 2022)** 的一个代码实现与拓展。项目不仅复现了论文中提出的 机器切换人类控制（MTHA）和人类切换机器控制（HTMA）两大核心框架，还对其底层算法进行了优化，并实现了包括自主边界（Autonomous Boundary）和可信度评估（Credibility Assessment）在内的关键机制。

与原论文不同，本实现采用 **Double DQN** 算法代替标准DQN，旨在缓解Q值过高估计问题，提升训练的稳定性与策略的可靠性。

## 核心特性

  * **切换控制框架**：完整实现了MTHA与HTMA两种人机协作模式。
  * **自主边界实现**：通过高效的 **k-近邻（k-NN）算法**，对论文中提出的“自主边界”概念进行了数据驱动的实现，使其能够在运行时提供高质量的备选决策。
  * **可信度评估**：基于 **蒙特卡洛 Dropout (MC Dropout)** 技术来近似贝叶斯神经网络，从而量化机器决策的不确定性；同时，通过分析历史行为数据来评估人类决策的可信度。
  * **算法优化**：采用 **Double DQN** 算法作为强化学习智能体的核心，以获得更精确的价值评估。
  * **人类在环交互**：通过 `Pygame` 模块实现了流畅的人类玩家实时输入，支持人类玩家在训练和评估中随时介入。
  * **模块化设计**：代码结构清晰，将智能体、配置文件、数据处理、可视化等功能解耦，易于理解和拓展。

## 项目结构

```
.
├── agents.py               # 定义强化学习智能体 (DoubleDQN)、经验回放池、自主边界 (KnnBoundary) 等核心类
├── config.py               # 存储所有超参数、路径、算法选择等全局配置
├── data_handler.py         # 用于在评估后收集和保存实验数据 (如奖励、成功率、轨迹等)
├── human_controller.py     # 处理人类玩家的键盘输入
├── main_mtha_b.py          # MTHA (机器切换人类) 场景的训练与评估主程序
├── main_htma_b.py          # HTMA (人类切换机器) 场景的训练与评估主程序
├── plotter.py              # 用于根据保存的数据绘制结果图表
├── model/                    # 存放训练好的模型文件
└── results/                  # 存放评估数据和生成的图表
```

## 环境配置

1.  克隆本仓库到本地：

    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  建议创建一个独立的Python虚拟环境（例如使用 `conda` 或 `venv`）。

3.  安装所需的依赖库。创建一个名为 `requirements.txt` 的文件，内容如下：

    ```
    gymnasium
    torch
    numpy
    pygame
    matplotlib
    pandas
    ```

    然后运行安装命令：

    ```bash
    pip install -r requirements.txt
    ```

## 使用指南

### 1\. 算法配置

在运行程序之前，请先修改 `config.py` 文件中的 `ALGORITHM` 变量，以选择您希望运行的算法模式。

  * **MTHA 场景** (运行 `main_mtha_b.py`):

      * `'hoa'`: 纯人类操作 (Human-Only Algorithm)。
      * `'mtha'`: 机器切换人类控制。
      * `'mtha-b'`: 机器切换人类控制，并启用自主边界。

  * **HTMA 场景** (运行 `main_htma_b.py`):

      * `'moa'`: 纯机器操作 (Machine-Only Algorithm)。
      * `'htma'`: 人类切换机器控制。
      * `'htma-b'`: 人类切换机器控制，并启用自主边界。

### 2\. 模型训练

要训练一个新的智能体模型，请使用 `--mode train` 参数。

**示例：训练 HTMA-B 模型**

1.  在 `config.py` 中设置 `ALGORITHM = 'htma-b'`。
2.  运行 `main_htma_b.py`：
    ```bash
    python main_htma_b.py --mode train
    ```
    训练过程中，人类玩家可以随时通过键盘（方向键）进行干预。训练完成后，模型和边界数据将自动保存到 `model/` 文件夹中。

### 3\. 模型评估

要评估已训练好的模型，请使用 `--mode eval` 参数。评估模式将关闭随机探索（epsilon=0），并记录详细的实验数据。

**示例：评估 MTHA-B 模型**

1.  确保 `model/` 文件夹中存在对应的预训练模型 (`dqn_lunarlander_mthab_finetuned.pth`) 和边界数据 (`mthab_boundary.pkl`)。
2.  在 `config.py` 中设置 `ALGORITHM = 'mtha-b'`。
3.  运行 `main_mtha_b.py`：
    ```bash
    python main_mtha_b.py --mode eval
    ```
    评估结束后，原始数据（`.npz` 文件）将保存到 `results/MTHA-B/` 文件夹下。

### 4\. 详细日志

在训练或评估时，可以添加 `--verbose` 标志来打印每一步详细的决策信息，包括人类、机器和边界的Q值、可信度以及最终的仲裁结果。

```bash
python main_htma_b.py --mode eval --verbose
```

### 5\. 结果可视化

评估数据保存后，可以运行 `plotter.py` 脚本来生成论文风格的对比图表。请确保 `plotter.py` 中的文件加载路径正确。

## 许可协议

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源许可协议。

## 致谢

本项目的实现与思想源于以下研究工作，特此致谢：

> Q. Zhang, Y. Kang, Y. -B. Zhao, P. Li and S. You, "Traded Control of Human-Machine Systems for Sequential Decision-Making Based on Reinforcement Learning," in IEEE Transactions on Artificial Intelligence, vol. 3, no. 4, pp. 553-565, Aug. 2022, doi: 10.1109/TAI.2021.3127857.
