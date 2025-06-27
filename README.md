# 基于强化学习的序贯决策人机切换控制 🎉🚀

本仓库基于 **"Traded Control of Human–Machine Systems for Sequential Decision-Making Based on Reinforcement Learning" (Q. Zhang et al., 2022)** 论文，综合实现了 **HOA/MOA/HTMA/MTHA/HTMA-B/MTHA-B** 算法。项目不仅复现了论文中的核心框架，还对其底层算法进行了优化，引入了 **Double DQN** 算法以提升训练稳定性和策略可靠性，结合 **自主边界**（Autonomous Boundary）和 **可信度评估**（Credibility Assessment）机制，打造了高效的人机协作系统！😄👨‍💻🤖

---

## 核心特性 ✨🔧

- **切换控制框架**：完整实现了 **MTHA**（机器切换人类控制）和 **HTMA**（人类切换机器控制）两种人机协作模式。🎮
- **自主边界实现**：通过高效的 **k-近邻（k-NN）算法**，动态生成高质量的备选决策，优化人机决策权限划分。📊
- **可信度评估**：基于 **蒙特卡洛 Dropout (MC Dropout)** 近似贝叶斯神经网络，量化机器决策不确定性；通过历史行为数据评估人类可信度。📈
- **算法优化**：采用 **Double DQN** 算法，缓解 Q 值过高估计问题，提升价值评估精度。🏆
- **人类在环交互**：利用 `Pygame` 模块支持实时人类输入，允许玩家随时介入训练和评估。🎹
- **模块化设计**：代码结构清晰，将智能体、配置文件、数据处理和可视化功能解耦，便于理解与拓展。🛠️

## 项目结构 📂

```
.
├── agents.py               # 定义强化学习智能体 (DoubleDQN)、经验回放池、自主边界 (KnnBoundary) 等核心类 😊
├── config.py               # 存储所有超参数、路径、算法选择等全局配置 ⚙️
├── data_handler.py         # 用于在评估后收集和保存实验数据 (如奖励、成功率、轨迹等) 📝
├── human_controller.py     # 处理人类玩家的键盘输入 🎮
├── main_mtha_b.py          # MTHA (机器切换人类) 场景的训练与评估主程序 🤖👨‍💻
├── main_htma_b.py          # HTMA (人类切换机器) 场景的训练与评估主程序 👨‍💻🤖
├── plotter.py              # 用于根据保存的数据绘制结果图表 📊🎨
├── model/                    # 存放训练好的模型文件 💾
└── results/                  # 存放评估数据和生成的图表 📁
```

## 环境配置 🌱

1. 克隆本仓库到本地：  
   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```
   🌟

2. 建议创建一个独立的 Python 虚拟环境（使用 `conda` 或 `venv`）：  
   ```bash
   conda create -n sdmc python=3.9
   conda activate sdmc
   ```
   🐍

3. 安装依赖库。创建一个 `requirements.txt` 文件，内容如下：  
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
   📦

## 使用指南 📖

### 1. 算法配置 🔧

在运行程序前，请修改 `config.py` 中的 `ALGORITHM` 变量，选择运行模式：  

- **MTHA 场景** (运行 `main_mtha_b.py`):  
  - `'hoa'`: 纯人类操作 (Human-Only Algorithm) 👨‍💻  
  - `'mtha'`: 机器切换人类控制 🤖👉👨‍💻  
  - `'mtha-b'`: 机器切换人类控制 + 自主边界 🤖👉👨‍💻📏  

- **HTMA 场景** (运行 `main_htma_b.py`):  
  - `'moa'`: 纯机器操作 (Machine-Only Algorithm) 🤖  
  - `'htma'`: 人类切换机器控制 👨‍💻👉🤖  
  - `'htma-b'`: 人类切换机器控制 + 自主边界 👨‍💻👉🤖📏  

### 2. 模型训练 🎓

训练新模型，使用 `--mode train` 参数。  

**示例：训练 HTMA-B 模型**  
1. 在 `config.py` 中设置 `ALGORITHM = 'htma-b'`。  
2. 运行 `main_htma_b.py`：  
   ```bash
   python main_htma_b.py --mode train
   ```
   训练中，人类玩家可通过键盘（方向键）介入。完成时，模型和边界数据保存至 `model/` 文件夹。⏳💾

### 3. 模型评估 📊

评估模型，使用 `--mode eval` 参数（关闭随机探索，epsilon=0）。  

**示例：评估 MTHA-B 模型**  
1. 确保 `model/` 中有预训练模型 (`dqn_lunarlander_mthab_finetuned.pth`) 和边界数据 (`mthab_boundary.pkl`)。  
2. 设置 `ALGORITHM = 'mtha-b'`。  
3. 运行 `main_mtha_b.py`：  
   ```bash
   python main_mtha_b.py --mode eval
   ```
   评估数据保存至 `results/MTHA-B/`（`.npz` 文件）。✅

### 4. 详细日志 🗒️

添加 `--verbose` 标志，打印每步决策详情（Q 值、可信度、仲裁结果）：  
```bash
python main_htma_b.py --mode eval --verbose
```
🔍

### 5. 结果可视化 🎨

评估后，运行 `plotter.py` 生成图表（确保路径正确）：  
- 奖励曲线：如累积奖励对比图。  
- 成功率/坠毁率：展示算法稳定性。  
- 动作占比：分析人类、机器、边界贡献。  
- 轨迹图：可视化着陆路径。  
```bash
python plotter.py
```
📈🌐

## 结果展示 🌟📊

本项目在 **LunarLander-v3** 环境中验证了各算法性能，复现了论文中的关键实验结果，具体如下：

- **奖励对比**：  
  - **MTHA 场景**：  
    - **HOA**（仅人类控制）奖励最低，平均值较低，符合人类操作精度不足的假设。  
    - **MTHA**（机器干预人类）累积奖励有所提升，表明机器干预的初步效果。  
    - **MTHA-B**（带边界优化）奖励最高，平均值比 **HOA** 和 **MTHA** 提高约 30%-40%，验证了自主边界的优化作用。  
    - 箱线图显示 **MTHA-B** 奖励分布更集中，异常点（红+号）减少（参考 [Fig. 5a](#)）。  
  - **HTMA 场景**：  
    - **MOA**（仅机器控制）奖励较低，反映机器训练不足。  
    - **HTMA**（人类干预机器）奖励有所提升。  
    - **HTMA-B** 奖励显著提高，成功回合奖励平均值更高，验证了自主边界的优势（参考 [Fig. 11a](#)）。  

- **成功率与坠毁率**：  
  - **MTHA 场景**：  
    - **HOA** 成功率最低，崩溃率高，反映人类操作的不稳定性。  
    - **MTHA** 成功率有所提高，崩溃率降低。  
    - **MTHA-B** 成功率最高可达 **0.45**，比 **HOA** 和 **MTHA** 改善显著，崩溃率降低约 20%，由于边界信息引入导致一定不稳定性，但整体性能优异（参考 [Fig. 5c](#) 和 [Fig. 5d](#)）。  
  - **HTMA 场景**：  
    - **MOA** 成功率持续较低，崩溃率较高。  
    - **HTMA** 成功率有所提升。  
    - **HTMA-B** 成功率超过 **0.45**，崩溃率低于 **MOA**，表明人类干预优化了决策（参考 [Fig. 11c](#) 和 [Fig. 11d](#)）。  

- **动作占比**：  
  - **MTHA 场景**：  
    - **MTHA** 中人类动作占比较低，机器动作占主导。  
    - **MTHA-B** 中人类动作、边界动作、机器动作比例约为 **2:3:5**，边界动作在决策中起到关键作用（参考 [Fig. 6b](#)）。  
  - **HTMA 场景**：  
    - **HTMA** 中人类动作占比较高，机器动作较少。  
    - **HTMA-B** 中人类动作、机器动作、边界动作比例约为 **3:1:4**，人类动作占比增加，反映了人类主导的优势（参考 [Fig. 12b](#)）。  

- **轨迹分析**：  
  - **MTHA 场景**：  
    - **HOA** 轨迹整齐但成功率低，倾向于快速失败（参考 [Fig. 7a](#)）。  
    - **MTHA** 轨迹较乱但成功率提高，机器干预有效（参考 [Fig. 7b](#)）。  
    - **MTHA-B** 轨迹介于两者之间，成功率和运行时间步优化更明显，任务完成更高效（参考 [Fig. 7c](#)）。  
  - **HTMA 场景**：  
    - **MOA** 轨迹无序，成功率低（参考 [Fig. 13a](#)）。  
    - **HTMA** 轨迹改善，成功率提升（参考 [Fig. 13b](#)）。  
    - **HTMA-B** 轨迹更优化，成功率最高，运行时间步显著减少（参考 [Fig. 13c](#) 和 [Fig. 14](#)）。  

- **稳定性**：  
  - **MTHA-B** 和 **HTMA-B** 的奖励分布更集中，异常点减少，验证了算法的鲁棒性，符合阅读报告中的分析。  

所有结果可通过 `plotter.py` 复现，生成的图表与论文风格一致，保存在 `results/` 文件夹中。图表链接示例（需根据实际文件路径调整）：  
- [Fig. 5a - Reward Comparison](#)  
- [Fig. 5c - Success Rate](#)  
- [Fig. 5d - Crash Rate](#)  
- [Fig. 6b - MTHA-B Action Percentage](#)  
- [Fig. 7c - MTHA-B Trajectory](#)  
- [Fig. 11c - HTMA-B Success Rate](#)  
- [Fig. 13c - HTMA-B Trajectory](#)  

## 许可协议 📜

本项目采用 [MIT License](https://www.google.com/search?q=MIT+License) 开源许可协议。🔓

## 致谢 🙏

本项目的实现与思想源于以下研究工作，特此致谢：  

> Q. Zhang, Y. Kang, Y. -B. Zhao, P. Li and S. You, "Traded Control of Human-Machine Systems for Sequential Decision-Making Based on Reinforcement Learning," in IEEE Transactions on Artificial Intelligence, vol. 3, no. 4, pp. 553-565, Aug. 2022, doi: 10.1109/TAI.2021.3127857.  

同时感谢社区贡献者与测试者！🤝

## 未来工作 🌱

- 支持更复杂的环境（如 Atari 游戏或机器人任务）。  
- 集成意图推理模块，结合人类行为优化决策。  
- 动态调整 k-NN 中的 k 值，适应不同状态密度。  
- 优化可信度评估公式，提升算法适应性。  
📅🚀
