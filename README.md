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
   git clone https://github.com/Kevinatwjw/Human-Machine_Trade_Control_-Algorithms_RL.git
   cd Human-Machine_Trade_Control_-Algorithms_RL
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

我们在 **LunarLander-v3** 仿真环境中，对项目中的各个算法进行了系统性的“大比武”！⚔️ 实验首先对作为机器智能体的 Double DQN 模型进行了500轮的预训练，以确保它身经百战、具备可靠的基准决策能力。随后，在100轮的正式评估中，我们收集到了以下激动人心的关键结果，完美复现并验证了论文的核心思想。🚀

### 1\. 累积奖励对比 🏆

- ![Reward Comparison](results/Reward%20Comparison.png)

累积奖励的分布情况清晰地揭示了各算法的“武力值” 💪，优秀的算法能够获得更高的分数！

  * **在机器交易人类 (MTHA) 场景中**：
      * **HOA (纯人类操作) 👨‍💻** 的中位数奖励位于负值区间，表现不尽人意。
      * [cite\_start]引入机器干预的 **MTHA** 与 **MTHA-B 🤖** 算法的奖励表现均显著优于 HOA，奖励中位数从负值大幅跃升至 **约250分** 的较高水平 [cite: 421]，证明了人机协作框架的强大威力！
  * **在人类交易机器 (HTMA) 场景中**：
      * **HTMA-B (引入自主边界) 🧠** 算法的奖励曲线和分布同样全面优于基准的 **MOA (纯机器操作)** 与 **HTMA**。
      * [cite\_start]HTMA-B 的中位数奖励提升至 **约270分**，且其奖励分布系统性地高于其他算法，表明其任务完成质量最优 [cite: 421]。

### 2\. 成功率与飞行轨迹 🛰️ 

- ![Trajectory with Success Rate](results/Trajectory%20with%20Success%20Rate.jpg)
 
任务成功率与飞行轨迹的分析进一步印证了奖励对比的结论，展示了算法的智慧与稳定。

  * [cite\_start]**HOA** 的成功率仅为 **4%**，其飞行轨迹散乱，像个没头苍蝇，难以有效完成着陆任务 [cite: 421]。
  * [cite\_start]引入机器干预后，**MTHA** 与 **MTHA-B** 算法的成功率均大幅提升至 **73%**，轨迹也表现出高度的目标导向性和稳定性，直奔目标！[cite: 421]
  * [cite\_start]在HTMA场景下，相较于 **MOA** 的 **66%** 成功率，**HTMA** 将其微弱提升至 **67%** [cite: 421]。
  * [cite\_start]然而，引入自主边界的 **HTMA-B** 算法取得了 **高达89%的成功率** 🎉，其飞行轨迹在所有算法中也最为平稳丝滑，展示了其在任务执行上的卓越性能 [cite: 421]。
  * [cite\_start]**值得一提的是**，通过将原论文的DQN优化为 **Double DQN**，系统整体的稳定性得到了较大提升 [cite: 419]，这也是HTMA-B能取得如此优异表现的关键因素之一。

### 3\. 决策来源分析 🤝 

- ![Action Rate](results/Action%20Rate.png)

决策权归谁？这个分析让我们能深入理解人、机、边界三者是如何协同工作的。

  * **在MTHA场景中**：
      * [cite\_start]性能的提升源于决策权的有效转移。在 **MTHA** 框架下，机器的高精度决策在绝大多数时间里占据主导，有效弥补了人类操作的不足 [cite: 422]。
      * [cite\_start]引入自主边界的 **MTHA-B** 算法在此基础上进一步优化，边界动作的出现替代了部分人类的次优操作，实现了更精细的决策 [cite: 422]。
  * **在HTMA场景中**：
      * **HTMA-B** 呈现出一种更为高效的 **“人-机-边界”三方协同模式** 👨‍💻🤖🧠。
      * [cite\_start]在该模式下，自主边界通过不断学习并融合人类与机器的历史最优经验，形成了一个可靠的第三方决策源，显著提升了机器在人类主导下的表现，是达成高成功率的核心 [cite: 422]。  

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
