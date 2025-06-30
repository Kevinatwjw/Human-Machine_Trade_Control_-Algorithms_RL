# Reinforcement Learning-Based Sequential Decision-Making for Human-Machine Traded Control 🎉🚀

This repository is based on the paper **"Traded Control of Human–Machine Systems for Sequential Decision-Making Based on Reinforcement Learning" (Q. Zhang et al., 2022)** and comprehensively implements the **HOA/MOA/HTMA/MTHA/HTMA-B/MTHA-B** algorithms. The project not only reproduces the core framework from the paper but also optimizes its underlying algorithms by integrating the **Double DQN** algorithm to enhance training stability and policy reliability. Combined with mechanisms for **Autonomous Boundary** and **Credibility Assessment**, it creates an efficient human-machine collaboration system! 😄👨‍💻🤖

---

## Key Features ✨🔧

- **Traded Control Framework**: Fully implements **MTHA** (Machine-Traded Human Control) and **HTMA** (Human-Traded Machine Control) for human-machine collaboration. 🎮
- **Autonomous Boundary**: Utilizes the efficient **k-Nearest Neighbors (k-NN) algorithm** to dynamically generate high-quality alternative decisions, optimizing the allocation of decision authority. 📏
- **Credibility Assessment**: Quantifies machine decision uncertainty using **Monte Carlo Dropout (MC Dropout)** for approximate Bayesian neural networks and evaluates human credibility based on historical behavior data. 📈
- **Algorithm Optimization**: Employs **Double DQN** to mitigate Q-value overestimation, improving value estimation accuracy. 🏆
- **Human-in-the-Loop Interaction**: Supports real-time human input via the `Pygame` module, allowing players to intervene during training and evaluation. 🎹
- **Modular Design**: Features a clear code structure, decoupling agents, configurations, data handling, and visualization for easy understanding and extensibility. 🛠️

## Project Structure 📂

```
.
├── agents.py               # Defines RL agents (DoubleDQN), experience replay buffer, and autonomous boundary (KnnBoundary) classes 😊
├── config.py               # Stores global configurations like hyperparameters, paths, and algorithm selection ⚙️
├── data_handler.py         # Handles collection and storage of experimental data (rewards, success rates, trajectories, etc.) 📝
├── human_controller.py     # Manages keyboard inputs from human players 🎮
├── main_mtha_b.py          # Main script for training and evaluating MTHA (Machine-Traded Human) scenarios 🤖👨‍💻
├── main_htma_b.py          # Main script for training and evaluating HTMA (Human-Traded Machine) scenarios 👨‍💻🤖
├── plotter.py              # Generates result visualizations based on saved data 📊🎨
├── model/                  # Stores trained model files 💾
└── results/                # Stores evaluation data and generated plots 📁
```

## Environment Setup 🌱

1. Clone the repository to your local machine:  
   ```bash
   git clone https://github.com/Kevinatwjw/Human-Machine_Trade_Control_-Algorithms_RL.git
   cd Human-Machine_Trade_Control_-Algorithms_RL
   ```
   🌟

2. Create a dedicated Python virtual environment (using `conda` or `venv`):  
   ```bash
   conda create -n sdmc python=3.9
   conda activate sdmc
   ```
   🐍

3. Install dependencies. Create a `requirements.txt` file with the following content:  
   ```
   gymnasium>=0.28.1
   torch>=2.0.0
   numpy>=1.21.0
   pygame>=2.5.0
   matplotlib>=3.5.0
   pandas>=1.3.0
   ```
   Then run:  
   ```bash
   pip install -r requirements.txt
   ```
   📦

## Usage Guide 📖

### 1. Algorithm Configuration 🔧

Before running the program, modify the `ALGORITHM` variable in `config.py` to select the operating mode:

- **MTHA Scenario** (run `main_mtha_b.py`):  
  - `'hoa'`: Human-Only Algorithm 👨‍💻  
  - `'mtha'`: Machine-Traded Human Control 🤖👉👨‍💻  
  - `'mtha-b'`: Machine-Traded Human Control + Autonomous Boundary 🤖👉👨‍💻📏  

- **HTMA Scenario** (run `main_htma_b.py`):  
  - `'moa'`: Machine-Only Algorithm 🤖  
  - `'htma'`: Human-Traded Machine Control 👨‍💻👉🤖  
  - `'htma-b'`: Human-Traded Machine Control + Autonomous Boundary 👨‍💻👉🤖📏  

### 2. Model Training 🎓

Train a new model using the `--mode train` parameter.  

**Example: Train HTMA-B Model**  
1. Set `ALGORITHM = 'htma-b'` in `config.py`.  
2. Run `main_htma_b.py`:  
   ```bash
   python main_htma_b.py --mode train
   ```
   During training, human players can intervene using keyboard inputs (arrow keys). Upon completion, the model and boundary data are saved to the `model/` folder. ⏳💾

### 3. Model Evaluation 📊

Evaluate the model using the `--mode eval` parameter (disables random exploration, epsilon=0).  

**Example: Evaluate MTHA-B Model**  
1. Ensure pretrained models (`dqn_lunarlander_mthab_finetuned.pth`) and boundary data (`mthab_boundary.pkl`) are in the `model/` folder.  
2. Set `ALGORITHM = 'mtha-b'`.  
3. Run `main_mtha_b.py`:  
   ```bash
   python main_mtha_b.py --mode eval
   ```
   Evaluation data is saved to `results/MTHA-B/` as `.npz` files. ✅

### 4. Detailed Logging 🗒️

Add the `--verbose` flag to print detailed decision information (Q-values, credibility, arbitration results):  
```bash
python main_htma_b.py --mode eval --verbose
```
🔍

### 5. Result Visualization 🎨

After evaluation, run `plotter.py` to generate visualizations (ensure paths are correct):  
- Reward curves: Cumulative reward comparison.  
- Success/Crash rates: Algorithm stability analysis.  
- Action distribution: Contribution of human, machine, and boundary decisions.  
- Trajectory plots: Visualization of landing paths.  
```bash
python plotter.py
```
📈🌐

## Results Showcase 🌟📊

We conducted a systematic “battle royale” of all algorithms in the **LunarLander-v3** simulation environment! ⚔️ The experiment began with 500 episodes of pretraining for the Double DQN model (acting as the machine agent) to ensure robust baseline decision-making capabilities. In a subsequent 100-episode evaluation, we collected the following exciting results, perfectly reproducing and validating the core ideas of the paper. 🚀

### 1. Cumulative Reward Comparison 🏆

- ![Reward Comparison](results/Reward%20Comparison.png)

The distribution of cumulative rewards clearly reveals the “strength” of each algorithm 💪, with superior algorithms achieving higher scores!

  * **In the Machine-Traded Human (MTHA) Scenario**:  
      * **HOA (Human-Only) 👨‍💻** has a median reward in the negative range, indicating poor performance.  
      * Algorithms with machine intervention, **MTHA** and **MTHA-B 🤖**, significantly outperform HOA, with median rewards soaring to **approximately 250 points**, demonstrating the power of the human-machine collaboration framework!  
  * **In the Human-Traded Machine (HTMA) Scenario**:  
      * **HTMA-B (with Autonomous Boundary) 🧠** consistently outperforms the baseline **MOA (Machine-Only)** and **HTMA**.  
      * HTMA-B achieves a median reward of **approximately 270 points**, with its reward distribution systematically higher than other algorithms, indicating superior task completion quality.

### 2. Success Rate and Flight Trajectories 🛰️

- ![Trajectory with Success Rate](results/Trajectory%20with%20Success%20Rate.jpg)

Success rate and trajectory analysis further confirm the reward comparison, showcasing the intelligence and stability of the algorithms.

  * **HOA** achieves a success rate of only **4%**, with chaotic trajectories resembling a “headless fly,” struggling to complete landing tasks.  
  * With machine intervention, **MTHA** and **MTHA-B** boost the success rate to **73%**, with highly goal-oriented and stable trajectories heading straight for the target!  
  * In the HTMA scenario, compared to **MOA**’s **66%** success rate, **HTMA** slightly improves it to **67%**.  
  * However, **HTMA-B**, with the autonomous boundary, achieves an impressive **89% success rate** 🎉, with the smoothest and most stable trajectories among all algorithms, showcasing its superior task execution performance.  
  * **Notably**, upgrading the original DQN to **Double DQN** significantly enhances system stability, a key factor in HTMA-B’s outstanding performance.

### 3. Decision Source Analysis 🤝

- ![Action Rate](results/Action%20Rate.png)

Who makes the decisions? This analysis provides insight into how humans, machines, and boundaries collaborate.

  * **In the MTHA Scenario**:  
      * Performance improvements stem from effective decision authority transfer. In the **MTHA** framework, the machine’s high-precision decisions dominate most of the time, compensating for human shortcomings.  
      * **MTHA-B**, with the autonomous boundary, further optimizes this by replacing some suboptimal human actions with boundary decisions, achieving finer decision-making.  
  * **In the HTMA Scenario**:  
      * **HTMA-B** exhibits a highly efficient **human-machine-boundary collaboration mode** 👨‍💻🤖🧠.  
      * The autonomous boundary continuously learns and integrates the best historical experiences of both humans and machines, forming a reliable third-party decision source that significantly enhances machine performance under human leadership, a core factor in achieving high success rates.

## License 📜

This project is licensed under the [MIT License](https://www.google.com/search?q=MIT+License). 🔓

## Acknowledgments 🙏

The implementation and ideas of this project are inspired by the following research work, to which we express our gratitude:

> Q. Zhang, Y. Kang, Y. -B. Zhao, P. Li and S. You, "Traded Control of Human-Machine Systems for Sequential Decision-Making Based on Reinforcement Learning," in IEEE Transactions on Artificial Intelligence, vol. 3, no. 4, pp. 553-565, Aug. 2022, doi: 10.1109/TAI.2021.3127857.

We also thank the community contributors and testers! 🤝

## Future Work 🌱

- Support more complex environments (e.g., Atari games or robotic tasks).  
- Integrate intent inference modules to optimize decisions based on human behavior.  
- Dynamically adjust the k value in k-NN to adapt to varying state densities.  
- Optimize the credibility assessment formula to improve algorithm adaptability.  
📅🚀
