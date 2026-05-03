# Sequential Multi-Target Delivery under Partial Observability

## CA6126 Reinforcement Learning Group Project | CA6126 强化学习小组作业

### Team Members | 小组成员

- **Wu Xiangyu (武翔宇)**
- **Chen Yunyi (可爱陈鋆怡)**
- **Jiang Huajian (温州帅哥姜华健)**

---

## Project Overview | 项目概述

**EN:** A delivery robot navigates a grid and must visit `K` landmarks in a **hidden order** that is randomized each episode. The agent receives only binary feedback (correct/wrong) after each visit attempt — it must *infer* the correct ordering from its interaction history. This makes the task a **Partially Observable Markov Decision Process (POMDP)** that requires memory-based policies.

**CN:** 一个配送机器人在网格中导航，必须按照每局随机生成的**隐藏顺序**访问 `K` 个地标。智能体每次访问后只能获得二元反馈（正确/错误），因此必须从交互历史中*推断*正确的访问顺序。这使得该任务成为一个需要记忆策略的**部分可观测马尔可夫决策过程（POMDP）**。

---

## Key Results | 核心结果

| Configuration | Algorithm | Success Rate | Reward |
|--------------|-----------|-------------|--------|
| 12×12, K=4, no shaping | PPO / DQN / RecurrentPPO | 0% | -3.0 |
| 6×6, K=3, shaping + low penalty | **RecurrentPPO (LSTM)** | **87.5%** | **+6.09** |

---

## Implementation Journey | 实现过程

**EN:** Our project went through 4 experimental phases, systematically debugging why standard RL algorithms fail on this POMDP:

**CN:** 我们的项目经历了 4 个实验阶段，系统性地排查了标准 RL 算法在此 POMDP 上失败的原因：

### Phase 1: Large Grid Baseline | 第一阶段：大网格基线
- **Setup | 设置:** 12×12 grid, 4 landmarks, penalty=0.5, no reward shaping
- **Result | 结果:** All algorithms (PPO, DQN, RecurrentPPO) converge to a "never visit" strategy (reward=-3.0, 0% success). The agent learns that doing nothing is safer than risking wrong-visit penalties.
- **结果:** 所有算法（PPO、DQN、RecurrentPPO）都收敛到"从不访问"策略（reward=-3.0，成功率 0%）。智能体发现什么都不做比冒着错误访问的惩罚风险更安全。

### Phase 2: Adding Reward Shaping | 第二阶段：添加奖励塑形
- **Setup | 设置:** +Potential-based reward shaping (guide agent toward landmarks)
- **Result | 结果:** Agent learns to approach landmarks (train reward improves to -2.69) but still 0% success — approaches but doesn't dare to visit.
- **结果:** 智能体学会了靠近地标（训练 reward 提升至 -2.69），但成功率仍为 0%——靠近了但不敢访问。

### Phase 3: Reducing Penalty | 第三阶段：降低惩罚
- **Setup | 设置:** Wrong-visit penalty reduced from 0.5 to 0.1, making exploration have positive expected value
- **Result | 结果:** Slight improvement (-2.62) but 12×12 grid still too large — agent can't find landmarks with 17% view coverage.
- **结果:** 略有改善（-2.62），但 12×12 网格仍然太大——智能体在 17% 视野覆盖率下找不到地标。

### Phase 4: Scaling Down — Breakthrough! | 第四阶段：缩小规模——突破！
- **Setup | 设置:** 6×6 grid (69% view coverage), 3 landmarks (6 orderings), reward shaping + low penalty
- **Result | 结果:** **RecurrentPPO achieves 87.5% success rate, reward +6.09 (best +7.91, near theoretical optimal)**
- **结果:** **RecurrentPPO 达到 87.5% 成功率，reward +6.09（最佳 +7.91，接近理论最优值）**

### Three Barriers Identified | 识别出的三个障碍

1. **Exploration Incentive | 探索激励:** Wrong-visit penalty must be low enough that trying is rational (期望值为正)
2. **Spatial Coverage | 空间覆盖:** Ego-view must cover enough of the grid to see landmarks (视野必须足够大)
3. **Memory Capacity | 记忆能力:** LSTM (RecurrentPPO) is essential — feed-forward PPO cannot solve the POMDP (LSTM 是必需的)

---

## Tech Stack | 技术栈

- **Framework | 框架:** [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) + [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- **Environment | 环境:** Custom [Gymnasium](https://gymnasium.farama.org/) environment (`env.py`)
- **Algorithms | 算法:** PPO, RecurrentPPO (LSTM), DQN, QR-DQN
- **Rendering | 渲染:** Pygame (headless-safe)
- **Hardware | 硬件:** NVIDIA L20 (40GB), 32 CPU cores

---

## Repository Structure | 项目结构

```
REINFORCEMENT-LEARNING/
├── env.py                  # Gymnasium POMDP environment | POMDP 环境
├── render.py               # Pygame renderer | 渲染器
├── agents/
│   ├── train_ppo.py        # PPO trainer
│   ├── train_recurrent_ppo.py  # RecurrentPPO (LSTM) trainer
│   ├── train_dqn.py        # DQN trainer
│   ├── train_qrdqn.py      # QR-DQN trainer
│   ├── eval.py             # Evaluation script | 评估脚本
│   ├── record_video.py     # Video recording | 录制视频
│   └── plot_curves.py      # Training curves | 训练曲线
├── report/
│   ├── report.tex          # LaTeX report | LaTeX 报告
│   └── report.pdf          # Compiled report | 编译后的报告
├── videos/                 # Agent demo videos (see below) | 演示视频（见下方）
├── docs/PRD.md             # Project requirements | 项目需求文档
├── tests/test_env.py       # Environment tests | 环境测试
├── requirements.txt
├── RUN.md                  # Server execution playbook | 服务器运行手册
└── README.md
```

---

## Quick Start | 快速开始

```bash
# Create environment | 创建环境
conda create -n delivery_pomdp python=3.11 -y
conda activate delivery_pomdp
pip install -r requirements.txt

# Smoke test | 冒烟测试
python -m tests.test_env
python env.py

# Train the best model (small grid) | 训练最佳模型（小网格）
export SDL_VIDEODRIVER=dummy
python -m agents.train_recurrent_ppo \
  --seed 0 --total-steps 2000000 \
  --n-envs 16 --logdir runs/rppo_small --device cuda \
  --reward-shaping --wrong-visit-penalty 0.1 \
  --grid-size 6 --n-landmarks 3 --max-steps 100

# Evaluate | 评估
python -m agents.eval --algo recurrent_ppo \
  --model runs/rppo_small/seed_0/best/best_model.zip --n-episodes 200

# Record video | 录制视频
python -m agents.record_video --algo recurrent_ppo \
  --model runs/rppo_small/seed_0/best/best_model.zip \
  --n-episodes 5 --fps 8 --out videos/trained_agent.mp4
```

For the full multi-algorithm training pipeline, see [`RUN.md`](RUN.md).

完整的多算法训练流程请参见 [`RUN.md`](RUN.md)。

---

## Demo Videos | 演示视频

Four demo videos are included in `videos/`, documenting the progression from failure to success:

`videos/` 中包含 4 个演示视频，记录了从失败到成功的完整过程：

| Video | Description |
|-------|-------------|
| [`random_agent.mp4`](videos/random_agent.mp4) | **Random agent on 12×12 grid (baseline).** Moves erratically, bumps walls, visits landmarks at random — heavy penalties, 0% success. |
| [`ppo_agent.mp4`](videos/ppo_agent.mp4) | **Trained PPO on 12×12 grid.** Moves smoothly but **never visits any landmark** — converged to the "never visit" avoidance strategy. Demonstrates the local optimum problem. |
| [`random_small.mp4`](videos/random_small.mp4) | **Random agent on 6×6 grid (small version baseline).** Still fails with random movement and wrong visits on the smaller grid. |
| [`rppo_small_best.mp4`](videos/rppo_small_best.mp4) | **Trained RecurrentPPO on 6×6 grid (best model).** Navigates purposefully, visits landmarks in correct order, **87.5% success rate, reward +6.09**. This is the main showcase. |

| 视频 | 说明 |
|------|------|
| `random_agent.mp4` | **12×12 随机基线** — 随机移动、撞墙、乱访问，重罚，成功率 0% |
| `ppo_agent.mp4` | **12×12 训练后 PPO** — 移动顺畅但**从不访问地标**，收敛到"回避策略"局部最优 |
| `random_small.mp4` | **6×6 随机基线** — 小网格上的随机策略，仍然失败 |
| `rppo_small_best.mp4` | **6×6 训练后 RecurrentPPO（最佳模型）** — 有目的地导航，按正确顺序访问，**87.5% 成功率** |

---

## Final Results (3 seeds) | 最终结果（3 个种子）

| Seed | Reward | Success Rate |
|------|--------|-------------|
| Seed 0 | +6.09 ± 3.04 | **87.5%** |
| Seed 1 | -2.00 ± 0.11 | 0% (failed to learn) |
| Seed 2 | +4.79 ± 4.59 | **78.5%** |

This variance across seeds highlights the stochastic nature of RL training — the same hyperparameters can yield very different outcomes.

不同种子间的差异体现了 RL 训练的随机性——相同超参数可能产生截然不同的结果。
