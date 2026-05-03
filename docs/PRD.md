# CA6126 Final Project — PRD & Task Outline

**Project title:** Sequential Multi-Target Delivery under Partial Observability
**Group:** CA6126 RL Group
**Members:** Wu Xiangyu (武翔宇), Chen Yunyi (陈鋆怡), Jiang Huajian (姜华健)

> **One-line pitch.** A delivery robot must visit `K` landmarks in a hidden
> order. The order is randomized each episode and never revealed; the agent
> must *infer* it from the binary feedback it receives at each visit. This
> turns a simple gridworld into a **POMDP** that requires memory.

---

## 1. Course context

This is the PRD for the **CA6126 final project**. The assignment asks
each group to *formulate a novel RL problem and solve it*. Grading:

- **RL formulation (10 pts)** — novelty (5), MDP/POMDP formalism (2),
  Gymnasium-style `env.py` (3).
- **RL solution (10 pts)** — showcase video of the agent playing (5),
  training-process documentation (5).

**Deliverables:**

- **Final report** (PDF, ≤ 20 pages) — written in **LaTeX**.
- **Videos** — (i) random agent (clearly bad) and (ii) trained agent
  (clearly better).
- **Source code (zip)**, no model checkpoints.

---

## 2. The problem: Sequential Multi-Target Delivery (POMDP)

### 2.1 Story

A delivery robot operates on a grid. `K` drop-off landmarks are placed at
random cells. The robot must visit them **in a specific order** that is
sampled uniformly at the start of each episode and **never revealed** to
the agent. The robot only learns whether a visit was *correct* or *wrong*
after trying it. To act optimally it must:

1. explore enough to identify the correct order, and
2. then take the shortest path through the remaining targets.

### 2.2 Why this is novel (and not a stock environment)

- Standard gridworlds expose the goal; here the goal-ordering is
  hidden, which is the whole point.
- It is **not** a TSP: the order is fixed but unknown. The agent
  must trade off **information gain** against **travel cost** —
  pure shortest-path heuristics fail.
- The optimal policy is non-Markov in the observation: the agent has
  to remember which landmarks it has tried and whether each attempt
  was rejected. This is what makes the POMDP non-trivial.

---

## 3. Formal POMDP specification

### 3.1 Notation

Let `G = {0..N-1} × {0..N-1}` be an `N × N` grid.
Let `Σ_K` be the set of permutations of `K` landmarks, so `|Σ_K| = K!`.

### 3.2 Hidden state `sₜ`

```
sₜ = ( pₜ , {ℓ_i}_{i=1}^K , π , kₜ , t )
```

- `pₜ ∈ G` — robot position.
- `ℓ_i ∈ G` — landmark positions (distinct, fixed within an episode).
- `π ∈ Σ_K` — required visiting order. **Hidden from the agent.**
- `kₜ ∈ {0,1,...,K}` — number of correct visits so far.
- `t` — step counter.

### 3.3 Observation `oₜ`

The agent receives:

- A `5×5` ego-centric view encoding `{empty, landmark_1, ..., landmark_K, wall}`.
- A history vector `hₜ ∈ {-1, 0, +1}^K` where, for each landmark `i`:
  - `-1` — tried and rejected,
  - `0` — unvisited / unknown,
  - `+1` — confirmed correct.
- Progress index `kₜ/K` and remaining-step budget `(T - t)/T`.

The hidden permutation `π` is **not** observable — this is what makes
the problem a POMDP.

### 3.4 Action space `𝒜`

Discrete, `|𝒜| = 5`:
```
𝒜 = { Up, Down, Left, Right, Visit }
```
`Visit` only has an effect when the agent stands on a landmark cell.

### 3.5 Transition `P(s' | s, a)`

- Movement is deterministic; moves into a wall leave `pₜ` unchanged.
- `Visit` on the `π(kₜ)`-th landmark advances `kₜ → kₜ + 1` and
  marks that landmark as `+1` in `hₜ`.
- `Visit` on a wrong landmark leaves `kₜ` unchanged, marks that
  landmark as `-1` in `hₜ`, and incurs a penalty.
- Episode terminates on `kₜ = K` (success) or `t = T` (timeout).

### 3.6 Reward `R(s, a, s')`

| Event                                | Reward |
| ------------------------------------ | -----: |
| Per-step cost (movement)             | -0.01  |
| Bumping into a wall                  | -0.05  |
| `Visit` on a non-landmark cell       | -0.05  |
| Correct landmark visit               | +1.00  |
| Wrong landmark visit                 | -p (configurable, default 0.1) |
| Completing the full sequence         | +5.00  |
| Timeout without completion           | -1.00  |

The wrong-visit penalty `p` is a critical hyperparameter. We found that
`p=0.5` (original) makes exploration irrational (negative expected value
per attempt), while `p=0.1` makes it rational.

### 3.7 Reward shaping

Optional potential-based shaping (Ng et al., 1999):
```
F(s, s') = γ · Φ(s') - Φ(s)
Φ(s) = -d_min(robot, nearest plausible landmark) / (2(N-1))
```
This guides the agent toward landmarks without changing the optimal policy.

---

## 4. Environment configurations

We tested two configurations:

| Setting              | Large (Phase 1-3)  | Small (Phase 4, **final**)  |
| -------------------- | ------------------ | --------------------------- |
| Grid size            | `12 × 12`         | **`6 × 6`**                 |
| Number of landmarks  | `4` (24 orderings) | **`3` (6 orderings)**       |
| Ego-view             | `5 × 5` (17% coverage) | **`5 × 5` (69% coverage)** |
| Max steps per ep.    | `200`              | **`100`**                   |
| Wrong-visit penalty  | `0.5`              | **`0.1`**                   |
| Reward shaping       | varies             | **Yes**                     |
| Hardware             | NVIDIA L20 (40 GB) | NVIDIA L20 (40 GB)         |

---

## 5. Algorithms

| Algorithm | Framework | Key feature |
| --------- | --------- | ----------- |
| **PPO** | Stable Baselines 3 | Feed-forward MLP [256,256], 16 parallel envs |
| **RecurrentPPO** | sb3-contrib | LSTM(128) + MLP [128,128], maintains belief across timesteps |
| **DQN** | Stable Baselines 3 | Experience replay (200k), ε-greedy exploration |
| **QR-DQN** | sb3-contrib | Distributional DQN, 51 quantiles |

---

## 6. Results summary

| Configuration | Algorithm | Success Rate | Reward |
| ------------- | --------- | ------------ | ------ |
| Large, no shaping | PPO / DQN / RecurrentPPO | **0%** | -3.0 |
| Large, shaping | RecurrentPPO | **0%** | -2.69 (train) |
| Large, shaping + low penalty | RecurrentPPO | **0%** | -2.62 (train) |
| **Small, shaping + low penalty** | **RecurrentPPO** | **87.5%** | **+6.09** |

---

## 7. Videos

Four demonstration videos are included in `videos/`:

| Video | Description |
| ----- | ----------- |
| [`random_agent.mp4`](../videos/random_agent.mp4) | **Random agent on 12×12 grid (Phase 1 baseline).** The agent moves erratically, bumps into walls, and attempts visits at random — accumulating heavy penalties. Shows what an untrained policy looks like. |
| [`ppo_agent.mp4`](../videos/ppo_agent.mp4) | **Trained PPO agent on 12×12 grid (Phase 1).** The agent moves smoothly without wall bumps but **never executes the Visit action** — it converged to the "never visit" avoidance strategy. Demonstrates the local optimum problem. |
| [`random_small.mp4`](../videos/random_small.mp4) | **Random agent on 6×6 grid (Phase 4 baseline).** Random policy on the small grid — still fails to complete the sequence, with erratic movement and random wrong visits. |
| [`rppo_small_best.mp4`](../videos/rppo_small_best.mp4) | **Trained RecurrentPPO on 6×6 grid (Phase 4, best model).** The agent navigates purposefully to landmarks, attempts visits in a systematic order, and **successfully completes the delivery sequence** in most episodes. 87.5% success rate, near-optimal reward. This is the main showcase video. |

---

## 8. Repository layout

```
REINFORCEMENT-LEARNING/
├── README.md               # Bilingual project overview
├── requirements.txt        # Python dependencies
├── RUN.md                  # Server execution playbook
├── env.py                  # Gymnasium POMDP environment
├── render.py               # Pygame renderer (headless-safe)
├── agents/
│   ├── train_ppo.py        # PPO trainer
│   ├── train_recurrent_ppo.py  # RecurrentPPO (LSTM) trainer
│   ├── train_dqn.py        # DQN trainer
│   ├── train_qrdqn.py      # QR-DQN trainer
│   ├── eval.py             # Evaluation script
│   ├── record_video.py     # Video recording
│   ├── plot_curves.py      # Training curve plotting
│   └── common.py           # Shared env factory & defaults
├── docs/
│   └── PRD.md              # This document
├── report/
│   ├── report.tex          # LaTeX report (6 pages)
│   └── report.pdf          # Compiled report
├── videos/
│   ├── random_agent.mp4    # Random agent demo (12×12)
│   ├── ppo_agent.mp4       # Trained PPO demo (12×12, avoidance)
│   ├── random_small.mp4    # Random agent demo (6×6)
│   └── rppo_small_best.mp4 # Trained RecurrentPPO demo (6×6, 87.5%)
├── tests/
│   └── test_env.py         # Environment unit tests
└── runs/                   # Training logs & checkpoints (gitignored)
```

---

## 9. Milestones (completed)

| Milestone | Output | Owner |
| --------- | ------ | ----- |
| M1 | PRD finalized (this document) | Team |
| M2 | `env.py` passing `check_env` | Wu Xiangyu |
| M3 | Random-agent video + sanity metrics | Team |
| M4 | PPO/DQN baseline trained — discovered "never visit" problem | Team |
| M5 | RecurrentPPO + reward shaping + scaling experiments | Team |
| M6 | 87.5% success on small grid, final report + videos | Team |

---

## 10. Mapping to the grading rubric

| Rubric item         | Pts    | Where addressed              |
| ------------------- | -----: | ---------------------------- |
| Novelty             | 5      | §2.2 — hidden-order POMDP    |
| MDP/POMDP formalism | 2      | §3 — full POMDP tuple        |
| Gymnasium env       | 3      | `env.py` (configurable)      |
| Showcase videos     | 5      | `videos/` (4 videos)         |
| Training process    | 5      | Report §5 (4-phase journey)  |
| **Total**           | **20** |                              |
