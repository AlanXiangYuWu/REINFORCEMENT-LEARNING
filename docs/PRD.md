# CA6126 Final Project — PRD & Task Outline

**Project title:** Sequential Multi-Target Delivery under Partial Observability
**Group:** _TBD_
**Members:** _TBD_

> **One-line pitch.** A delivery robot must visit three drop-off points
> `A`, `B`, `C` in a hidden order. The order is randomized each episode
> and never revealed; the agent must *infer* it from the binary feedback
> it receives at each visit. This turns a simple gridworld into a
> **POMDP** that requires memory.

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

A delivery robot operates on a small grid. Three drop-off points
`A`, `B`, `C` are placed at random cells. The robot must visit them
**in a specific order** that is sampled uniformly at the start of each
episode and **never revealed** to the agent. The robot only learns
whether a visit was *correct* or *wrong* after trying it. To act
optimally it must:

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
  was rejected. This is what makes the POMDP non-trivial without
  making the task itself too hard.
- Inspired by the lecture's elevator example: a small, intuitive
  system whose decision-making is non-trivial because the relevant
  state is partially observed.

---

## 3. Formal POMDP specification

### 3.1 Notation

Let `G = {0..N-1} × {0..N-1}` be an `N × N` grid (default `N = 10`).
Let `Σ₃` be the set of permutations of `(A, B, C)`, so `|Σ₃| = 6`.

### 3.2 Hidden state `sₜ`

```
sₜ = ( pₜ , (ℓ_A, ℓ_B, ℓ_C) , π , kₜ , t )
```

- `pₜ ∈ G` — robot position.
- `ℓ_A, ℓ_B, ℓ_C ∈ G` — landmark positions (distinct, fixed within
  an episode).
- `π ∈ Σ₃` — required visiting order. **Hidden from the agent.**
- `kₜ ∈ {0,1,2,3}` — number of correct visits so far.
- `t` — step counter.

### 3.3 Observation `oₜ`

The agent receives:

- Its own position `pₜ`.
- A `5×5` ego-centric view encoding `{empty, A, B, C, wall}`.
- A history vector `hₜ ∈ {-1, 0, +1}³` where, for each landmark `i`:
  - `-1` — tried and rejected,
  - `0` — unvisited / unknown,
  - `+1` — confirmed correct.
  This is the agent's *sufficient summary* of past attempts.
- Progress index `kₜ` and remaining-step budget `T - t`.

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
- Episode terminates on `kₜ = 3` (success) or `t = T` (timeout,
  default `T = 100`).

### 3.6 Reward `R(s, a, s')`

| Event                                | Reward |
| ------------------------------------ | -----: |
| Per-step cost (movement)             | -0.01  |
| Bumping into a wall                  | -0.05  |
| `Visit` on a non-landmark cell       | -0.05  |
| Correct landmark visit               | +1.00  |
| Wrong landmark visit                 | -0.50  |
| Completing the full sequence         | +5.00  |
| Timeout without completion           | -1.00  |

Per-step cost encourages efficiency; the wrong-visit penalty is larger
than the per-step cost so the agent prefers to scout via geometry
rather than guess.

### 3.7 State and action space size

- Robot positions: `|G| = 64`.
- Landmark placements: `C(64, 3) · 3! ≈ 2.5 × 10⁵`.
- Hidden orderings: `6`.
- Progress: `4`.
- Step counter: `≤ 101`.
- History `h`: `3³ = 27`.
- **Hidden state space** `|S| ≈ 64 · 2.5×10⁵ · 6 · 4 · 101 ≈ 4 × 10¹⁰`.
- **Observation space** (what the policy sees) is much smaller: a
  `5×5×5` tensor (one-hot for `{empty, A, B, C, wall}`) plus a small
  vector of length 7.
- Actions: `|𝒜| = 5`.

---

## 4. Environment (`env.py`)

### 4.1 API

The environment will follow the `gymnasium.Env` API exactly:

```python
import gymnasium as gym
from delivery_pomdp.env import SequentialDeliveryEnv

env = SequentialDeliveryEnv(grid_size=8, max_steps=100, seed=0)
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
```

### 4.2 Spaces

- `action_space`: `Discrete(5)`.
- `observation_space`: `Dict` with keys
  - `view`: `Box(0, 1, (5, 5, 5), float32)` — one-hot ego view.
  - `history`: `Box(-1, 1, (3,), int8)`.
  - `progress`: `Discrete(4)`.
  - `steps_left`: `Box(0, 1, (1,), float32)`.

### 4.3 Rendering

`render_mode="rgb_array"` returns a Pygame-rendered RGB frame; this
is what the video script captures via `imageio.mimsave(...)`.

---

## 5. Solution plan

### 5.1 Algorithm choice

Because the optimal policy depends on history, we will compare:

1. **PPO** with the agent's history vector `hₜ` baked into the
   observation (Markovian summary).
2. **Recurrent PPO** (`sb3-contrib`'s `RecurrentPPO`) with an LSTM
   policy — treats the problem as a true POMDP and lets the network
   learn its own belief.
3. **DQN** on the same observation, as a sanity baseline.

### 5.2 Why this should train

- Action space is small (`|𝒜| = 5`); episodes are short (`T = 100`);
  reward is dense enough thanks to the per-correct-visit bonus.
- The hand-crafted history vector `hₜ` already gives feed-forward PPO
  most of what it needs, so we expect PPO to work; RecurrentPPO is
  the more principled solution.

### 5.3 Risks and mitigations

- **Sparse early reward.** Add a small potential-based shaping term
  toward the nearest *plausible* landmark (those still marked `0`
  or `+1` in `hₜ`).
- **Wrong-visit penalty too harsh** ⇒ agent avoids visiting at all.
  Tune via grid sweep on `{-0.25, -0.5, -1.0}`.
- **Non-stationarity from random ordering.** Evaluate on a fixed
  eval seed bank to track progress cleanly.

---

## 6. Repository layout

```
REINFORCEMENT-LEARNING/
  README.md
  pyproject.toml          # or requirements.txt
  env.py                  # gymnasium environment (top-level, per spec)
  delivery_pomdp/
    __init__.py
    env.py                # full implementation
    render.py             # pygame rendering
    wrappers.py           # observation flattening, frame stacking
  agents/
    train_ppo.py
    train_recurrent_ppo.py
    train_dqn.py
    eval.py               # rollouts + metrics
    record_video.py       # produces mp4s for the report
  docs/
    PRD.md                # this document
  report/
    report.tex            # final report (LaTeX)
  videos/
    random_agent.mp4
    trained_agent.mp4
  runs/                   # tensorboard logs (gitignored)
```

---

## 7. Milestones

| Milestone | Output                                                    | Owner |
| --------- | --------------------------------------------------------- | ----- |
| M1        | PRD finalized (this document)                             | TBD   |
| M2        | `env.py` passing `gymnasium.utils.env_checker.check_env`  | TBD   |
| M3        | Random-agent video + sanity metrics                       | TBD   |
| M4        | PPO baseline trained, learning curve plotted              | TBD   |
| M5        | RecurrentPPO trained, comparison plot                     | TBD   |
| M6        | Trained-agent video + final report (LaTeX)                | TBD   |

---

## 8. Mapping to the grading rubric

| Rubric item         | Pts    | Where addressed              |
| ------------------- | -----: | ---------------------------- |
| Novelty             | 5      | §2.2                         |
| MDP/POMDP formalism | 2      | §3                           |
| Gymnasium env       | 3      | `env.py` (M2)                |
| Showcase videos     | 5      | `videos/` (M3, M6)           |
| Training process    | 5      | §5 + curves added at M4–M5   |
| **Total**           | **20** |                              |

---

## 9. Decisions

| Setting              | Value                                              |
| -------------------- | -------------------------------------------------- |
| Grid size            | `12 × 12`                                          |
| Number of landmarks  | `4` (24 hidden orderings → strong POMDP signal)    |
| Ego-view             | `5 × 5`                                            |
| Max steps per ep.    | `200`                                              |
| Walls / obstacles    | None for v1 (open grid); add as v2 ablation        |
| Hardware             | NVIDIA L20 (40 GB)                                 |
| Algorithms           | PPO, RecurrentPPO (LSTM), DQN, QR-DQN              |
| Training budget      | 10 seeds × 5 M env steps per algorithm             |
| Vectorized envs      | 16 parallel envs (`SubprocVecEnv`)                 |
| Eval                 | Fixed bank of 100 eval seeds, every 50 k steps     |
