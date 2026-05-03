# Server execution playbook (L20 40 GB)

Follow top-to-bottom. Every block is a copy-paste-able shell snippet.
Total wall-clock at the chosen budget (12×12 grid, 4 landmarks,
10 seeds × 5 M steps × 4 algorithms): **~20–25 h** (one overnight run).

> All paths are relative to the repo root (`REINFORCEMENT-LEARNING/`).

---

## 0. Hardware sanity

```bash
nvidia-smi                        # confirm L20 40G is visible
nproc                             # SubprocVecEnv uses CPU cores -- want >= 16
free -g                           # need ~20 GB RAM headroom for replay buffers
df -h .                           # need ~10 GB for runs/, videos/, tb logs
```

If `nproc` < 16, drop `--n-envs` accordingly in the PPO/RecurrentPPO
commands below.

---

## 1. Get the code onto the server

Two options — pick one.

### 1a. Clone (if you've pushed to GitHub)

```bash
cd ~
git clone git@github.com:AlanXiangYuWu/REINFORCEMENT-LEARNING.git
cd REINFORCEMENT-LEARNING
```

### 1b. rsync from your laptop

```bash
# from your Mac:
rsync -avz --exclude runs --exclude videos --exclude __pycache__ \
  /Users/alanv/Desktop/ntu-aai/REINFORCEMENT\ LEARNING/REINFORCEMENT\ LEARNING-teamwork/REINFORCEMENT-LEARNING/ \
  user@server:~/REINFORCEMENT-LEARNING/
ssh user@server
cd ~/REINFORCEMENT-LEARNING
```

---

## 2. Create the environment

`stable-baselines3` 2.3.x requires Python 3.9–3.12. Use 3.11.

```bash
# conda (preferred)
conda create -n delivery_pomdp python=3.11 -y
conda activate delivery_pomdp

# OR venv
# python3.11 -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# verify torch sees the GPU
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `cuda: True NVIDIA L20`.

---

## 3. Smoke test the environment

```bash
# Headless pygame -- safe to set globally for this session
export SDL_VIDEODRIVER=dummy

python -m tests.test_env          # check_env + rollout + reward signs
python env.py                     # one random rollout
```

Both should finish in < 5 s. `check_env` complaining about anything
is a blocker; fix before training.

---

## 4. Record the random-agent baseline video

We need this for the rubric (5 pts: showcase video). Do it before
training so we have something to commit early.

```bash
mkdir -p videos
python -m agents.record_video \
  --algo random --n-episodes 3 --fps 8 \
  --out videos/random_agent.mp4
```

Open `videos/random_agent.mp4`. The agent should look obviously
clueless — wandering, hitting walls, visiting landmarks in wrong
order. That's the point.

Also record a baseline-metrics report:

```bash
python -m agents.eval \
  --algo random --n-episodes 200 \
  --out runs/baseline_random.json
cat runs/baseline_random.json
```

Expected: `success_rate ≈ 0.00–0.02`, `reward_mean` strongly
negative.

---

## 5. Launch training

Each algo trains 10 seeds. Within an algo, seeds run sequentially
on the single GPU (each algo only needs ~3 GB so we don't try to
parallelise across seeds — it would just thrash). Across algos, run
sequentially in one big `tmux` session.

```bash
tmux new -s delivery_pomdp
```

### 5a. PPO (fastest, ~7 h for 10 seeds)

```bash
mkdir -p runs/ppo
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== PPO seed=$SEED ====="
  python -m agents.train_ppo \
    --seed $SEED --total-steps 5000000 \
    --n-envs 16 --logdir runs/ppo --device cuda \
    2>&1 | tee -a runs/ppo/train.log
done
```

### 5b. RecurrentPPO (the principled POMDP solution, ~10 h)

```bash
mkdir -p runs/recurrent_ppo
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== RecurrentPPO seed=$SEED ====="
  python -m agents.train_recurrent_ppo \
    --seed $SEED --total-steps 5000000 \
    --n-envs 16 --logdir runs/recurrent_ppo --device cuda \
    2>&1 | tee -a runs/recurrent_ppo/train.log
done
```

### 5c. DQN (single-env, ~6 h)

```bash
mkdir -p runs/dqn
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== DQN seed=$SEED ====="
  python -m agents.train_dqn \
    --seed $SEED --total-steps 5000000 \
    --logdir runs/dqn --device cuda \
    2>&1 | tee -a runs/dqn/train.log
done
```

### 5d. QR-DQN (~6 h)

```bash
mkdir -p runs/qrdqn
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== QR-DQN seed=$SEED ====="
  python -m agents.train_qrdqn \
    --seed $SEED --total-steps 5000000 \
    --logdir runs/qrdqn --device cuda \
    2>&1 | tee -a runs/qrdqn/train.log
done
```

> **Detach with `Ctrl-b d`** and reattach later via `tmux a -t delivery_pomdp`.
> If something fails partway, the loop `for SEED in ...` lets you
> resume by editing the seed list.

---

## 6. Monitor progress (optional)

In a second terminal on the server:

```bash
# Live tensorboard:
tensorboard --logdir runs --bind_all --port 6006
# Then on your laptop:  ssh -L 6006:localhost:6006 user@server
# and open http://localhost:6006
```

You should see `eval/mean_reward` climbing within ~200k steps for
PPO; RecurrentPPO is slower to start but reaches a higher ceiling.

---

## 7. Plot training curves

After training (or even partway through, for a sanity check):

```bash
python -m agents.plot_curves \
  --runs-root runs \
  --algos ppo recurrent_ppo dqn qrdqn \
  --out runs/curves.png
```

This reads each algo's `runs/<algo>/seed_*/eval/evaluations.npz`,
takes mean ± std across seeds, and writes `runs/curves.png`.

---

## 8. Pick the best models and evaluate

`EvalCallback` already saved `best/best_model.zip` per seed. Pick the
best seed per algo (highest final eval reward — TensorBoard or
`grep` the train log) or just use seed 0:

```bash
# Edit BEST=<seed> per algo as needed.
for ALGO in ppo recurrent_ppo dqn qrdqn; do
  BEST=0
  python -m agents.eval \
    --algo $ALGO \
    --model runs/$ALGO/seed_$BEST/best/best_model.zip \
    --n-episodes 200 \
    --out runs/$ALGO/eval_metrics.json
done

# Compare:
for ALGO in ppo recurrent_ppo dqn qrdqn; do
  echo "=== $ALGO ==="
  cat runs/$ALGO/eval_metrics.json
done
```

Target metric: `success_rate ≥ 0.80` for at least one of PPO /
RecurrentPPO is a "we have a working RL solution" signal.

---

## 9. Record the trained-agent video (rubric: 5 pts)

```bash
# Pick the algo with the highest success_rate from step 8.
BEST_ALGO=recurrent_ppo
BEST_SEED=0

python -m agents.record_video \
  --algo $BEST_ALGO \
  --model runs/$BEST_ALGO/seed_$BEST_SEED/best/best_model.zip \
  --n-episodes 5 --fps 8 \
  --out videos/trained_agent.mp4
```

Open `videos/trained_agent.mp4`. It should:
- visit landmarks in the *correct* hidden order (no red X overlays)
- not waste time bumping walls
- complete most episodes well before the 200-step horizon

If it doesn't, train another seed or more steps; or fall back to
PPO. Either way, document what you saw in the report.

---

## 10. Pull artifacts back to the laptop

You need: videos, training curves, eval metrics, train logs.
Skip the multi-GB model checkpoints (the spec says no checkpoints in
the submission anyway).

```bash
# from your Mac:
mkdir -p ~/Desktop/delivery_pomdp_artifacts
rsync -avz \
  --include='*/' \
  --include='videos/***' \
  --include='runs/curves.png' \
  --include='runs/*/eval_metrics.json' \
  --include='runs/*/train.log' \
  --include='runs/baseline_random.json' \
  --exclude='*' \
  user@server:~/REINFORCEMENT-LEARNING/ \
  ~/Desktop/delivery_pomdp_artifacts/
```

---

## 11. Write the final report (LaTeX)

Once steps 1–10 are done, the LaTeX report can be filled in. The
skeleton lives at `report/report.tex` (will be added in the next
step). Compile with:

```bash
cd report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex   # second pass for refs
```

Output: `report/report.pdf`. Keep it ≤ 20 pages (project spec).

---

## Troubleshooting

**`pygame.error: No available video device`**
→ `export SDL_VIDEODRIVER=dummy` before running `record_video.py`.
`render.py` already sets this by default but some shells override it.

**`AssertionError: The observation returned by reset() ...`**
from `check_env` → an env field is the wrong dtype. Compare to the
spaces in `env.py`'s `__init__`.

**RecurrentPPO OOMs**
→ Lower `n_envs` to 8 or `n_steps` to 128 in
`agents/train_recurrent_ppo.py`. L20 40 GB shouldn't OOM, but a
crowded shared box might.

**Training reward stuck at ~ -2**
→ Agent is timing out without any correct visit. Try
`--reward-shaping` to add the potential-based hint, then turn it
back off for a clean comparison run.

**`ffmpeg` not found when writing mp4**
→ `pip install imageio-ffmpeg` (already in `requirements.txt`); on
some boxes also need `apt-get install -y ffmpeg`.

---

## What ends up in the submission zip

```
submission.zip
├── env.py                         # gymnasium env (rubric: 3 pts)
├── render.py
├── agents/                        # train + eval + video scripts
├── tests/
├── docs/PRD.md
├── report/
│   └── report.pdf                 # ≤ 20 pages (rubric: report)
├── videos/
│   ├── random_agent.mp4           # rubric: showcase 5 pts
│   └── trained_agent.mp4          # rubric: showcase 5 pts
├── runs/
│   ├── curves.png                 # rubric: training process 5 pts
│   ├── baseline_random.json
│   └── <algo>/eval_metrics.json
├── requirements.txt
├── README.md
└── RUN.md
```

**Excluded from zip:** `runs/*/seed_*/best/`, `runs/*/seed_*/final.zip`
(model checkpoints), `runs/*/seed_*/tb/` (tensorboard event files),
and `__pycache__/`.

```bash
# Final packaging step:
zip -r submission.zip . \
  -x '*/__pycache__/*' '*.pyc' \
     'runs/*/seed_*/best/*' 'runs/*/seed_*/final.zip' \
     'runs/*/seed_*/tb/*' \
     '.git/*' '.venv/*' 'env_local/*'
```
