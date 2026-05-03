#!/bin/bash
# Fast training: 3 seeds, 2 groups of 2 algos in parallel
# Group 1: PPO + DQN (PPO has seed 0 done, start from 1)
# Group 2: RecurrentPPO + QRDQN
set -e
export PATH="/home/ubuntu/miniconda3/envs/delivery_pomdp/bin:$PATH"
export SDL_VIDEODRIVER=dummy
cd /home/ubuntu/alan_repository/REINFORCEMENT-LEARNING

LOG_DIR="/tmp/rl_logs"
mkdir -p "$LOG_DIR"

SEEDS="0 1 2"
STEPS=5000000

echo "===== GROUP 1: PPO + DQN started at $(date) ====="

# PPO: seed 0 already done, run 1 and 2
(
  mkdir -p runs/ppo
  for SEED in 1 2; do
    echo "===== PPO seed=$SEED ====="
    python -m agents.train_ppo \
      --seed $SEED --total-steps $STEPS \
      --n-envs 16 --logdir runs/ppo --device cuda
  done
  echo "===== PPO DONE at $(date) ====="
) > "$LOG_DIR/ppo.log" 2>&1 &
PID_PPO=$!

# DQN: 3 seeds sequential
(
  mkdir -p runs/dqn
  for SEED in $SEEDS; do
    echo "===== DQN seed=$SEED ====="
    python -m agents.train_dqn \
      --seed $SEED --total-steps $STEPS \
      --logdir runs/dqn --device cuda
  done
  echo "===== DQN DONE at $(date) ====="
) > "$LOG_DIR/dqn.log" 2>&1 &
PID_DQN=$!

echo "Group 1 PIDs: PPO=$PID_PPO, DQN=$PID_DQN"
wait $PID_PPO $PID_DQN
echo "===== GROUP 1 FINISHED at $(date) ====="

echo "===== GROUP 2: RecurrentPPO + QRDQN started at $(date) ====="

# RecurrentPPO: 3 seeds
(
  mkdir -p runs/recurrent_ppo
  for SEED in $SEEDS; do
    echo "===== RecurrentPPO seed=$SEED ====="
    python -m agents.train_recurrent_ppo \
      --seed $SEED --total-steps $STEPS \
      --n-envs 16 --logdir runs/recurrent_ppo --device cuda
  done
  echo "===== RecurrentPPO DONE at $(date) ====="
) > "$LOG_DIR/rppo.log" 2>&1 &
PID_RPPO=$!

# QRDQN: 3 seeds
(
  mkdir -p runs/qrdqn
  for SEED in $SEEDS; do
    echo "===== QRDQN seed=$SEED ====="
    python -m agents.train_qrdqn \
      --seed $SEED --total-steps $STEPS \
      --logdir runs/qrdqn --device cuda
  done
  echo "===== QRDQN DONE at $(date) ====="
) > "$LOG_DIR/qrdqn.log" 2>&1 &
PID_QRDQN=$!

echo "Group 2 PIDs: RecurrentPPO=$PID_RPPO, QRDQN=$PID_QRDQN"
wait $PID_RPPO $PID_QRDQN
echo "===== GROUP 2 FINISHED at $(date) ====="

echo "===== ALL TRAINING COMPLETE at $(date) ====="
