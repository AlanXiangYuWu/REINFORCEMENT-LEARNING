#!/bin/bash
# Parallel training: all 4 algos run concurrently, DQN/QRDQN also parallelize seeds
set -e
export PATH="/home/ubuntu/miniconda3/envs/delivery_pomdp/bin:$PATH"
export SDL_VIDEODRIVER=dummy
cd /home/ubuntu/alan_repository/REINFORCEMENT-LEARNING

LOG_DIR="/tmp/rl_logs"
mkdir -p "$LOG_DIR"

###############################################################################
# PPO - 10 seeds sequential (16 envs each = heavy on CPU)
# seed 0 already done, start from 1
###############################################################################
train_ppo() {
  mkdir -p runs/ppo
  for SEED in 1 2 3 4 5 6 7 8 9; do
    echo "===== PPO seed=$SEED ====="
    python -m agents.train_ppo \
      --seed $SEED --total-steps 5000000 \
      --n-envs 16 --logdir runs/ppo --device cuda
  done
  echo "===== PPO ALL DONE ====="
}

###############################################################################
# RecurrentPPO - 10 seeds sequential (16 envs each)
###############################################################################
train_rppo() {
  mkdir -p runs/recurrent_ppo
  for SEED in 0 1 2 3 4 5 6 7 8 9; do
    echo "===== RecurrentPPO seed=$SEED ====="
    python -m agents.train_recurrent_ppo \
      --seed $SEED --total-steps 5000000 \
      --n-envs 16 --logdir runs/recurrent_ppo --device cuda
  done
  echo "===== RecurrentPPO ALL DONE ====="
}

###############################################################################
# DQN - 10 seeds, 2 at a time (only 1 env each, very light)
###############################################################################
train_dqn() {
  mkdir -p runs/dqn
  for BATCH_START in 0 2 4 6 8; do
    S1=$BATCH_START
    S2=$((BATCH_START + 1))
    echo "===== DQN seeds $S1 & $S2 ====="
    python -m agents.train_dqn \
      --seed $S1 --total-steps 5000000 \
      --logdir runs/dqn --device cuda &
    python -m agents.train_dqn \
      --seed $S2 --total-steps 5000000 \
      --logdir runs/dqn --device cuda &
    wait
  done
  echo "===== DQN ALL DONE ====="
}

###############################################################################
# QRDQN - 10 seeds, 2 at a time
###############################################################################
train_qrdqn() {
  mkdir -p runs/qrdqn
  for BATCH_START in 0 2 4 6 8; do
    S1=$BATCH_START
    S2=$((BATCH_START + 1))
    echo "===== QRDQN seeds $S1 & $S2 ====="
    python -m agents.train_qrdqn \
      --seed $S1 --total-steps 5000000 \
      --logdir runs/qrdqn --device cuda &
    python -m agents.train_qrdqn \
      --seed $S2 --total-steps 5000000 \
      --logdir runs/qrdqn --device cuda &
    wait
  done
  echo "===== QRDQN ALL DONE ====="
}

###############################################################################
# Launch all 4 algo groups in parallel
###############################################################################
echo "===== PARALLEL TRAINING STARTED at $(date) ====="

train_ppo  > "$LOG_DIR/ppo.log" 2>&1 &
PID_PPO=$!

train_rppo > "$LOG_DIR/rppo.log" 2>&1 &
PID_RPPO=$!

train_dqn  > "$LOG_DIR/dqn.log" 2>&1 &
PID_DQN=$!

train_qrdqn > "$LOG_DIR/qrdqn.log" 2>&1 &
PID_QRDQN=$!

echo "PIDs: PPO=$PID_PPO, RecurrentPPO=$PID_RPPO, DQN=$PID_DQN, QRDQN=$PID_QRDQN"
echo "Logs: $LOG_DIR/{ppo,rppo,dqn,qrdqn}.log"

# Wait for all to finish
wait $PID_PPO
echo "PPO finished at $(date)"
wait $PID_RPPO
echo "RecurrentPPO finished at $(date)"
wait $PID_DQN
echo "DQN finished at $(date)"
wait $PID_QRDQN
echo "QRDQN finished at $(date)"

echo "===== ALL PARALLEL TRAINING COMPLETE at $(date) ====="
