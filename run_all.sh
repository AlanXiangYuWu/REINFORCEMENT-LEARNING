#!/bin/bash
set -e
export PATH="/home/ubuntu/miniconda3/envs/delivery_pomdp/bin:$PATH"
export SDL_VIDEODRIVER=dummy
cd /home/ubuntu/alan_repository/REINFORCEMENT-LEARNING

# PPO - 10 seeds, 5M steps, 16 envs
mkdir -p runs/ppo
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== PPO seed=$SEED ====="
  python -m agents.train_ppo \
    --seed $SEED --total-steps 5000000 \
    --n-envs 16 --logdir runs/ppo --device cuda \
    2>&1 | tee -a runs/ppo/train.log
done

# RecurrentPPO - 10 seeds, 5M steps, 16 envs
mkdir -p runs/recurrent_ppo
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== RecurrentPPO seed=$SEED ====="
  python -m agents.train_recurrent_ppo \
    --seed $SEED --total-steps 5000000 \
    --n-envs 16 --logdir runs/recurrent_ppo --device cuda \
    2>&1 | tee -a runs/recurrent_ppo/train.log
done

# DQN - 10 seeds, 5M steps
mkdir -p runs/dqn
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== DQN seed=$SEED ====="
  python -m agents.train_dqn \
    --seed $SEED --total-steps 5000000 \
    --logdir runs/dqn --device cuda \
    2>&1 | tee -a runs/dqn/train.log
done

# QR-DQN - 10 seeds, 5M steps
mkdir -p runs/qrdqn
for SEED in 0 1 2 3 4 5 6 7 8 9; do
  echo "===== QR-DQN seed=$SEED ====="
  python -m agents.train_qrdqn \
    --seed $SEED --total-steps 5000000 \
    --logdir runs/qrdqn --device cuda \
    2>&1 | tee -a runs/qrdqn/train.log
done

echo "===== ALL TRAINING COMPLETE ====="
