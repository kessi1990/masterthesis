#!/usr/bin/env bash
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_dqn.sh Breakout-v0 0 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_dqn.sh Seaquest-v0 0 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_dqn.sh BeamRider-v0 0 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_dqn.sh SpaceInvaders-v0 0 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_dqn.sh Pong-v0 0 0

sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 2 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 4 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 8 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Breakout-v0 2 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Breakout-v0 4 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Breakout-v0 8 0

sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 2 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 4 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 8 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Seaquest-v0 2 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Seaquest-v0 4 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Seaquest-v0 8 0

#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 2 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 4 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 8 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh BeamRider-v0 2 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh BeamRider-v0 4 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh BeamRider-v0 8 0

sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 2 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 4 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 8 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh SpaceInvaders-v0 2 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh SpaceInvaders-v0 4 0
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh SpaceInvaders-v0 8 0

#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 2 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 4 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 8 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Pong-v0 2 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Pong-v0 4 0
#sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Pong-v0 8 0
