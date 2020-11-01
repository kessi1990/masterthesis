#!/usr/bin/env bash

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 2 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 2 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 2 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 2 concat 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 2 2 add 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 2 2 concat 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 2 2 add 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 2 2 concat 128

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 2 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 2 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 2 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 2 concat 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 2 2 add 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 2 2 concat 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 2 2 add 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 2 2 concat 128

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 2 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 2 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 2 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 2 concat 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 2 2 add 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 2 2 concat 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 2 2 add 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 2 2 concat 128

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 2 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 2 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 2 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 2 concat 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 2 2 add 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 2 2 concat 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 2 2 add 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 2 2 concat 128

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 2 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 2 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 2 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 2 concat 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 2 2 add 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 2 2 concat 64
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 2 2 add 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 2 2 concat 128

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_dqn.sh Breakout-v0 0 2 0 0
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_dqn.sh Seaquest-v0 0 2 0 0
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_dqn.sh BeamRider-v0 0 2 0 0
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_dqn.sh SpaceInvaders-v0 0 2 0 0
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_dqn.sh Pong-v0 0 2 0 0
