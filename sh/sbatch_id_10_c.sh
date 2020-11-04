#!/usr/bin/env bash
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 10 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 10 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 10 dot 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 2 10 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 2 10 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 2 10 dot 64

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 10 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 10 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 10 dot 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 2 10 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 2 10 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 2 10 dot 64

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 10 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 10 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 10 dot 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 2 10 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 2 10 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 2 10 dot 64

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 10 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 10 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 10 dot 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 2 10 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 2 10 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 2 10 dot 64

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 10 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 10 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 10 dot 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 2 10 add 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 2 10 concat 64
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 2 10 dot 64


