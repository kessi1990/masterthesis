#!/usr/bin/env bash
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 20 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 20 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Seaquest-v0 1 20 dot 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh Seaquest-v0 1 20 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh Seaquest-v0 1 20 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh Seaquest-v0 1 20 dot 128

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 20 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 20 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Pong-v0 1 20 dot 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh Pong-v0 1 20 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh Pong-v0 1 20 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh Pong-v0 1 20 dot 128

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 20 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 20 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh BeamRider-v0 1 20 dot 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh BeamRider-v0 1 20 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh BeamRider-v0 1 20 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh BeamRider-v0 1 20 dot 128

sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 20 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 20 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 1 20 dot 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh Breakout-v0 1 20 add 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh Breakout-v0 1 20 concat 128
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh Breakout-v0 1 20 dot 128

# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 20 add 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 20 concat 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_darqn.sh SpaceInvaders-v0 1 20 dot 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh SpaceInvaders-v0 1 20 add 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh SpaceInvaders-v0 1 20 concat 128
# sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive batch_cead.sh SpaceInvaders-v0 1 20 dot 128


