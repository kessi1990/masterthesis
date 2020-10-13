#!/usr/bin/env bash
sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh Seaquest-v0 1
sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh SpaceInvaders-v0 1
sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh Breakout-v0 1
sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh Seaquest-v0 1
sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh SpaceInvaders-v0 1
sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh Breakout-v0 1
sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh Seaquest-v0 2
sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh SpaceInvaders-v0 2
sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh Breakout-v0 2
sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh Seaquest-v0 2
sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh SpaceInvaders-v0 2
sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh Breakout-v0 2
sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh Seaquest-v0 49
sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh SpaceInvaders-v0 49
sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh Breakout-v0 49
sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh Seaquest-v0 49
sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh SpaceInvaders-v0 49
sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh Breakout-v0 49
# sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh Seaquest-v0 256
# sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh SpaceInvaders-v0 256
# sbatch --partition=Gobi --gres=gpu:0 --exclusive cead_c.sh Breakout-v0 256
# sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh Seaquest-v0 256
# sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh SpaceInvaders-v0 256
# sbatch --partition=Gobi --gres=gpu:0 --exclusive darqn_c.sh Breakout-v0 256
