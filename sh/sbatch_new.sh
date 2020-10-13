#!/usr/bin/env bash
# sbatch --gres=gpu:1 cead.sh Seaquest-v0 1
# sbatch --gres=gpu:1 cead.sh SpaceInvaders-v0 1
# sbatch --gres=gpu:1 cead.sh Breakout-v0 1
# sbatch --gres=gpu:1 darqn.sh Seaquest-v0 1
# sbatch --gres=gpu:1 darqn.sh SpaceInvaders-v0 1
# sbatch --gres=gpu:1 darqn.sh Breakout-v0 1
sbatch --gres=gpu:1 cead.sh Seaquest-v0 2
sbatch --gres=gpu:1 cead.sh SpaceInvaders-v0 2
sbatch --gres=gpu:1 cead.sh Breakout-v0 2
sbatch --gres=gpu:1 darqn.sh Seaquest-v0 2
sbatch --gres=gpu:1 darqn.sh SpaceInvaders-v0 2
sbatch --gres=gpu:1 darqn.sh Breakout-v0 2
# sbatch --gres=gpu:1 cead.sh Seaquest-v0 49
# sbatch --gres=gpu:1 cead.sh SpaceInvaders-v0 49
# sbatch --gres=gpu:1 cead.sh Breakout-v0 49
# sbatch --gres=gpu:1 darqn.sh Seaquest-v0 49
# sbatch --gres=gpu:1 darqn.sh SpaceInvaders-v0 49
# sbatch --gres=gpu:1 darqn.sh Breakout-v0 49
# sbatch --gres=gpu:1 cead.sh Seaquest-v0 256
# sbatch --gres=gpu:1 cead.sh SpaceInvaders-v0 256
# sbatch --gres=gpu:1 cead.sh Breakout-v0 256
# sbatch --gres=gpu:1 darqn.sh Seaquest-v0 256
# sbatch --gres=gpu:1 darqn.sh SpaceInvaders-v0 256
# sbatch --gres=gpu:1 darqn.sh Breakout-v0 256
