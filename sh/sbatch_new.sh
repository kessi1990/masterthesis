#!/usr/bin/env bash
sbatch --gres=gpu:1 --output=cead_seaquest_1.txt cead.sh Seaquest-v0 1
sbatch --gres=gpu:1 --output=cead_spaceinvaders_1.txt cead.sh SpaceInvaders-v0 1
sbatch --gres=gpu:1 --output=cead_breakout_1.txt cead.sh Breakout-v0 1
sbatch --gres=gpu:1 --output=darqn_seaquest_1.txt darqn.sh Seaquest-v0 1
sbatch --gres=gpu:1 --output=darqn_spaceinvaders_1.txt darqn.sh SpaceInvaders-v0 1
sbatch --gres=gpu:1 --output=darqn_breakout_1.txt darqn.sh Breakout-v0 1
sbatch --gres=gpu:1 --output=cead_seaquest_49.txt cead.sh Seaquest-v0 49
sbatch --gres=gpu:1 --output=cead_spaceinvaders_49.txt cead.sh SpaceInvaders-v0 49
sbatch --gres=gpu:1 --output=cead_breakout_49.txt cead.sh Breakout-v0 49
sbatch --gres=gpu:1 --output=darqn_seaquest_49.txt darqn.sh Seaquest-v0 49
sbatch --gres=gpu:1 --output=darqn_spaceinvaders_49.txt darqn.sh SpaceInvaders-v0 49
sbatch --gres=gpu:1 --output=darqn_breakout_49.txt darqn.sh Breakout-v0 49
# sbatch --gres=gpu:1 --output=cead_seaquest_256.txt cead.sh Seaquest-v0 256
# sbatch --gres=gpu:1 --output=cead_spaceinvaders_256.txt cead.sh SpaceInvaders-v0 256
# sbatch --gres=gpu:1 --output=cead_breakout_256.txt cead.sh Breakout-v0 256
# sbatch --gres=gpu:1 --output=darqn_seaquest_256.txt darqn.sh Seaquest-v0 256
# sbatch --gres=gpu:1 --output=darqn_spaceinvaders_256.txt darqn.sh SpaceInvaders-v0 256
# sbatch --gres=gpu:1 --output=darqn_breakout_256.txt darqn.sh Breakout-v0 256
