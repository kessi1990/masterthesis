#!/usr/bin/env bash
sbatch --gres=gpu:0 darqn.sh Seaquest-v0 2 1
# sbatch --gres=gpu:0 darqn.sh Seaquest-v0 4 1
# sbatch --gres=gpu:0 darqn.sh Seaquest-v0 8 1

sbatch --gres=gpu:0 darqn.sh BeamRider-v0 2 1
# sbatch --gres=gpu:0 darqn.sh BeamRider-v0 4 1
# sbatch --gres=gpu:0 darqn.sh BeamRider-v0 8 1

# sbatch --gres=gpu:0 darqn.sh Breakout-v0 2 1
# sbatch --gres=gpu:0 darqn.sh Breakout-v0 4 1
# sbatch --gres=gpu:0 darqn.sh Breakout-v0 8 1