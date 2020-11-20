#!/usr/bin/env bash
sbatch --gres=gpu:0 dqn_grid_dbs.sh False small 21
sbatch --gres=gpu:0 dqn_grid_dbs.sh True small 21
# sbatch --gres=gpu:0 dqn_grid_dbs.sh False large 21
# sbatch --gres=gpu:0 dqn_grid_dbs.sh True large 21

sbatch --gres=gpu:0 dqn_dbs.sh False Pong-v0 21
sbatch --gres=gpu:0 dqn_dbs.sh True Pong-v0 21

sbatch --gres=gpu:0 dqn_dbs.sh False Seaquest-v0 21
sbatch --gres=gpu:0 dqn_dbs.sh True Seaquest-v0 21

sbatch --gres=gpu:0 dqn_dbs.sh False Breakout-v0 21
sbatch --gres=gpu:0 dqn_dbs.sh True Breakout-v0 21

sbatch --gres=gpu:0 dqn_dbs.sh False BeamRider-v0 21
sbatch --gres=gpu:0 dqn_dbs.sh True BeamRider-v0 21
