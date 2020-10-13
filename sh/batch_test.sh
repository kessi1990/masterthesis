#!/usr/bin/env bash
sbatch --partition=Gobi --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 4 True
sbatch --partition=Gobi --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 4 False
sbatch --partition=Gobi --gres=gpu:0 --exclusive batch_cead.sh Breakout-v0 4 True
sbatch --partition=Gobi --gres=gpu:0 --exclusive batch_cead.sh Breakout-v0 4 False