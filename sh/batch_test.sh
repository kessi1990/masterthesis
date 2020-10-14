#!/usr/bin/env bash
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_darqn.sh Breakout-v0 4 True
sbatch --partition=Luna --cpus-per-task=4 --gres=gpu:0 --exclusive batch_cead.sh Breakout-v0 4 True
