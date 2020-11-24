#!/usr/bin/env bash
sbatch --gres=gpu:0 no_lstm.sh small 30 4
sbatch --gres=gpu:0 no_lstm.sh small 30 8
sbatch --gres=gpu:0 no_lstm.sh large 30 4
sbatch --gres=gpu:0 no_lstm.sh large 30 8



