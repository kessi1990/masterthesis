#!/usr/bin/env bash
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive no_lstm.sh small 30 4
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive no_lstm.sh small 30 8
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive no_lstm.sh large 30 4
sbatch --partition=Gobi,Luna --gres=gpu:0 --exclusive no_lstm.sh large 30 8



