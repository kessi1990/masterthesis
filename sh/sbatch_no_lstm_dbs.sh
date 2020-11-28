#!/usr/bin/env bash
sbatch --gres=gpu:0 no_lstm_dbs.sh small 30 4 identity concat
sbatch --gres=gpu:0 no_lstm_dbs.sh small 30 4 identity general
sbatch --gres=gpu:0 no_lstm_dbs.sh large 30 4 identity concat
sbatch --gres=gpu:0 no_lstm_dbs.sh large 30 4 identity general

sbatch --gres=gpu:0 no_lstm_dbs.sh small 30 4 no-lstm concat
sbatch --gres=gpu:0 no_lstm_dbs.sh small 30 4 no-lstm general
sbatch --gres=gpu:0 no_lstm_dbs.sh small 30 8 no-lstm concat
sbatch --gres=gpu:0 no_lstm_dbs.sh small 30 8 no-lstm general
sbatch --gres=gpu:0 no_lstm_dbs.sh large 30 4 no-lstm concat
sbatch --gres=gpu:0 no_lstm_dbs.sh large 30 4 no-lstm general
sbatch --gres=gpu:0 no_lstm_dbs.sh large 30 8 no-lstm concat
sbatch --gres=gpu:0 no_lstm_dbs.sh large 30 8 no-lstm general