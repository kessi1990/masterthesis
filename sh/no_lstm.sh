#!/bin/bash
#
#SBATCH --job-name no_rec_gr
#SBATCH --gres=gpu:0
env=$1
dir_id=$2
hidden=$3
/home/k/kesslermi/anaconda3/envs/masterthesis/bin/python /home/k/kesslermi/Desktop/ma-test/masterthesis/run/run_no-lstm_grid.py $env $dir_id $hidden