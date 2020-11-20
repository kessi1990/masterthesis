#!/bin/bash
#
#SBATCH --job-name base
#SBATCH --gres=gpu:0
fs=$1
env=$2
dir_id=$3
/home/k/kesslermi/anaconda3/envs/masterthesis/bin/python /home/k/kesslermi/Desktop/ma-test/masterthesis/run/run_dqn.py $fs $env $dir_id