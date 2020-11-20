#!/bin/bash
#
#SBATCH --job-name base_gw
#SBATCH --gres=gpu:0
fs=$1
env=$2
dir_id=$3
/home/stud/kesslermi/anaconda3/envs/masterthesis/bin/python /home/stud/kesslermi/masterthesis/run/run_dqn_grid.py $fs $env $dir_id