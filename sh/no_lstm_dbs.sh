#!/bin/bash
#
#SBATCH --job-name norecgr
#SBATCH --gres=gpu:0
env=$1
dir_id=$2
hidden=$3
/home/stud/kesslermi/anaconda3/envs/masterthesis/bin/python /home/stud/kesslermi/masterthesis/run/run_no-lstm_grid.py $env $dir_id $hidden