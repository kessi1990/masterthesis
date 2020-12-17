#!/bin/bash
#
#SBATCH --job-name norecgr
#SBATCH --gres=gpu:0
framestack=$1
model=$2
hidden=$3
alignment=$4
env=$5
random_start=$6
dir_id=$7
/home/stud/kesslermi/anaconda3/envs/masterthesis/bin/python /home/stud/kesslermi/masterthesis/run/run_grid.py $framestack $model $hidden $alignment $env $random_start $dir_id