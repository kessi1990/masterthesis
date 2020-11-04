#!/bin/bash
#
#SBATCH --job-name base
#SBATCH --gres=gpu:0
env=$1
cells=$2
dir_id=$3
/home/k/kesslermi/anaconda3/envs/masterthesis/bin/python /home/k/kesslermi/Desktop/ma-test/masterthesis/run/run_dqn.py dqn $env $cells $dir_id
