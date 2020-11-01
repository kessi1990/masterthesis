#!/bin/bash
#
#SBATCH --job-name darqn
#SBATCH --gres=gpu:0
env=$1
cells=$2
dir_id=$3
alignment=$4
hidden_size=$5
/home/k/kesslermi/anaconda3/envs/masterthesis/bin/python /home/k/kesslermi/Desktop/ma-test/masterthesis/run/run.py darqn $env $cells $dir_id