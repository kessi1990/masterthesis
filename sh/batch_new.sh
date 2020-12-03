#!/bin/bash
#
#SBATCH --job-name recnofs
#SBATCH --gres=gpu:0
fs=$1
model=$2
alignment=$3
env=$4
dir_id=$5
/home/k/kesslermi/anaconda3/envs/masterthesis/bin/python /home/k/kesslermi/Desktop/ma-test/masterthesis/run/run_new.py $fs $model $alignment $env $dir_id