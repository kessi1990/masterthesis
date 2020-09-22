#!/bin/bash
#
#SBATCH --job-name darqn
#SBATCH --gres=gpu:0
env=$1
cells=$2
/home/k/kesslermi/anaconda3/envs/masterthesis/bin/python /home/k/kesslermi/PycharmProjects/masterthesis/run/run_2.py $env darqn $cells