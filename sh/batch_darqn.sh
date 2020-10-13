#!/bin/bash
#
#SBATCH --job-name db4b
#SBATCH --gres=gpu:0
env=$1
cells=$2
batch=$3
/home/k/kesslermi/anaconda3/envs/masterthesis/bin/python /home/k/kesslermi/Desktop/ma-test/masterthesis/run/run.py darqn $env $cells $batch