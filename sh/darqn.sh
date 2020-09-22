#!/bin/bash
#
#SBATCH --job-name darqn
#SBATCH --gres=gpu:1
env=$1
cells=$2
/home/stud/kesslermi/anaconda3/envs/masterthesis/bin/python /home/stud/kesslermi/masterthesis/run/run_2.py darqn $env $cells