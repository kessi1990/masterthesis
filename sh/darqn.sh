#!/bin/bash
#
#SBATCH --job-name darqn
#SBATCH --gres=gpu:0
env=$1
cells=$2
dir_id=$3
/home/stud/kesslermi/anaconda3/envs/masterthesis/bin/python /home/stud/kesslermi/masterthesis/run/run.py darqn $env $cells $dir_id