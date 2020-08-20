#!/bin/bash
#
#SBATCH --gres=gpu:0
id=$1
configfile=../config/test/test_conf_${id}.yaml
/home/k/kesslermi/anaconda3/envs/masterthesis/bin/python /home/k/kesslermi/PycharmProjects/masterthesis/run/run.py --config $configfile