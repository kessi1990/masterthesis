#!/bin/bash
#
#SBATCH --gres=gpu:1
id=$1
configfile=../config/test/test_conf_${id}.yaml
/home/stud/kesslermi/anaconda3/envs/masterthesis/bin/python /home/stud/kesslermi/masterthesis/run/run_old.py --config $configfile