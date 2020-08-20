#!/usr/bin/env bash
for i in `seq 0 29`;
    do
        sbatch --partition=All --gres=gpu:0 --exclusive run_c.sh $i
    done