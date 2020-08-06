#!/usr/bin/env bash
for i in `seq 0 39`;
    do
        sbatch --gres=gpu:1 run.sh $i
    done