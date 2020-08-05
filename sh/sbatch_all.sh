#!/usr/bin/env bash
for i in `seq 0 39`;
    do
        sbatch run.sh $i
    done