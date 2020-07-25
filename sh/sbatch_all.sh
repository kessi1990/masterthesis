#!/usr/bin/env bash
for i in `seq 0 35`;
    do
        sbatch run.sh $i
    done