#!/usr/bin/env bash
id=$1
configfile = "../config/test/test_conf_${id}.yaml"
~/PycharmProjects/astrepo/venv/bin/python /home/k/kesslermi/PycharmProjects/masterthesis/run/run.py --head cnn --config configfile