#!/bin/sh
export OPENAI_REMOTE_VERBOSE=0
ps -ef | grep tensorboard | grep -v grep | awk '{print $2}' | xargs kill
cd agent
tensorboard --logdir=results/deep_q/ &
nohup python run.py -typ colors &
