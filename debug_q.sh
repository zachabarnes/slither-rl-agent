#!/bin/sh
export OPENAI_REMOTE_VERBOSE=0
#tensorboard --logdir=results/
cd agent
python run.py -tst 2
