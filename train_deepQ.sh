#!/bin/sh

# Force remotes to be quiet
export OPENAI_REMOTE_VERBOSE=0
cd agent
#tensorboard --logdir=results/
python run.py