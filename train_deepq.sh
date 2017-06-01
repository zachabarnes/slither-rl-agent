#!/bin/sh
export OPENAI_REMOTE_VERBOSE=0
ps -ef | grep tensorboard | grep -v grep | awk '{print $2}' | xargs kill
tensorboard --logdir=results/ &
cd agent
nohup python run.py &
