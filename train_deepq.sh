#!/bin/sh
export OPENAI_REMOTE_VERBOSE=0
ps -ef | grep tensorboard | grep -v grep | awk '{print $2}' | xargs kill
cd agent
rm -r nohup.out
rm -r results/deep_q/events.out.tfevents*
rm -r results/deep_q/log.txt
tensorboard --logdir=results/deep_q/ &
nohup python run.py -mod q&
