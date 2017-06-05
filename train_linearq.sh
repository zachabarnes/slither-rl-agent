#!/bin/sh
export OPENAI_REMOTE_VERBOSE=0
ps -ef | grep tensorboard | grep -v grep | awk '{print $2}' | xargs kill
cd agent
rm -r nohup.out
rm -r results/linear_q/events.out.tfevents*
rm -r results/linear_q/log.txt
tensorboard --logdir=results/linear_q/ &
nohup python run.py -net linear_q -mod q -typ features&
