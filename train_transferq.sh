#!/bin/sh
export OPENAI_REMOTE_VERBOSE=0
ps -ef | grep tensorboard | grep -v grep | awk '{print $2}' | xargs kill
cd agent
rm -r nohup.out
rm -r results/transfer_q/events.out.tfevents*
rm -r results/transfer_q/log.txt
tensorboard --logdir=results/transfer_q/ &
nohup python run.py --network_type transfer_q --model_type q   --state_type transfer --buffer_size 10000 &
