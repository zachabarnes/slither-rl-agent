#!/bin/sh
export OPENAI_REMOTE_VERBOSE=0
cd agent
python run.py -tst 1 -rec False -lst 10 -bat 1
