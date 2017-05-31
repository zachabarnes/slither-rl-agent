import sys
sys.path
sys.path.append('..')

import gym
import universe
import argparse

from utils.env import create_slither_env

from schedule import LinearExploration, LinearSchedule
from model import Model

from configs.RattLe import config

from six.moves import shlex_quote

parser = argparse.ArgumentParser(description="Run commands")

parser.add_argument('-r', '--remotes', default=None, help='Number of remotes to run')

parser.add_argument('-e', '--env-id', type=str, default="internet.SlitherIO-v0", help="Environment id")

parser.add_argument('-l', '--log-dir', type=str, default="/tmp/pong", help="Log directory path")

if __name__ == '__main__':
	FLAGS = parser.parse_args()
	# make env
	env = create_slither_env()
	env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})

	record_env = create_slither_env()
	gym.wrappers.Monitor(record_env, config.record_path, video_callable=lambda x: True, resume=True)
	record_env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})

	# exploration strategy
	exp_schedule = LinearExploration(env, config.eps_begin, config.eps_end, config.eps_nsteps)

	# learning rate schedule
	lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

	state_size = (74,124,3)
	# train model
	model = Model(env, record_env, config, "deep_q", state_size)
	model.run(exp_schedule, lr_schedule)