import sys
sys.path
sys.path.append('..')
import os

import gym
import universe
import argparse
import network 

from utils.env import create_slither_env

from schedule import LinearExploration, LinearSchedule
from model import Model

parser = argparse.ArgumentParser(description="Run commands")

# Model params
parser.add_argument('-net', '--network_name',type=str, default="deep_q", help="The network to train")
parser.add_argument('-rem', '--remotes',type=int, default=1, help='Number of remotes to run')
parser.add_argument('-env', '--env-id', type=str, default="internet.SlitherIO-v0", help="Environment id")
parser.add_argument('-log', '--log-dir', type=str, default="/tmp/pong", help="Log directory path")
parser.add_argument('-rec', '--record', type=bool, default=True, help="Record videos during train")
parser.add_argument('-buf', '--buffer_size', type=int, default=50000, help="Size of replay buffer")

# Train Params
parser.add_argument('-trn', '--train_steps', type=int, default=200000, help="Number of steps to train")
parser.add_argument('-evl', '--num_test', type=int, default=10, help="Number of episodes to test model")
parser.add_argument('-bat', '--batch_size', type=int, default=32, help="Batch_size")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.00025, help="Initial learning rate")
parser.add_argument('-eps', '--epsilon', type=float, default=1.0, help="Initial Exploration constant (e-greedy)")


if __name__ == '__main__':
	FLAGS = parser.parse_args()

	# Set constants
	FLAGS.output_dir   = "/results/"      + FLAGS.network_name
	FLAGS.model_path   = FLAGS.output_dir + "/model.weights/"
	FLAGS.log_path     = FLAGS.output_dir + "/log.txt"
	FLAGS.plot_path    = FLAGS.output_dir + "/scores.png"
	FLAGS.record_path  = FLAGS.output_dir + "/monitor/"

	FLAGS.grad_clip    = True
	FLAGS.clip_val     = 10

	FLAGS.check_every  = FLAGS.train_steps/20
	FLAGS.log_every    = 200
	FLAGS.target_every = 1000
	FLAGS.learn_every  = 4
	FLAGS.learn_start  = 500

	FLAGS.gamma        = 0.99
	FLAGS.lr_end       = 0.00005
	FLAGS.lr_nsteps    = FLAGS.train_steps/2
	FLAGS.eps_end      = 0.1
	FLAGS.eps_nsteps   = FLAGS.train_steps

	FLAGS.fps  		   = 5
	FLAGS.state_hist   = 2
	FLAGS.high_val     = 1.0

	feature_size = [4,1,1]
	color_size   = [74,124,3]
	shape_size   = [74,124,1]

	if FLAGS.network_name == 'linear_q':
		FLAGS.state 	 = 'features'
		FLAGS.state_size = feature_size
		network = network.LinearQ(FLAGS)

	elif FLAGS.network_name == 'feedforward_q':
		FLAGS.state 	 = 'colors'
		FLAGS.state_size = color_size
		network = network.FeedQ(FLAGS)

	elif FLAGS.network_name == 'deep_q':
		FLAGS.state 	 = 'entities'
		FLAGS.state_size = shape_size
		network = network.DeepQ(FLAGS)

	elif FLAGS.network_name == 'deep_ac':
		FLAGS.state 	 = 'entities'
		FLAGS.state_size = shape_size
		network = network.DeepAC(FLAGS)

	else:
		raise NotImplementedError()

	# Command line setup
	os.system("export OPENAI_REMOTE_VERBOSE=0")
	#os.system("tensorboard --logdir=results/")

	# Make rl environment
	env = create_slither_env(FLAGS.state, FLAGS.state_size)
	env.configure(fps=FLAGS.fps, remotes=FLAGS.remotes, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})

	# Make recording env
	record_env = create_slither_env(FLAGS.state, FLAGS.state_size)
	gym.wrappers.Monitor(record_env, FLAGS.record_path, video_callable=lambda x: True, resume=True)
	record_env.configure(fps=30, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})

	# Initialize exploration strategy
	exp_schedule = LinearExploration(env, FLAGS.epsilon, FLAGS.eps_end, FLAGS.eps_nsteps)

	# Initialize exploration rate schedule
	lr_schedule  = LinearSchedule(FLAGS.learning_rate, FLAGS.lr_end, FLAGS.lr_nsteps)

	# train model
	model = Model(env, record_env, network, FLAGS)
	model.run(exp_schedule, lr_schedule)




