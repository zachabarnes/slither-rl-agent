import sys
sys.path
sys.path.append('..')
import os

import gym
import universe
import argparse
import network

from utils.env import create_slither_env
from universe.wrappers import Unvectorize

from schedule import LinearExploration, LinearSchedule
from modelAC import ModelAC
from model import Model

parser = argparse.ArgumentParser(description="Run commands")

# Model params
parser.add_argument('-net', '--network_type', type=str,  default="deep_q",               help="Network type (linear_q, feedforward_q, deep_q, deep_ac")
parser.add_argument('-mod', '--model_type',   type=str,  default="q",               help="Network type (q, ac)")
parser.add_argument('-typ', '--state_type',   type=str,  default="shapes",                help="State type (features, colors, shapes)")
parser.add_argument('-rem', '--remotes',      type=int,  default=1,                       help='Number of remotes to run')
parser.add_argument('-env', '--env-id',       type=str,  default="internet.SlitherIO-v0", help="Environment id")
parser.add_argument('-rec', '--record',       type=bool, default=False,                    help="Record videos during train")
parser.add_argument('-buf', '--buffer_size',  type=int,  default=50000,                   help="Size of replay buffer")

# Train Params
parser.add_argument('-trn', '--train_steps',  type=int,   default=200000,  help="Number of steps to train")
parser.add_argument('-tst', '--num_test',     type=int,   default=10,      help="Number of episodes to test model")
parser.add_argument('-bat', '--batch_size',   type=int,   default=32,      help="Batch_size")
parser.add_argument('-lnr', '--learning_rate',type=float, default=0.00025, help="Initial learning rate")
parser.add_argument('-eps', '--epsilon',      type=float, default=1.0,     help="Initial Exploration constant (e-greedy)")
parser.add_argument('-lst', '--learn_start',  type=int,   default=200,     help="Num ep before learning")


if __name__ == '__main__':
  FLAGS = parser.parse_args()

  # Set constants
  FLAGS.output_path  = "results/"      + FLAGS.network_type
  FLAGS.model_path   = FLAGS.output_path + "/model.weights/"
  FLAGS.log_path     = FLAGS.output_path + "/log.txt"
  FLAGS.plot_path    = FLAGS.output_path + "/scores.png"
  FLAGS.record_path  = FLAGS.output_path + "/monitor/"

  FLAGS.grad_clip    = True
  FLAGS.clip_val     = 10

  FLAGS.check_every  = FLAGS.train_steps/10
  FLAGS.log_every    = 500
  FLAGS.target_every = 800
  FLAGS.learn_every  = 1

  FLAGS.gamma        = 0.99
  FLAGS.lr_end       = 0.00005
  FLAGS.lr_nsteps    = FLAGS.train_steps/2
  FLAGS.eps_end      = 0.1
  FLAGS.eps_nsteps   = FLAGS.train_steps/2

  FLAGS.fps          = 5
  FLAGS.state_hist   = 4

  # Make rl environment
  universe.configure_logging(False)
  env = create_slither_env(FLAGS.state_type)

  FLAGS.state_size   = env.state_size
  FLAGS.high_val     = env.high_val
  FLAGS.num_actions  = env.action_space.n

  env = Unvectorize(env)
  env.configure(fps=FLAGS.fps, remotes=FLAGS.remotes, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})

  # Make recording env
  record_env = None
  if FLAGS.record:
    record_env = create_slither_env(FLAGS.state_type)
    record_env = Unvectorize(record_env)
    record_env.configure(fps=30, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})
    record_env = gym.wrappers.Monitor(record_env, FLAGS.record_path, video_callable=lambda x: True, resume=True)

  # Create network
  if FLAGS.network_type == 'linear_q':
    network = network.LinearQ(FLAGS)

  elif FLAGS.network_type == 'feedforward_q':
    network = network.FeedQ(FLAGS)

  elif FLAGS.network_type == 'deep_q':
    network = network.DeepQ(FLAGS)

  elif FLAGS.network_type == 'deep_ac':
    network = network.DeepAC(FLAGS)

  else: raise NotImplementedError

  # Initialize exploration strategy
  exp_schedule = LinearExploration(env, FLAGS.epsilon, FLAGS.eps_end, FLAGS.eps_nsteps)

  # Initialize exploration rate schedule
  lr_schedule  = LinearSchedule(FLAGS.learning_rate, FLAGS.lr_end, FLAGS.lr_nsteps)

  # train model
  model = None
  if FLAGS.model_type == 'q':
    model = Model(env, record_env, network, FLAGS)
  elif FLAGS.model_type == 'ac':
    model = ModelAC(env, record_env, network, FLAGS)
  else:
    raise NotImplementedError

  try:
    model.run(exp_schedule, lr_schedule)
  except Exception as e:
    print(e)
    network.save()
