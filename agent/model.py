import os
import sys
import time
import logging

import gym
import universe

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.env import create_slither_env

from gym import wrappers
from collections import deque

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer

import network

class Model(object):
	def __init__(self, env, record_env, config, network_name, state_size, logger=None):
		# directory for training outputs
		if not os.path.exists(config.output_path):
			os.makedirs(config.output_path)
			
		# store hyper params
		self.config = config
		self.logger = logger
		if logger is None:
			self.logger = get_logger(config.log_path)
		self.env = env
		self.record_env = record_env
		self.num_actions = self.env.action_space.n
		self.state_size = state_size

		# Create network
		if network_name == 'linear_q':
			self.network = network.LinearQ(config, self.num_actions, self.state_size)
		elif network_name == 'feedforward_q':
			self.network = network.FeedQ(config, self.num_actions, self.state_size)
		elif network_name == 'deep_q':
			self.network = network.DeepQ(config, self.num_actions, self.state_size)
		elif network_name == 'deep_ac':
			self.network = network.DeepAC(config, self.num_actions, self.state_size)
		else:
			raise NotImplementedError()

		self.network.build()

	@property
	def policy(self):
		return lambda state: self.get_action(state)

	def get_action(self, state):
		if np.random.random() < self.config.soft_epsilon:
			return self.env.action_space.sample()
		else:
			return self.network.get_best_action(state)[0]

	def init_averages(self):
		self.avg_reward = 0.
		self.max_reward = 0.
		self.std_reward = 0

		self.avg_q = 0
		self.max_q = 0
		self.std_q = 0
		
		self.eval_reward = 0.

	def update_averages(self, rewards, max_q_values, q_values, scores_eval):
		self.avg_reward = np.mean(rewards)
		self.max_reward = np.max(rewards)
		self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

		self.max_q      = np.mean(max_q_values)
		self.avg_q      = np.mean(q_values)
		self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

		if len(scores_eval) > 0:
			self.eval_reward = scores_eval[-1]


	def train(self, exp_schedule, lr_schedule):

		# initialize replay buffer and variables
		replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
		rewards = deque(maxlen=self.config.num_episodes_test)
		max_q_values = deque(maxlen=1000)
		q_values = deque(maxlen=1000)
		self.init_averages()

		t = last_eval = last_record = 0 # time control of nb of steps
		scores_eval = [] # list of scores computed at iteration time
		scores_eval += [self.evaluate()]
		
		prog = Progbar(target=self.config.nsteps_train)

		# interact with environment
		while t < self.config.nsteps_train:
			total_reward = 0
			state = self.env.reset()
			while True:
				t += 1
				last_eval += 1
				last_record += 1
				if self.config.render_train: self.env.render()
				# replay memory stuff
				idx     = replay_buffer.store_frame(state)
				q_input = replay_buffer.encode_recent_observation()

				# chose action according to current Q and exploration
				best_action, q_values = self.get_best_action(q_input)
				action                = exp_schedule.get_action(best_action)

				# store q values
				max_q_values.append(max(q_values))
				q_values += list(q_values)

				# perform action in env
				new_state, reward, done, info = self.env.step(action)

				# store the transition
				replay_buffer.store_effect(idx, action, reward, done)
				state = new_state

				# perform a training step
				loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

				# logging stuff
				if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
				   (t % self.config.learning_freq == 0)):
					self.update_averages(rewards, max_q_values, q_values, scores_eval)
					exp_schedule.update(t)
					lr_schedule.update(t)
					if len(rewards) > 0:
						prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward), 
										("Max R", np.max(rewards)), ("eps", exp_schedule.epsilon), 
										("Grads", grad_eval), ("Max Q", self.max_q), 
										("lr", lr_schedule.epsilon)])

				elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
					sys.stdout.write("\rPopulating the memory {}/{}...".format(t, 
														self.config.learning_start))
					sys.stdout.flush()

				# count reward
				total_reward += reward
				if done or t >= self.config.nsteps_train:
					break

			# updates to perform at the end of an episode
			rewards.append(total_reward)          

			if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
				# evaluate our policy
				last_eval = 0
				print("")
				scores_eval += [self.evaluate()]

			if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
				self.logger.info("Recording...")
				last_record =0
				self.record()

		# last words
		self.logger.info("- Training done.")
		self.network.save()
		scores_eval += [self.evaluate()]
		export_plot(scores_eval, "Scores", self.config.plot_output)

	def train_step(self, t, replay_buffer, lr):
		loss_eval, grad_eval = 0, 0

		# perform training step
		if (t > self.config.learning_start and t % self.config.learning_freq == 0):
			loss_eval, grad_eval = self.network.update_step(t, replay_buffer, lr)

		# occasionaly update target network with q network
		if t % self.config.target_update_freq == 0:
			self.network.update_target_params()
			
		# occasionaly save the weights
		if (t % self.config.saving_freq == 0):
			self.network.save()

		return loss_eval, grad_eval

	def evaluate(self, env=None, num_episodes=None):
		# log our activity only if default call
		if num_episodes is None:
			self.logger.info("Evaluating...")

		# arguments defaults
		if num_episodes is None:
			num_episodes = self.config.num_episodes_test

		if env is None:
			env = self.env

		# replay memory to play
		replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
		rewards = []

		for i in range(num_episodes):
			total_reward = 0
			state = env.reset()
			while True:
				if self.config.render_test: env.render()

				# store last state in buffer
				idx     = replay_buffer.store_frame(state)
				q_input = replay_buffer.encode_recent_observation()

				tic = time.clock()
				action = self.get_action(q_input)
				toc = time.clock()
				#self.logger.info("Time for actions" + str(tic-toc))

				# perform action in env
				new_state, reward, done, info = env.step(action)
				#self.logger.info(info)

				# store in replay memory
				replay_buffer.store_effect(idx, action, reward, done)
				state = new_state

				# count reward
				total_reward += reward
				if done:
					break

			# updates to perform at the end of an episode
			rewards.append(total_reward)     

		avg_reward = np.mean(rewards)
		sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

		if num_episodes > 1:
			msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
			self.logger.info(msg)

		return avg_reward

	def record(self):
		self.evaluate(self.record_env, 1)

	def run(self, exp_schedule, lr_schedule):
		# initialize
		self.network.initialize()

		# record one game at the beginning
		if self.config.record:
			self.record()

		# model
		self.train(exp_schedule, lr_schedule)

		# record one game at the end
		if self.config.record:
			self.record()
		
