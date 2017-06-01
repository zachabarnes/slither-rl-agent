import os
import sys
import time
import logging

import gym
import universe

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from gym import wrappers
from collections import deque

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer

class Model(object):
  def __init__(self, env, record_env, network, FLAGS, logger=None):
    # Directory for training outputs
    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)

    # Store hyper params
    self.FLAGS     = FLAGS
    self.env     = env
    self.record_env  = record_env
    self.network     = network

    # Setup Logger
    if logger is None: self.logger = get_logger(FLAGS.log_path)
    else:          self.logger = logger

    # Create network
    self.network.build()

  @property
  def policy(self):
    return lambda state: self.network.get_best_action(state)[0]

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

    if len(scores_eval) > 0: self.eval_reward = scores_eval[-1]

  def update_logs(self, t, loss_eval, rewards, epsilon, grad_eval, lr):
    if len(rewards) > 0:
      prog.update (t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward),
            ("Max R", np.max(rewards)), ("eps", epsilon),
            ("Grads", grad_eval), ("Max Q", self.max_q), ("lr", lr)])

  def train(self, exp_schedule, lr_schedule):
    # Initialize replay buffer and variables
    replay_buffer = ReplayBuffer(self.FLAGS.buffer_size, self.FLAGS.state_hist)
    rewards = deque(maxlen=self.FLAGS.num_test)
    max_q_values = deque(maxlen=1000)
    q_values = deque(maxlen=1000)
    self.init_averages()

    t = 0 # time control of nb of steps
    loss_eval = grad_eval = 0
    scores_eval = [] # list of scores computed at iteration time
    scores_eval += [self.evaluate(self.env, self.FLAGS.num_test)]

    prog = Progbar(target=self.FLAGS.train_steps)

    # Train for # of train steps
    while t < self.FLAGS.train_steps:
      total_reward = 0
      ep_len = 0
      state = self.env.reset()

      # Run for 1 episode and update the buffer
      while True:
        ep_len += 1

        # replay memory stuff
        idx     = replay_buffer.store_frame(state)
        q_input = replay_buffer.encode_recent_observation()

        # chose action according to current Q and exploration
        best_action, q_values = self.network.get_best_action(q_input)
        action                = exp_schedule.get_action(best_action)

        # store q values
        max_q_values.append(max(q_values))
        q_values += list(q_values)

        # perform action in env
        new_state, reward, done, info = self.env.step(action)

        # store the transition
        replay_buffer.store_effect(idx, action, reward, done)
        state = new_state

        # Count reward
        total_reward += reward

        # Stop at end of episode
        if done: break

      # Learn using replay
      while True:
        t += 1
        ep_len -= 1

        # Make train step if necessary
        if ((t > self.FLAGS.learning_start) and (t % self.FLAGS.learn_every == 0)):
          loss_eval, grad_eval = self.network.update_step(t, replay_buffer, lr)
          exp_schedule.update(t)
          lr_schedule.update(t)

        if (t % self.FLAGS.target_every == 0):
          self.network.update_target_params()

        # Update logs if necessary
        if ((t > self.FLAGS.learning_start) and (t % self.FLAGS.log_every == 0)):
          self.update_averages(rewards, max_q_values, q_values, scores_eval)
          self.update_logs(t, loss_eval, rewards, exp_schedule.epsilon, grad_eval, lr_schedule.epsilon)

        # Update logs if necessary
        elif (t < self.FLAGS.learning_start) and (t % self.FLAGS.log_every == 0):
          sys.stdout.write("\rPopulating the memory {}/{}...".format(t, self.FLAGS.learning_start))
          sys.stdout.flush()

        if ((t > self.FLAGS.learning_start) and (t % self.FLAGS.check_every == 0)):
          # Evaluate current model
          self.logger.info("Evaluating...")
          scores_eval += [self.evaluate(self.env, self.FLAGS.num_test)]

          # Save current Model
          self.network.save()

          # Record video of current model
          if self.FLAGS.record:
            self.logger.info("Recording...")
            self.record()

        if ep_len <= 0 or t >= self.FLAGS.train_steps: break

      # Update episodic rewards
      rewards.append(total_reward)

    # End of training
    self.logger.info("- Training done.")
    self.network.save()
    scores_eval += [self.evaluate(self.env, self.FLAGS.num_test)]
    export_plot(scores_eval, "Scores", self.FLAGS.plot_path)

  def evaluate(self, env=self.env, num_episodes=self.FLAGS.num_test):
    replay_buffer = ReplayBuffer(self.FLAGS.state_hist, self.FLAGS.state_hist)
    rewards = []

    for i in range(num_episodes):
      total_reward = 0
      state = env.reset()
      while True:
        # Store last state in buffer
        idx     = replay_buffer.store_frame(state)
        q_input = replay_buffer.encode_recent_observation()

        # Get greedy action
        action = self.network.get_best_action(q_input)[0]

        # Perform action in env
        new_state, reward, done, info = env.step(action)

        # Store in replay memory
        replay_buffer.store_effect(idx, action, reward, done)
        state = new_state

        # count reward
        total_reward += reward
        if done: break

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
    # Initialize network session
    self.network.initialize()

    # Record one game at the beginning
    if self.FLAGS.record: self.record()

    # Train model
    self.train(exp_schedule, lr_schedule)

    # Record one game at the end
    if self.FLAGS.record: self.record()
