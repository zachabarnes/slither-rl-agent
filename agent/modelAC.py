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
from utils.replay_bufferAC import ReplayBufferAC

class Summary(object):
  def __init__(self):
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0

    self.avg_q = 0
    self.max_q = 0
    self.std_q = 0

    self.eval_reward = 0.

class ModelAC(object):
  def __init__(self, env, record_env, network, FLAGS, logger=None):
    # Directory for training outputs
    if not os.path.exists(FLAGS.output_path):
      os.makedirs(FLAGS.output_path)

    # Store hyper params
    self.FLAGS       = FLAGS
    self.env         = env
    self.record_env  = record_env
    self.network     = network
    self.summary     = Summary()

    # Setup Logger
    if logger is None: self.logger = get_logger(FLAGS.log_path)
    else:          self.logger = logger

    # Create network
    self.network.build()

  @property
  def policy(self):
    return lambda state: self.network.get_best_action(state)[0]

  def init_averages(self):
    self.summary.avg_reward = 0.
    self.summary.max_reward = 0.
    self.summary.std_reward = 0

    self.summary.avg_q = 0
    self.summary.max_q = 0
    self.summary.std_q = 0

    self.summary.eval_reward = 0.

  def update_averages(self, rewards, max_q_values, q_values, scores_eval):
    self.summary.avg_reward = np.mean(rewards)
    self.summary.max_reward = np.max(rewards)
    self.summary.std_reward = np.sqrt(np.var(rewards) / len(rewards))

    self.summary.max_q      = np.mean(max_q_values)
    self.summary.avg_q      = np.mean(q_values)
    self.summary.std_q      = np.sqrt(np.var(q_values) / len(q_values))

    if len(scores_eval) > 0: self.summary.eval_reward = scores_eval[-1]

  def update_logs(self, t, loss_eval, rewards, epsilon, grad_eval, lr):
    if len(rewards) > 0:
      self.prog.update (t + 1, exact=[("Loss", loss_eval), ("Avg R", self.summary.avg_reward),
            ("Max R", np.max(rewards)), ("eps", epsilon),
            ("Grads", grad_eval), ("Max Q", self.summary.max_q), ("lr", lr), ("actor", 1), ("critic", 0)])

  def update_logs2(self, t, loss_eval, rewards, epsilon, grad_eval, lr):
    if len(rewards) > 0:
      self.prog2.update (t + 1, exact=[("Loss", loss_eval), ("Avg R", self.summary.avg_reward),
            ("Max R", np.max(rewards)), ("eps", epsilon),
            ("Grads", grad_eval), ("Max Q", self.summary.max_q), ("lr", lr), ("critic", 1), ("actor", 0)])

  def train(self, exp_schedule, lr_schedule):
    # Initialize replay buffer and variables
    replay_buffer = ReplayBufferAC(self.FLAGS.buffer_size, self.FLAGS.state_hist)
    rewards = deque(maxlen=self.FLAGS.num_test)
    max_q_values = deque(maxlen=1000)
    q_values = deque(maxlen=1000)
    self.init_averages()

    t = 0 # time control of nb of steps
    loss_eval = grad_eval = 0
    scores_eval = [] # list of scores computed at iteration time
    #scores_eval += [self.evaluate(self.env, self.FLAGS.num_test)]

    self.prog = Progbar(target=self.FLAGS.train_steps)
    self.prog2 = Progbar(target=self.FLAGS.train_steps)

    # Train for # of train steps
    while t < self.FLAGS.train_steps:
      total_reward = 0
      ep_len = 0
      state = self.env.reset()
      reward = 0
      first = 1
      q_input = None
      # Run for 1 episode and update the buffer
      while True:
        ep_len += 1
        # replay memory stuff
        if first == 1:
          first = 0
          idx     = replay_buffer.store_frame(state)
          q_input = replay_buffer.encode_recent_observation()
        # chose action according to current Q and exploration
        best_action, q_values = self.network.get_best_action(q_input)
        action                = exp_schedule.get_action(best_action)
        orig_val = self.network.calcState(q_input)


        # store q values
        max_q_values.append(max(q_values))
        q_values += list(q_values)

        # perform action in env
        new_state, new_reward, done, info = self.env.step(action)
        idx = replay_buffer.store_frame(state)
        q_input = replay_buffer.encode_recent_observation()
        new_val = self.network.calcState(q_input)
        orig_val = orig_val[0][0]
        new_val = new_val[0][0]
        print (orig_val, new_reward, done, new_val, ep_len)


        if not done: # Non-terminal state.
          target = reward + ( self.FLAGS.gamma * new_val)
        else:
          target = reward + ( self.FLAGS.gamma * new_reward )

        best_val = max((orig_val), target)

        actor_delta = new_val - orig_val

        replay_buffer.store_effect(idx-1, action, new_reward, done, best_val, actor_delta)
        state = new_state

        if done:
          replay_buffer.store_effect(idx, action, 0, done, 0, 0)

        # Count reward
        total_reward += new_reward

        reward=new_reward

        # Stop at end of episode
        if done: break

      old_t = t
      temp_ep_len = ep_len
      while True:
        t += 1
        temp_ep_len -= 1

        if ((t > self.FLAGS.learn_start) and (t % self.FLAGS.learn_every == 0)):
          if replay_buffer.can_sample(self.FLAGS.batch_size) == True: 
            loss_eval, grad_eval = self.network.update_critic_step(t, replay_buffer, lr_schedule.epsilon, self.summary)


        # Update logs if necessary
        if ((t > self.FLAGS.learn_start) and (t % self.FLAGS.log_every == 0)):
          self.update_logs2(t, loss_eval, rewards, exp_schedule.epsilon, grad_eval, lr_schedule.epsilon)

        if temp_ep_len <= 0 or t >= self.FLAGS.train_steps: break


      rewards.append(total_reward)

      # Learn using replay
      while True:

        t += 1
        ep_len -= 1

        # Make train step if necessary
        if ((t > self.FLAGS.learn_start) and (t % self.FLAGS.learn_every == 0)):
          if replay_buffer.can_sample(self.FLAGS.batch_size) == True: 
            loss_eval, grad_eval = self.network.update_actor_step(t, replay_buffer, lr_schedule.epsilon, self.summary)
            exp_schedule.update(t)
            lr_schedule.update(t)

        # Update logs if necessary
        if ((t > self.FLAGS.learn_start) and (t % self.FLAGS.log_every == 0)):
          self.update_averages(rewards, max_q_values, q_values, scores_eval)
          self.update_logs(t, loss_eval, rewards, exp_schedule.epsilon, grad_eval, lr_schedule.epsilon)

        # Update logs if necessary
        elif (t < self.FLAGS.learn_start) and (t % self.FLAGS.log_every == 0):
          sys.stdout.write("\rPopulating the memory {}/{}...".format(t, self.FLAGS.learn_start))
          sys.stdout.flush()

        if ((t > self.FLAGS.learn_start) and (t % self.FLAGS.check_every == 0)):
          # Evaluate current model
          scores_eval += [self.evaluate(self.env, self.FLAGS.num_test)]

          # Save current Model
          self.network.save()

          # Record video of current model
          if self.FLAGS.record:
            self.record()

        if ep_len <= 0 or t >= self.FLAGS.train_steps: break

      # Update episodic rewards

    # End of training
    self.logger.info("- Training done.")
    self.network.save()
    scores_eval += [self.evaluate(self.env, self.FLAGS.num_test)]
    export_plot(scores_eval, "Scores", self.FLAGS.plot_path)

  def evaluate(self, env, num_episodes):
    replay_buffer = ReplayBufferAC(self.FLAGS.state_hist, self.FLAGS.state_hist)
    rewards = []

    if num_episodes > 1: self.logger.info("Evaluating...")

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
        replay_buffer.store_effect(idx, action, reward, done, None, None)
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
    self.logger.info("Recording...")
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
