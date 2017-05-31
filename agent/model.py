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

class DeepQ(object):
    """
    Class 
    """
    def __init__(self, env, record_env, config, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
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

        # build model
        self.build()

    def add_placeholders_op(self):
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, nchannels = 74, 124, 3
        self.s = tf.placeholder(tf.uint8, [None, img_height, img_width, nchannels*self.config.state_history])
        self.a = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.float32, [None])
        self.sp = tf.placeholder(tf.uint8, [None, img_height, img_width, nchannels*self.config.state_history])
        self.done_mask = tf.placeholder(tf.bool, [None])
        self.lr = tf.placeholder(tf.float32, [])


    def get_q_values_op(self, state, scope, reuse=False):
        """
        set Q values, of shape = (batch_size, num_actions)
        """
        num_actions = self.num_actions
        out = state
        with tf.variable_scope(scope):
        #CONV: 32 filters of 8 x 8 with stride 4 -> RELU
            out = layers.conv2d(inputs=state, num_outputs = 32, kernel_size=[8,8], stride=[4,4], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"1")

            #CONV: 64 filters of 4 x 4 with stride 2 -> RELU
            out = layers.conv2d(inputs=out, num_outputs = 64, kernel_size=[4,4], stride=[2,2], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"2")

            #CONV: 64 filters of 3 x 3 with stride 1 -> RELU
            out = layers.conv2d(inputs=out, num_outputs = 64, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"3")
            #Fully Connected (512) -> RELU
            out = layers.flatten(out, scope=scope)

            out = layers.fully_connected(inputs=out, num_outputs = 512, activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"4")

            #Fully Connected (num_actions) no activation
            out = layers.fully_connected(inputs=out, num_outputs = num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"5")

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        Update_target_op will be called periodically 
        to copy Q network to target Q network
    
        Args:
            q_scope: name of the scope of variables for q
            target_q_scope: name of the scope of variables for the target
                network
        """
        q_col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)

        assn_vec = []
        for i,var in enumerate(q_col):
            assign = tf.assign(target_q_col[i], q_col[i])
            assn_vec.append(assign)

        self.update_target_op = tf.group(*assn_vec)


    def add_loss_op(self, q, target_q):
        """
        Set (Q_target - Q)^2
        """
        num_actions = self.num_actions 
        q_samp = self.r + self.config.gamma*(-tf.to_float(self.done_mask)+1.0)*tf.reduce_max(target_q,axis=1)
        q_s = tf.reduce_sum(q * tf.one_hot(self.a, num_actions), axis=1)
        losses = tf.square(q_samp-q_s)
        self.loss = tf.reduce_mean(losses)


    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
        opt = tf.train.AdamOptimizer(self.lr)

        grads_and_vars = opt.compute_gradients(self.loss, var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))

        clipped_grads_and_vars=[]
        clipped_grads_list=[]
        if (self.config.grad_clip):
            for grads,var in grads_and_vars:
                clipped_grads = tf.clip_by_norm(grads, self.config.clip_val)
                clipped_grads_and_vars.append((clipped_grads,var))
                clipped_grads_list.append(clipped_grads)

            self.train_op = opt.apply_gradients(clipped_grads_and_vars)
            self.grad_norm = tf.global_norm(clipped_grads_list)
        else:
            self.train_op = opt.apply_gradients(grads_and_vars)
            self.grad_norm = tf.global_norm([grads for grads, _ in grads_and_vars])


    def build(self):
        """
        Build model
        """
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.process_state(self.s)
        self.q = self.get_q_values_op(s, scope="q", reuse=False)

        # compute Q values of next state
        sp = self.process_state(self.sp)
        self.target_q = self.get_q_values_op(sp, scope="target_q", reuse=False)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q, self.target_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")


    @property
    def policy(self):
        """
        model.policy(state) = action
        """
        return lambda state: self.get_action(state)

    def initialize(self):
        """
        Initialize variables if necessary
        """
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()

    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg Q", self.avg_q_placeholder)
        tf.summary.scalar("Max Q", self.max_q_placeholder)
        tf.summary.scalar("Std Q", self.std_q_placeholder)

        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)
            
        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, 
                                                self.sess.graph)

    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state

    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)

    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)


        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch, 
            self.done_mask: done_mask_batch,
            self.lr: lr, 
            # extra info
            self.avg_reward_placeholder: self.avg_reward, 
            self.max_reward_placeholder: self.max_reward, 
            self.std_reward_placeholder: self.std_reward, 
            self.avg_q_placeholder: self.avg_q, 
            self.max_q_placeholder: self.max_q, 
            self.std_q_placeholder: self.std_q, 
            self.eval_reward_placeholder: self.eval_reward, 
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm, 
                                                 self.merged, self.train_op], feed_dict=fd)


        # tensorboard stuff
        self.file_writer.add_summary(summary, t)
        
        return loss_eval, grad_norm_eval


    def get_action(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)


    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = 0.
        self.max_reward = 0.
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0
        
        self.eval_reward = 0.


    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q      = np.mean(max_q_values)
        self.avg_q      = np.mean(q_values)
        self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]


    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

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
                idx      = replay_buffer.store_frame(state)
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
        self.save()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)


    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()
            
        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval


    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if env is self.record_env:
            self.logger.info("Recording video...")

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
        """
        Re create an env and record a video for one episode
        """
        #env = create_slither_env()
        #env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        #env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})
        
        self.evaluate(self.record_env, 1)


    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # record one game at the beginning
        if self.config.record:
            self.record()

        # model
        self.train(exp_schedule, lr_schedule)

        # record one game at the end
        if self.config.record:
            self.record()
        
