import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import tensorflow as tf
import tensorflow.contrib.layers as layers

import numpy as np

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear

from configs.q6_bonus_question import config


class MyDQN(Linear):
    """
    Going beyond - implement your own Deep Q Network to find the perfect
    balance between depth, complexity, number of parameters, etc.
    You can change the way the q-values are computed, the exploration
    strategy, or the learning rate schedule. You can also create your own
    wrapper of environment and transform your input to something that you
    think we'll help to solve the task. Ideally, your network would run faster
    than DeepMind's and achieve similar performance!

    You can also change the optimizer (by overriding the functions defined
    in TFLinear), or even change the sampling strategy from the replay buffer.

    If you prefer not to build on the current architecture, you're welcome to
    write your own code.

    You may also try more recent approaches, like double Q learning
    (see https://arxiv.org/pdf/1509.06461.pdf) or dueling networks 
    (see https://arxiv.org/abs/1511.06581), but this would be for extra
    extra bonus points.
    """
    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        fd = {
        self.s: [state],
        self.is_training: False
        }
        action_values = self.sess.run(self.q, feed_dict=fd)[0]
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
            self.is_training: True,  
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

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        # here, typically, a state shape is (80, 80, 1)
        state_shape = list(self.env.observation_space.shape)

        img_height, img_width, nchannels = state_shape[0], state_shape[1], state_shape[2]
        self.s = tf.placeholder(tf.uint8, [None, img_height, img_width, nchannels*self.config.state_history])
        self.a = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.float32, [None])
        self.sp = tf.placeholder(tf.uint8, [None, img_height, img_width, nchannels*self.config.state_history])
        self.done_mask = tf.placeholder(tf.bool, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.is_training = tf.placeholder(tf.bool, [])

    def add_optimizer_op(self, scope):
        opt = tf.train.RMSPropOptimizer(self.lr)
        extra_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update):

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


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state
        ##############################################################

        def bn_conv_relu_drop(inputs, is_training, k = 3, s = 1, num_outputs=10, keep_prob = 1, scope = ""):
            out = layers.conv2d(inputs, num_outputs = num_outputs, kernel_size = [k,k], stride=[s,s], padding="SAME",activation_fn=None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope)
            out = layers.batch_norm(out, decay=.9, center=True, scale=True, epsilon=1e-3, activation_fn=None, is_training=is_training, scope=scope)
            if (keep_prob < 1):
                out = tf.dropout(out, keep_prob)
            return tf.nn.relu(out)

        def bn_fully_drop(inputs, is_training, num_outputs=10, keep_prob = 1, scope = ""):
            out = layers.fully_connected(inputs=inputs, num_outputs = num_outputs, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope)
            out = layers.batch_norm(out, decay=.9, center=True, scale=True, epsilon=1e-3, activation_fn=None, is_training=is_training, scope=scope)
            if (keep_prob < 1):
                out = tf.dropout(out, keep_prob)
            return tf.nn.relu(out)

        with tf.variable_scope(scope):
            out = bn_conv_relu_drop(inputs=state, is_training = self.is_training, k=8, s=4, num_outputs=32, keep_prob=1, scope = "conv1")
            out = bn_conv_relu_drop(inputs=out, is_training = self.is_training, k=6, s=3, num_outputs=32, keep_prob=1, scope = "conv2")
            out = bn_conv_relu_drop(inputs=out, is_training = self.is_training, k=4, s=2, num_outputs=64, keep_prob=1, scope = "conv3")
            out = bn_conv_relu_drop(inputs=out, is_training = self.is_training, k=3, s=1, num_outputs=64, keep_prob=1, scope = "conv4")
            out = layers.flatten(out, scope=scope)
            out = bn_fully_drop(inputs = out, is_training = self.is_training, num_outputs=512, keep_prob=1, scope="fc1")
            out = layers.fully_connected(inputs=out, num_outputs = num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope="fc2")

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use a different architecture for the Atari game. Please report the final result.
Feel free to change the configuration. If so, please report your hyperparameters.
"""


if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=config.overwrite_render)
    #env = EnvTest((80, 80, 1))

    # exploration strategy
    # you may want to modify this schedule
    exp_schedule = LinearExploration(env, config.eps_begin, config.eps_end, config.eps_nsteps)

    # you may want to modify this schedule
    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = MyDQN(env, config)
    model.run(exp_schedule, lr_schedule)
