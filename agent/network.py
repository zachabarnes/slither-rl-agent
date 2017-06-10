import os

import gym
import universe

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

class Network(object):
  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.num_actions = FLAGS.num_actions
    self.img_height, self.img_width, self.img_depth = FLAGS.state_size

  def build(self):
    self.scope = "scope"
    self.target_scope = "target_scope"

    self.add_placeholders_op()

    self.proc_s = self.process_state(self.s)
    self.proc_sp = self.process_state(self.sp)

    self.add_loss_op()

    self.add_update_target_op()

    self.add_optimizer_op()

  def record_initialize(self, checkpoint_path):
    self.sess = tf.Session()
    self.saver = tf.train.Saver(max_to_keep=2)
    self.load(checkpoint_path)

  def initialize(self):
    self.sess = tf.Session()

    # tensorboard stuff
    self.add_summary()

    # initiliaze all variables
    self.sess.run(tf.global_variables_initializer())

    # synchronise q and target_q network
    self.update_target_params()

    # for saving network weights
    self.saver = tf.train.Saver(max_to_keep=2)

  def get_best_action(self, state):
    action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
    return np.argmax(action_values), action_values

  def update_target_params(self):
    self.sess.run(self.update_target_op)

  def save(self):
    if not os.path.exists(self.FLAGS.model_path):
      os.makedirs(self.FLAGS.model_path)
    self.saver.save(self.sess, self.FLAGS.model_path)

  def load(self, checkpoint_path):
    assert os.path.exists(checkpoint_path)
    self.saver.restore(self.sess, checkpoint_path)

  def add_placeholders_op(self):
    self.s = tf.placeholder(tf.uint8, [None, self.img_height, self.img_width, self.img_depth*self.FLAGS.state_hist])
    self.a = tf.placeholder(tf.int32, [None])
    self.r = tf.placeholder(tf.float32, [None])
    self.sp = tf.placeholder(tf.uint8, [None, self.img_height, self.img_width, self.img_depth*self.FLAGS.state_hist])
    self.done_mask = tf.placeholder(tf.bool, [None])
    self.lr = tf.placeholder(tf.float32, [])

  def add_update_target_op(self):
    scope_col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
    target_scop_col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_scope)

    assn_vec = []
    for i,var in enumerate(scope_col):
      assign = tf.assign(target_scop_col[i], scope_col[i])
      assn_vec.append(assign)

    self.update_target_op = tf.group(*assn_vec)

  def update_step(self, t, replay_buffer, lr, summary):
    s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(self.FLAGS.batch_size)

    fd = {
      # inputs
      self.s: s_batch,
      self.a: a_batch,
      self.r: r_batch,
      self.sp: sp_batch,
      self.done_mask: done_mask_batch,
      self.lr: lr,
      # extra info
      self.avg_reward_placeholder: summary.avg_reward,
      self.max_reward_placeholder: summary.max_reward,
      self.std_reward_placeholder: summary.std_reward,
      self.avg_q_placeholder: summary.avg_q,
      self.max_q_placeholder: summary.max_q,
      self.std_q_placeholder: summary.std_q,
      self.eval_reward_placeholder: summary.eval_reward,
    }

    output_list = [self.loss, self.grad_norm, self.merged, self.train_op]
    loss_eval, grad_norm_eval, summary, _ = self.sess.run(output_list, feed_dict=fd)

    # tensorboard stuff
    self.file_writer.add_summary(summary, t)

    return loss_eval, grad_norm_eval

  def process_state(self, state):
    state = tf.cast(state, tf.float32)
    state /= self.FLAGS.high_val
    return state

  def add_summary(self):
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
    self.file_writer = tf.summary.FileWriter(self.FLAGS.output_path, self.sess.graph)

  def add_loss_op(self):
    self.q = self.get_q_values_op(self.proc_s, scope=self.scope, reuse=False)
    self.target_q = self.get_q_values_op(self.proc_sp, scope=self.target_scope, reuse=False)

    q_samp = self.r + self.FLAGS.gamma*(-tf.to_float(self.done_mask)+1.0)*tf.reduce_max(self.target_q,axis=1)
    q_s = tf.reduce_sum(self.q * tf.one_hot(self.a, self.num_actions), axis=1)

    losses = tf.square(q_samp-q_s)
    self.loss = tf.reduce_mean(losses)

  def add_optimizer_op(self):
    opt = tf.train.AdamOptimizer(self.lr)
    grads_and_vars = opt.compute_gradients(self.loss, var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope))
    clipped_grads_and_vars=[]
    clipped_grads_list=[]
    if (self.FLAGS.grad_clip):
      for grads,var in grads_and_vars:
        clipped_grads = tf.clip_by_norm(grads, self.FLAGS.clip_val)
        clipped_grads_and_vars.append((clipped_grads,var))
        clipped_grads_list.append(clipped_grads)

      self.train_op = opt.apply_gradients(clipped_grads_and_vars)
      self.grad_norm = tf.global_norm(clipped_grads_list)
    else:
      self.train_op = opt.apply_gradients(grads_and_vars)
      self.grad_norm = tf.global_norm([grads for grads, _ in grads_and_vars])

class LinearQ(Network):
  def get_q_values_op(self, state, scope, reuse=False):
    with tf.variable_scope(scope):
      out = layers.flatten(state, scope=scope)
      out = layers.fully_connected(inputs=out, num_outputs = self.num_actions, activation_fn=None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"1")
    return out

class FeedQ(Network):
  def get_q_values_op(self, state, scope, reuse=False):
    with tf.variable_scope(scope):
      out = layers.flatten(state, scope=scope)
      out = layers.fully_connected(inputs=out, num_outputs = 32, activation_fn=None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"1")
      out = tf.nn.relu(out)
      out = layers.fully_connected(inputs=out, num_outputs = 64, activation_fn=None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"2")
      out = tf.nn.relu(out)
      out = layers.fully_connected(inputs=out, num_outputs = self.num_actions, activation_fn=None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"3")
    return out

class DeepQ(Network):
  def get_q_values_op(self, state, scope, reuse=False):
    with tf.variable_scope(scope):
      out = layers.conv2d(inputs=state, num_outputs = 32, kernel_size=[8,8], stride=[4,4], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"1")
      out = layers.conv2d(inputs=out, num_outputs = 64, kernel_size=[4,4], stride=[2,2], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"2")
      out = layers.conv2d(inputs=out, num_outputs = 64, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"3")
      out = layers.flatten(out, scope=scope)
      out = layers.fully_connected(inputs=out, num_outputs = 512, activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"4")
      out = layers.fully_connected(inputs=out, num_outputs = self.num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"5")
    return out

class TransferQ(Network):
  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.num_actions = FLAGS.num_actions
    self.img_height, self.img_width, self.img_depth = FLAGS.state_size
    self.vgg = tf.contrib.slim.nets.vgg
    self.lstm_size = 64

    self.trainable_scope_vars = {}
    self.all_scope_vars = {}
    self.scope_init = {}

  def build(self):
    self.scope = "scope"
    self.target_scope = "target_scope"

    self.add_placeholders_op()

    self.proc_s = self.process_state(self.s)
    self.proc_sp = self.process_state(self.sp)

    self.load_model_and_graph_op()

    self.add_loss_op()

    self.transfer_loaded_graph_op()

    self.add_update_target_op()

    self.add_optimizer_op()

  def load_model_and_graph_op(self):
    dummy_size = tf.split(self.proc_s, self.FLAGS.state_hist, axis=3)[0]
    with slim.arg_scope(self.vgg.vgg_arg_scope()):
      q_vals, _ = self.vgg.vgg_16(dummy_size, num_classes=self.lstm_size)

    model_path = './vgg_16.ckpt'
    assert(os.path.isfile(model_path))

    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8',self.scope,self.target_scope])
    print(variables_to_restore)
    self.init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

    fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
    self.fc8_init = tf.variables_initializer(fc8_variables)
    self.all_loaded_vars = variables_to_restore + fc8_variables

  def get_q_values_op(self, state, scope, reuse=False):
    with tf.variable_scope(scope):
      with tf.variable_scope("all_vggs"):
        frames = tf.split(state, self.FLAGS.state_hist, axis=3)
        vgg_frames = []
        for i,f in enumerate(frames):
          if(i==0):
            vgg_frames.append(self.vgg.vgg_16(f, num_classes=self.lstm_size)[0])
          else:
            tf.get_variable_scope().reuse_variables()
            vgg_frames.append(self.vgg.vgg_16(f, num_classes=self.lstm_size)[0])

      vgg_tensor_input = tf.stack(vgg_frames,axis=1)

      with tf.variable_scope("trainable"):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        _, ( _, lstm_out) = tf.nn.dynamic_rnn(lstm_cell, vgg_tensor_input, dtype=tf.float32,scope=scope)
        q_vals = layers.fully_connected(inputs=lstm_out, num_outputs = self.num_actions, activation_fn=None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"fc")

      fc8_variables = tf.contrib.framework.get_variables(scope+'/all_vggs/vgg_16/fc8')
      other_trainable = tf.contrib.framework.get_variables(scope+'/trainable')
      print(fc8_variables+other_trainable)
      self.trainable_scope_vars[scope] = fc8_variables+other_trainable
      self.scope_init[scope] = tf.variables_initializer(fc8_variables+other_trainable)
      self.all_scope_vars[scope] = tf.contrib.framework.get_variables(scope)

      return q_vals

  def transfer_loaded_graph_op(self):
    loaded_vars = self.all_loaded_vars
    scope_vars = self.all_scope_vars[self.scope]
    target_scope_vars = self.all_scope_vars[self.target_scope]

    assn_vec = []
    for i,loaded_var in enumerate(loaded_vars):
      print(loaded_var)
      print(scope_vars[i])
      print(target_scope_vars[i])
      assign_scope = tf.assign(scope_vars[i], loaded_var)
      assn_vec.append(assign_scope)

    self.transfer_op = tf.group(*assn_vec)

  def add_update_target_op(self):
    scope_vars = self.all_scope_vars[self.scope]
    target_scope_vars = self.all_scope_vars[self.target_scope]

    assn_vec = []
    for i,var in enumerate(scope_vars):
      assign = tf.assign(target_scope_vars[i], scope_vars[i])
      assn_vec.append(assign)

    self.update_target_op = tf.group(*assn_vec)


  def initialize(self):
    self.sess = tf.Session()

    # tensorboard stuff
    self.add_summary()

    # initiliaze all variables
    self.sess.run(tf.global_variables_initializer())
    self.init_fn(self.sess)
    self.sess.run(self.fc8_init)

    self.sess.run(self.scope_init[self.scope])
    self.sess.run(self.scope_init[self.target_scope])

    self.sess.run(self.transfer_op)

    # synchronise q and target_q network
    self.update_target_params()

    # for saving network weights
    self.saver = tf.train.Saver(max_to_keep=2)

  def add_optimizer_op(self):
    opt = tf.train.AdamOptimizer(self.lr)
    grads_and_vars = opt.compute_gradients(self.loss, var_list = self.trainable_scope_vars[self.scope])
    clipped_grads_and_vars=[]
    clipped_grads_list=[]
    if (self.FLAGS.grad_clip):
      for grads,var in grads_and_vars:
        clipped_grads = tf.clip_by_norm(grads, self.FLAGS.clip_val)
        clipped_grads_and_vars.append((clipped_grads,var))
        clipped_grads_list.append(clipped_grads)

      self.train_op = opt.apply_gradients(clipped_grads_and_vars)
      self.grad_norm = tf.global_norm(clipped_grads_list)
    else:
      self.train_op = opt.apply_gradients(grads_and_vars)
      self.grad_norm = tf.global_norm([grads for grads, _ in grads_and_vars])

class RecurrentQ(Network):
  def get_q_values_op(self, state, scope, reuse=False):
    with tf.variable_scope(scope):
      frames = tf.split(state, self.FLAGS.state_hist, axis=3)
      cnn_frames = [self.cnn_network(f, scope, reuse=False) if i == 0 else self.cnn_network(f, scope, reuse=True) for i,f in enumerate(frames)]
      cnn_tensor_input = tf.stack(cnn_frames,axis=1)

      lstm_cell = tf.contrib.rnn.BasicLSTMCell(512)
      _, ( _, lstm_out) = tf.nn.dynamic_rnn(lstm_cell,cnn_tensor_input,dtype=tf.float32,scope=scope)
      q_vals = layers.fully_connected(inputs=lstm_out, num_outputs = self.num_actions, activation_fn=None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"fc")
      assert(q_vals.get_shape().as_list() == [None, self.num_actions])
      return q_vals

  def cnn_network(self, frame, scope, reuse=False):
    out = layers.conv2d(inputs=frame, num_outputs = 32, kernel_size=[8,8], stride=[4,4], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"1", reuse=reuse)
    out = layers.conv2d(inputs=out, num_outputs = 64, kernel_size=[4,4], stride=[2,2], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"2", reuse=reuse)
    out = layers.conv2d(inputs=out, num_outputs = 64, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"3", reuse=reuse)
    out = layers.flatten(out, scope=scope)
    out = layers.fully_connected(inputs=out, num_outputs = 512, activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"4", reuse=reuse)
    return out

#In development and testing
class HunterGathererQN(Network):
  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.num_actions = FLAGS.num_actions
    self.img_height, self.img_width, self.img_depth = FLAGS.state_size

    self.gatherer = RecurrentQ(self.FLAGS)
    self.hunter = RecurrentQ(self.FLAGS)
    selctor_FLAGS = self.FLAGS
    selector_FLAGS.num_actions = 2
    self.selector = RecurrentQ(selector_FLAGS)

  def build(self):
    self.gatherer.build()
    self.hunter.build()
    self.selector.build()

    self.add_total_loss_op()

    self.add_total_optimizer_op()

  def add_total_loss_op(self):
    hunter_loss = self.hunter.loss
    gatherer_loss = self.gatherer.loss
    self.add_loss_op()
    self.loss += hunter_loss + getherer_loss

  def add_total_optimizer_op():
    self.selector.loss = self.loss

  def get_q_values_op(self, state, scope, reuse=False):
    with tf.variable_scope(scope):
      self.hunter_q_vals = self.hunter.get_q_values_op(state, "hunter")
      self.gatherer_q_vals = self.hunter.get_q_values_op(state, "gatherer")
      self.selector_vals = self.hunter.get_q_values_op(state, "selector")
      to_hunt = self.selector_vals[0] > self.selector_vals[1]
      return tf.cond(to_hunt, self.hunter_q_vals, self.gatherer.q_vals)

class DeepAC(Network):
  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.num_actions = FLAGS.num_actions
    self.img_height, self.img_width, self.img_depth = FLAGS.state_size

  def build(self):
    self.scope = "scope"
    self.target_scope = "target_scope"

    self.add_placeholders_op()

    self.proc_s = self.process_state(self.s)
    self.proc_sp = self.process_state(self.sp)

    self.add_loss_op()

    self.add_optimizer_op()

  def initialize(self):
    self.sess = tf.Session()

    # tensorboard stuff
    self.add_summary()

    # initiliaze all variables
    self.sess.run(tf.global_variables_initializer())

    # for saving network weights
    self.saver = tf.train.Saver()

  def save(self):
    if not os.path.exists(self.FLAGS.model_path):
      os.makedirs(self.FLAGS.model_path)
    self.saver.save(self.sess, self.FLAGS.model_path)


  def process_state(self, state):
    state = tf.cast(state, tf.float32)
    state /= self.FLAGS.high_val
    return state

  def add_summary(self):
    # extra placeholders to log stuff from python
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

    self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
    self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
    self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

    # add placeholders from the graph
    tf.summary.scalar("actorloss", self.actorLoss)
    tf.summary.scalar("criticloss", self.criticLoss)
    #tf.summary.scalar("grads norm", self.grad_norm)

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
    self.file_writer = tf.summary.FileWriter(self.FLAGS.output_path, self.sess.graph)

  def get_best_action(self, state):
    action_values = self.sess.run(self.actor, feed_dict={self.s: [state]})[0]
    return np.argmax(action_values), action_values

  def add_placeholders_op(self):
    self.s = tf.placeholder(tf.uint8, [None, self.img_height, self.img_width, self.img_depth*self.FLAGS.state_hist])
    self.a = tf.placeholder(tf.int32, [None])
    self.r = tf.placeholder(tf.float32, [None])
    self.criticBest = tf.placeholder(tf.float32, [None])
    self.actorDiff = tf.placeholder(tf.float32, [None])
    self.sp = tf.placeholder(tf.uint8, [None, self.img_height, self.img_width, self.img_depth*self.FLAGS.state_hist])
    self.done_mask = tf.placeholder(tf.bool, [None])
    self.lr = tf.placeholder(tf.float32, [])

  def add_loss_op(self):
    self.actor, self.critic = self.get_actor_critic_values(self.proc_s, scope=self.scope, reuse=False)
    self.criticLoss = tf.reduce_mean(tf.square(self.critic-self.criticBest))

    value = tf.constant([i for i in range(self.FLAGS.batch_size)])
    print (value.shape, self.a.shape)
    inds = tf.stack([value, self.a], axis=1)
    sli = tf.gather_nd(self.actor, inds)
    self.actorLoss = tf.reduce_mean(tf.square( sli - self.actorDiff ))

  def add_optimizer_op(self):
    #need implementation
    opt = tf.train.AdamOptimizer(self.lr)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope + "_base")
    var_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope + "_actor"))
    grads_and_vars = opt.compute_gradients(self.actorLoss, var_list = var_list)
    clipped_grads_and_vars=[]
    clipped_grads_list=[]
    if (self.FLAGS.grad_clip):
      for grads,var in grads_and_vars:
        clipped_grads = tf.clip_by_norm(grads, self.FLAGS.clip_val)
        clipped_grads_and_vars.append((clipped_grads,var))
        clipped_grads_list.append(clipped_grads)

      self.train_op_actor = opt.apply_gradients(clipped_grads_and_vars)
      self.grad_norm_actor = tf.global_norm(clipped_grads_list)
    else:
      self.train_op_actor = opt.apply_gradients(grads_and_vars)
      self.grad_norm_actor = tf.global_norm([grads for grads, _ in grads_and_vars])

    opt = tf.train.AdamOptimizer(self.lr)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope + "_base")
    var_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope + "_critic"))
    var_list_without_old_adam_vars = []
    for el in var_list:
      if ("Adam:0" not in el.name.split('/') and "Adam_1:0" not in el.name.split('/')):
        var_list_without_old_adam_vars.append(el)
    grads_and_vars = opt.compute_gradients(self.criticLoss, var_list = var_list_without_old_adam_vars)
    clipped_grads_and_vars=[]
    clipped_grads_list=[]
    if (self.FLAGS.grad_clip):
      for grads,var in grads_and_vars:
        clipped_grads = tf.clip_by_norm(grads, self.FLAGS.clip_val)
        clipped_grads_and_vars.append((clipped_grads,var))
        clipped_grads_list.append(clipped_grads)

      self.train_op_critic = opt.apply_gradients(clipped_grads_and_vars)
      self.grad_norm_critic = tf.global_norm(clipped_grads_list)
    else:
      self.train_op_critic = opt.apply_gradients(grads_and_vars)
      self.grad_norm_critic = tf.global_norm([grads for grads, _ in grads_and_vars])

  def get_actor_critic_values(self, state, scope, reuse=False):
    #with tf.variable_scope(scope):
    out = layers.conv2d(inputs=state, num_outputs = 32, kernel_size=[8,8], stride=[4,4], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"_base_1")
    out = layers.conv2d(inputs=out, num_outputs = 64, kernel_size=[4,4], stride=[2,2], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"_base_2")
    out = layers.conv2d(inputs=out, num_outputs = 64, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"_base_3")
    out = layers.flatten(out, scope=scope)
    out = layers.fully_connected(inputs=out, num_outputs = 512, activation_fn=tf.nn.relu, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"_base_4")
    out1 = layers.fully_connected(inputs=out, num_outputs = self.num_actions, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"_actor")
    out2 = layers.fully_connected(inputs=out, num_outputs = 1, activation_fn = None, weights_initializer=layers.xavier_initializer(), biases_initializer=tf.constant_initializer(0), scope=scope+"_critic")
    return out1, out2


  def update_actor_step(self, t, replay_buffer, lr, summary):
    s_batch, a_batch, r_batch, sp_batch, done_mask_batch, criticBest_batch, actorDiff_batch = replay_buffer.sample(self.FLAGS.batch_size)

    fd = {
      # inputs

      self.s: s_batch,
      self.a: a_batch,
      self.r: r_batch,
      self.sp: sp_batch,
      self.done_mask: done_mask_batch,
      self.lr: lr,
      self.criticBest: criticBest_batch,
      self.actorDiff: actorDiff_batch,
      # extra info
      self.avg_reward_placeholder: summary.avg_reward,
      self.max_reward_placeholder: summary.max_reward,
      self.std_reward_placeholder: summary.std_reward,
      self.avg_q_placeholder: summary.avg_q,
      self.max_q_placeholder: summary.max_q,
      self.std_q_placeholder: summary.std_q,
      self.eval_reward_placeholder: summary.eval_reward,
    }
    output_list = [self.actorLoss, self.grad_norm_actor, self.merged, self.train_op_actor]
    loss_eval, grad_norm_eval, summary, _ = self.sess.run(output_list, feed_dict=fd)
    #print ("Updating actor ", np.mean(actorDiff_batch), np.amax(actorDiff_batch))

    # tensorboard stuff
    self.file_writer.add_summary(summary, t)

    return loss_eval, grad_norm_eval

  def update_critic_step(self, t, replay_buffer, lr, summary):
    s_batch, a_batch, r_batch, sp_batch, done_mask_batch, criticBest_batch, actorDiff_batch = replay_buffer.sample(self.FLAGS.batch_size)

    fd = {
      # inputs
      self.s: s_batch,
      self.a: a_batch,
      self.r: r_batch,
      self.sp: sp_batch,
      self.done_mask: done_mask_batch,
      self.lr: lr,
      self.criticBest: criticBest_batch,
      self.actorDiff: actorDiff_batch,
      # extra info
      self.avg_reward_placeholder: summary.avg_reward,
      self.max_reward_placeholder: summary.max_reward,
      self.std_reward_placeholder: summary.std_reward,
      self.avg_q_placeholder: summary.avg_q,
      self.max_q_placeholder: summary.max_q,
      self.std_q_placeholder: summary.std_q,
      self.eval_reward_placeholder: summary.eval_reward,
    }
    #print ("Updating critic ", np.mean(criticBest_batch), np.amax(criticBest_batch))
    output_list = [self.criticLoss, self.grad_norm_critic, self.merged, self.train_op_critic]
    loss_eval, grad_norm_eval, summary, _ = self.sess.run(output_list, feed_dict=fd)

    # tensorboard stuff
    self.file_writer.add_summary(summary, t)

    return loss_eval, grad_norm_eval

  def calcState(self, state):
    return self.sess.run([self.critic], feed_dict = {self.s: np.expand_dims(state, axis=0)})[0] #need to expand dims because it is batch of one
