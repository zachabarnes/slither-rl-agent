import gym
import universe  # register the universe environments
import numpy as np
import pyglet
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from gym.spaces.box import Box
from gym import spaces
from collections import deque

from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe.wrappers.experimental import SafeActionSpace
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode

class SimpleImageViewer(object):
  def __init__(self, display=None):
    self.window = None
    self.isopen = False
    self.display = display

  def imshow(self, arr):
    if self.window is None:
      height, width, channels = arr.shape
      self.window = pyglet.window.Window(width=width, height=height, display=self.display)
      self.width = width
      self.height = height
      self.isopen = True

    nchannels = arr.shape[-1]
    if nchannels == 1:
      _format = "I"
    elif nchannels == 3:
      _format = "RGB"
    else:
      raise NotImplementedError
    image = pyglet.image.ImageData(self.width, self.height, "RGB", arr.tobytes())

    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    image.blit(0,0)
    self.window.flip()

  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

  def __del__(self):
    self.close()

class CropScreen(vectorized.ObservationWrapper):
  def __init__(self, env, height, width, top=0, left=0):
    super(CropScreen, self).__init__(env)
    self.height = height
    self.width = width
    self.top = top
    self.left = left
    self.observation_space = Box(0, 255, shape=(height, width, 3))

  def _observation(self, observation_n):
    return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None for ob in observation_n]

class FixedKeyState(object):
  def __init__(self, keys):
    self._keys = [keycode(key) for key in keys]
    self._down_keysyms = set()

  def apply_vnc_actions(self, vnc_actions):
    for event in vnc_actions:
      if isinstance(event, vnc_spaces.KeyEvent):
        if event.down:
          self._down_keysyms.add(event.key)
        else:
          self._down_keysyms.discard(event.key)

  def to_index(self):
    action_n = 0
    for key in self._down_keysyms:
      if key in self._keys:
        # If multiple keys are pressed, just use the first one
        action_n = self._keys.index(key) + 1
        break
    return action_n

class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
  def __init__(self, env, keys):
    super(DiscreteToFixedKeysVNCActions, self).__init__(env)

    self._keys = keys
    self._generate_actions()
    self.action_space = spaces.Discrete(len(self._actions))

  def _generate_actions(self):
    self._actions = []
    uniq_keys = set()
    for key in self._keys:
      for cur_key in key.split(' '):
        uniq_keys.add(cur_key)

    for key in [''] + self._keys:
      split_keys = key.split(' ')
      cur_action = []
      for cur_key in uniq_keys:
        cur_action.append(vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in split_keys)))
      self._actions.append(cur_action)
    self.key_state = FixedKeyState(uniq_keys)

  def _action(self, action_n):
    # Each action might be a length-1 np.array. Cast to int to avoid warnings.
    return [self._actions[int(action)] for action in action_n]

class RenderWrapper(vectorized.Wrapper):
  def __init__(self, env, state_type):
    self.viewer = None

    self.state_type = state_type
    self.processor = SlitherProcessor(state_type)
    self.state_size = self.processor.state_size
    self.high_val = self.processor.high_val

    super(RenderWrapper, self).__init__(env)

  def _reset(self):
    self.orig_obs = self.env.reset()
    self.proc_obs = self.processor.process(np.copy(self.orig_obs))
    small_proc_obs = self.processor.resize(np.copy(self.proc_obs))
    return small_proc_obs

  def _step(self, action):
    self.orig_obs, reward, done, info = self.env.step(action)
    self.proc_obs = self.processor.process(np.copy(self.orig_obs))
    small_proc_obs = self.processor.resize(np.copy(self.proc_obs))
    return small_proc_obs, reward, done, info

  def _render(self, mode='human', close=False):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if self.state_type == 'shapes':
      gray_img = self.proc_obs[0]*100
      proc_obs = np.concatenate((gray_img, gray_img, gray_img),2)
      img = np.concatenate((self.orig_obs[0],proc_obs),1)

    elif self.state_type == 'colors':
      img = np.concatenate((self.orig_obs[0],self.proc_obs[0]),1)

    else:
      img = self.orig_obs[0]

    img = img.astype(np.uint8)
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = SimpleImageViewer()
      self.viewer.imshow(img)

class SlitherProcessor(object):
  def __init__(self, state_type):
    self.state_type = state_type

    if self.state_type == 'features':
      self.state_size = [5,1,1]
      self.zoom = (1,1,1)
      self.high_val = 1.0

    elif self.state_type == 'colors':
      self.state_size = [30,50,3]
      self.zoom = (.1,.1,1)
      self.high_val = 255.0

    elif self.state_type == 'shapes':
      self.state_size = [30,50,1]
      self.zoom = (.1,.1,1)
      self.high_val = 1.0

    elif self.state_type == 'transfer':
      self.state_size = [224,224,3]
      self.zoom = (.75,.75,1)
      self.high_val = 255.0

    else: NotImplementedError

  def process(self, frames):
    if self.state_type == 'features':
      return [self.process_features(f) for f in frames]

    elif self.state_type == 'colors':
      return [self.process_colors(f) for f in frames]

    elif self.state_type == 'shapes':
      return [self.process_shapes(f) for f in frames]

    elif self.state_type == 'transfer':
      frames = [self.process_colors(f) for f in frames]
      return [f[1:,101:-100,:] for f in frames]

  def resize(self, frames):
    if self.state_type != 'features':
      return [ndimage.zoom(f, self.zoom, order=2) for f in frames]
    else:
      return frames

  def process_shapes(self,frame):
    frame = self.remove_background(frame)
    label = self.connected_components(frame)
    return self.extract_shapes(label)

  def process_colors(self,frame):
    frame = self.remove_background(frame)
    label = self.connected_components(frame)
    return self.extract_colors(frame,label)

  def process_features(self,frame):
    frame = self.remove_background(frame)
    label = self.connected_components(frame)
    frame = self.extract_colors(frame,label)
    return self.extract_features(frame)

  def remove_background(self,frame):
    abs_t = 115
    frame[(frame[:,:,0]<abs_t)*(frame[:,:,1]<abs_t)*(frame[:,:,2]<abs_t)] = 0

    rel_t = 30
    avg_pix = np.mean(frame,2)
    diff = np.abs(avg_pix-frame[:,:,0]) + np.abs(avg_pix-frame[:,:,1]) + np.abs(avg_pix-frame[:,:,2])
    frame[:,:,:] = 255
    frame[diff<rel_t] = 0
    return frame

  def connected_components(self, frame):
    sing_frame = ndimage.grey_erosion(frame[:,:,1], size=(2,2))
    blur_radius = .35
    sing_frame = ndimage.gaussian_filter(sing_frame, blur_radius)
    labeled, self.nr_objects = ndimage.label(sing_frame)
    return labeled[:,:,np.newaxis]

  def extract_shapes(self, frame):
    frame[frame != 0] = 1
    return frame

  def extract_colors(self, frame, label):
    label = label[:,:,0]
    snake_threshold = 235
    enemy_c = [255,0,0]
    me_c = [0,255,0]
    food_c = [0,0,255]
    frame[:,:,:] = 0
    me_label = np.bincount(label[145:155,245:255].flatten().astype(int))[1:]
    if len(me_label)>0:
      me_label = np.argmax(me_label) + 1
    else:
      me_label = -1
    for i in range(self.nr_objects):
      cur_label = i+1
      size = np.count_nonzero(label[label==cur_label])
      if size<snake_threshold:
        frame[label==cur_label] = food_c
      elif me_label==cur_label:
        frame[label==cur_label] = me_c
      else:
        frame[label==cur_label] = enemy_c
    return frame

  def extract_features(self, frame):
    snake_layer = frame[:,:,0]
    me_layer = frame[:,:,1]
    food_layer = frame[:,:,2]

    num_pix = frame.shape[0]*frame.shape[1]
    max_dis = 150+250

    snake_pix = np.count_nonzero(snake_layer)
    me_pix = np.count_nonzero(me_layer)
    food_pix = np.count_nonzero(food_layer)

    snake_perc = (num_pix - snake_pix)*1.0/num_pix
    food_perc = 1.0*food_pix/num_pix
    me_perc = 1.0*me_pix/num_pix

    snake_inds = np.nonzero(snake_layer)
    snake_inds = zip(snake_inds[0].tolist(),snake_inds[1].tolist())
    food_inds = np.nonzero(food_layer)
    snake_inds = zip(food_inds[0].tolist(),food_inds[1].tolist())

    snake_dis = min([self.d(i) for i in snake_inds])
    food_dis  = min([self.d(i) for i in food_inds])

    min_snake = snake_dis*1.0/max_dis
    min_food  = 1.0*(max_dis - food_dis)/max_dis

    features = np.array([me_perc,snake_perc,food_perc,min_snake,min_food])
    return features[:, np.newaxis, np.newaxis]


  def d(self, ind):
    return abs(15-ind[0]) + abs(25-ind[1])


def create_slither_env(state_type):
  env = gym.make('internet.SlitherIO-v0')
  env = Vision(env)

  #Because logging is annoying
  #env = Logger(env)

  env = BlockingReset(env)
  env = CropScreen(env, 300, 500, 84, 18)
  env = DiscreteToFixedKeysVNCActions(env, ['left', 'right'])#['left', 'right', 'space', 'left space', 'right space'])
  env = EpisodeID(env)
  env = RenderWrapper(env, state_type)
  return env
