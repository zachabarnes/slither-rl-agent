import gym
import universe  # register the universe environments
import numpy as np
import gym
import cv2
from gym.spaces.box import Box
from gym import spaces
from collections import deque
from utils.preprocess import greyscale
from utils.wrappers import MaxAndSkipEnv
from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe.wrappers.experimental import SafeActionSpace
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode

from schedule import LinearExploration, LinearSchedule
from model import DeepQ

from configs.RattLe import config

from env import create_slither_env
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.misc import imresize

import pyglet


class SimpleImageViewer(object):
    """
    Modified version of gym viewer to chose format (RBG or I)
    see source here https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
    """
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

        ##########################
        ####### old version ######
        # assert arr.shape == (self.height, self.width, I), "You passed in an image with the wrong number shape"
        # image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes())
        ##########################
        
        ##########################
        ####### new version ######
        nchannels = arr.shape[-1]
        if nchannels == 1:
            _format = "I"
        elif nchannels == 3:
            _format = "RGB"
        else:
            raise NotImplementedError
        image = pyglet.image.ImageData(self.width, self.height, "RGB", arr.tobytes())
        ##########################
        
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
        return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None
                for ob in observation_n]

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
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]

class SlitherRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(SlitherRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [300, 500, 3])

    def _observation(self, observation_n):
        return [_process_frame_slither(observation) for observation in observation_n]

def _process_frame_slither(frame):
    frame = cv2.resize(frame, (500, 300))
    # frame = frame.mean(2).astype(np.float32)
    # frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [300, 500, 3])
    return frame

class RenderWrapper(vectorized.Wrapper):
    """
    Wrapper for Pong to apply preprocessing
    Stores the state into variable self.obs
    """
    def __init__(self, env):
        self.viewer = None
        super(RenderWrapper, self).__init__(env)

    def _step(self, action):
        """
        Overwrites _step function from environment to apply preprocess
        """
        self.orig_obs, reward, done, info = self.env.step(action)
        self.orig_obs[0] = self.orig_obs[0][::-1,:,:]
        self.proc_obs = slither_process(np.copy(self.orig_obs))
        return self.proc_obs, reward, done, info

    def _render(self, mode='human', close=False):
        """
        Overwrite _render function to vizualize preprocessing
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        np.save("image",self.orig_obs[0])
        img = np.concatenate((self.orig_obs[0],self.proc_obs[0]),1)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)

def slither_process(frame):
    frame = frame[0]
    abs_t = 115
    frame[(frame[:,:,0]<abs_t)*(frame[:,:,1]<abs_t)*(frame[:,:,2]<abs_t)] = 0
    
    rel_t = 30
    avg_pix = np.mean(frame,2)
    diff = np.abs(avg_pix-frame[:,:,0]) + np.abs(avg_pix-frame[:,:,1]) + np.abs(avg_pix-frame[:,:,2])
    frame[:,:,:] = 255
    frame[diff<rel_t] = 0
    
    sing_frame = ndimage.grey_erosion(frame[:,:,1], size=(2,2))
    blur_radius = .35
    sing_frame = ndimage.gaussian_filter(sing_frame, blur_radius)
    labeled, nr_objects = ndimage.label(sing_frame)
    
    snake_threshold = 235
    enemy_c = [255,0,0]
    me_c = [0,255,0]
    food_c = [0,0,255]
    frame[:,:,:] = 0
    me_label = np.bincount(labeled[145:155,245:255].flatten().astype(int))[1:]
    me_label = np.argmax(me_label) + 1
    for i in range(nr_objects):
        label = i+1
        size = np.count_nonzero(labeled[labeled==label])
        if size<snake_threshold:
            frame[labeled==label] = food_c
        elif me_label  == label:
            frame[labeled==label] = me_c
        else:
            frame[labeled==label] = enemy_c
    return [frame]

if __name__ == '__main__':
	env = create_slither_env()

	env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={
	'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})  

	observation_n = env.reset()

	while True:
		#action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
		action_n = 1
		observation_n, reward_n, done_n, info = env.step(action_n)
		env.render()









