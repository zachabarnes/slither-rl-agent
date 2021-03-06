import gym
import universe  # register the universe environments
import sys

from utils.env import create_slither_env
from universe.wrappers import Unvectorize

if __name__ == '__main__':

  # Create customized and processed slither env
  #universe.configure_logging(False)
  env = create_slither_env('shapes')
  env = Unvectorize(env)
  env.configure(fps=5.0, remotes=1, start_timeout=15 * 60, vnc_driver='go', vnc_kwargs={'encoding': 'tight', 'compress_level': 0, 'fine_quality_level': 50})

  observation_n = env.reset()

  print("Num_actions: " + str(env.action_space.n))

  while True:
    action_n = env.action_space.sample()
    observation_n, reward_n, done_n, info = env.step(0)
    if sys.platform == 'linux':
      #Cant render on server
      print(observation_n)
      print(observation_n.shape)
    else:
      env.render()
