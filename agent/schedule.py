import numpy as np
class LinearSchedule(object):
  def __init__(self, eps_begin, eps_end, nsteps):
    """
    Args:
      eps_begin: initial exploration
      eps_end: end exploration
      nsteps: number of steps between the two values of eps
    """
    self.epsilon        = eps_begin
    self.eps_begin      = eps_begin
    self.eps_end        = eps_end
    self.nsteps         = nsteps


  def update(self, t):
    """
    Args:
      t: (int) nth frames
    """
    if t == 0:
      self.epsilon = self.eps_begin
    elif t >= self.nsteps:
      self.epsilon = self.eps_end
    else:
      eps_decay = 1.0*(self.eps_begin - self.eps_end)/self.nsteps
      self.epsilon = self.eps_begin - t*eps_decay


class LinearExploration(LinearSchedule):
  def __init__(self, env, eps_begin, eps_end, nsteps):
    """
    Args:
      env: gym environment
      eps_begin: initial exploration
      eps_end: end exploration
      nsteps: number of steps between the two values of eps
    """
    self.env = env
    super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)

  def get_action(self, best_action):
    """
    Args:
      best_action: (int) best action according some policy
    Returns:
      an action
    """
    if np.random.random() <= self.epsilon:
      return self.env.action_space.sample()
    return best_action

class ExpSchedule(object):
  def __init__(self, eps_begin, eps_end, eps_decay, nsteps, steps_per):
    """
    Args:
      eps_begin: initial exploration
      eps_decay: decay rate for epsilon
      nsteps: number of steps before bottom threshold
    """
    self.epsilon        = eps_begin
    self.eps_decay      = eps_decay
    self.eps_end        = eps_end
    self.nsteps         = nsteps
    self.steps_per = steps_per

  def update(self, t):
    if t == 0:
      self.epsilon = self.eps_begin
    elif t >= self.nsteps:
      self.epsilon = self.eps_end
    elif (t % self.steps_per == 0):
      self.epsilon = self.eps_begin*self.eps_decay

class ExpExploration(LinearSchedule):
  def __init__(self, env, eps_begin, eps_end, eps_decay, nsteps, steps_per):
    """
    Args:
      env: gym environment
      eps_begin: initial exploration
      eps_end: end exploration
      nsteps: number of steps between the two values of eps
    """
    self.env = env
    super(LinearExploration, self).__init__(eps_begin, eps_end, eps_decay, nsteps, steps_per)

  def get_action(self, best_action):
    """
    Args:
      best_action: (int) best action according some policy
    Returns:
      an action
    """
    if np.random.random() <= self.epsilon:
      return self.env.action_space.sample()
    return best_action

class BGreedySchedule(object):
  def __init__(self, eps_begin, eps_end, eps_decay1, eps_decay2, nsteps, steps_per, iter_decay):
    """
    Args:
      eps_begin: initial exploration
      eps_decay: decay rate for epsilon
      nsteps: number of steps before bottom threshold
    """
    self.eps_begin      = eps_begin
    self.epsilon1       = eps_begin
    self.epsilon2       = eps_begin
    self.eps_decay1     = eps_decay1
    self.eps_decay2     = eps_decay2
    self.eps_end        = eps_end
    self.eps_diff       = 0
    self.nsteps         = nsteps
    self.steps_per = steps_per

  def update(self, t):
    if t == 0:
      self.epsilon1       = self.eps_begin
      self.epsilon2       = self.eps_begin
    elif t >= self.nsteps:
      self.epsilon1       = self.eps_end
      self.epsilon2       = self.eps_end
    elif (t % self.steps_per == 0):
      self.epsilon1       = self.epsilon1*eps_decay1
      self.epsilon2       = self.epsilon2*eps_decay2
      self.eps_diff       = abs(self.epsilon1 - self.epsilon2)

class BGreedyExploration(BGreedySchedule):
  def __init__(self, eps_begin, eps_end, eps_decay1, eps_decay2, nsteps, steps_per, iter_decay):
    """
    Args:
      env: gym environment
      eps_begin: initial exploration
      eps_end: end exploration
      nsteps: number of steps between the two values of eps
    """
    self.env = env
    super(BGreedyExploration, self).__init__(eps_begin, eps_end, eps_decay1, eps_decay2, nsteps, steps_per,iter_decay)

  def get_action(self, best_action, ep_step_number):
    """
    Args:
      best_action: (int) best action according some policy
    Returns:
      an action
    """
    iter_epsilon = max(self.epsilon1,self.epsilon2) - self.eps_diff*(iter_decay**ep_step_number)
    if np.random.random() <= iter_epsilon:
      return self.env.action_space.sample()
    return best_action
