from minigrid.envs import CrossingEnv
from gym.core import Wrapper, ActionWrapper, ObservationWrapper
from gym import spaces
from minigrid.core.constants import OBJECT_TO_IDX
import numpy as np

class GymActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(self.action_space.n)

    def action(self, action):
      return action


class OldGymWrapper(Wrapper):
  """
  Wrapper that
    1. converts term, trunc to done
    2. old reset
  """

  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    """Steps through the environment with `action`."""
    obs, reward, terminated, truncated, info = self.env.step(action)
    done = terminated or truncated
    return obs, reward, done, info

  def reset(self, **kwargs):
    """Resets the environment with `kwargs`."""
    return self.env.reset(**kwargs)[0]

  def render(self, mode="human", **kwargs):
    return self.env.render()


class SymbolicObsWrapper(ObservationWrapper):
    """
    Fully observable grid with a symbolic state representation.
    The symbol is a triple of (X, Y, IDX), where X and Y are
    the coordinates on the grid, and IDX is the id of the object.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-1, high=100, shape=(self.env.width * self.env.height + 2 + 1, ), dtype="float32")
        # self.observation_space = spaces.Box(low=-1, high=100, shape=(3, ), dtype="float32")


    def observation(self, obs):
        objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
        )
        agent_pos = self.env.agent_pos
        # import ipdb; ipdb.set_trace()
        # w, h = self.width, self.height
        # grid = np.mgrid[:w, :h]
        # grid = np.concatenate([grid, objects.reshape(1, w, h)])
        # grid = np.transpose(grid, (1, 2, 0))
        # grid[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX["agent"]
        # flat_grid = np.zeros(81*3 + 1, dtype="float32")
        # flat_grid[:-1] = np.array(grid).ravel()
        # flat_grid[-1] = obs["direction"]
        # obs["image"] = grid
        # 81 obj + 2 agent coord + 1 direction = 84
        return np.concatenate([objects, agent_pos, [obs['direction']]])
        # return np.concatenate([agent_pos, [obs['direction']]])

def make_lava_env(render_mode="rgb_array"):
  from minigrid.envs import CrossingEnv
  from minigrid.wrappers import ReseedWrapper

  env = CrossingEnv(size=5,  max_steps=100, render_mode=render_mode)
  env = OldGymWrapper(env) # revert reset, step back to old gym semantics.
  env = SymbolicObsWrapper(env) # change obs to flat vector of [81 * (object), direction]
  env = GymActionWrapper(env)
  env = ReseedWrapper(env, seeds=[0]) # make lava spawn deterministic.
  return env

def make_lava_cliff_env(render_mode="rgb_array"):
  from minigrid.envs import CliffEnv
  from minigrid.wrappers import ReseedWrapper

  env = CliffEnv(size=7,  n_obstacles=1, max_steps=100, render_mode=render_mode)
  env = OldGymWrapper(env) # revert reset, step back to old gym semantics.
  env = SymbolicObsWrapper(env) # change obs to flat vector of [81 * (object), direction]
  env = GymActionWrapper(env)
  # env = ReseedWrapper(env, seeds=[0]) # make lava spawn deterministic.
  return env

if __name__ == "__main__":
  # env = CrossingEnv(size=9,  max_steps=100, render_mode="human")
  # env = SymbolicObsWrapper(env)
  # env = ReseedWrapper(env, seeds=[0]) # make lava spawn deterministic.
  env = make_lava_env("human")
  env.reset()
  env.render()
  # first move down.
  action = 1  # User-defined policy function
  observation, reward, done, info = env.step(action)
  while True:
    env.render()
  print(reward, done)
  for _ in range(10):
    action = 2  # User-defined policy function
    observation, reward, done, info = env.step(action)
    env.render()
    print(reward, done)

  # then move right
  action = 0  # User-defined policy function
  observation, reward, done, info = env.step(action)
  env.render()
  print(reward, done)
  for _ in range(10):
    action = 2  # User-defined policy function
    observation, reward, done, info = env.step(action)
    env.render()
    print(reward, done)

  env.close()