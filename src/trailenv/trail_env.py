from enum import IntEnum

# import gymnasium as gym
import ipdb
import numpy as np
# from gymnasium import spaces
import gym
from gym import spaces


class Actions(IntEnum):
  left = 0
  right = 1
  up = 2
  down = 3
  topleft= 4
  topright = 5
  botleft = 6
  botright = 7

ACTION_COORDS = {
  Actions.left: np.array([0,-1]),
  Actions.right: np.array([0,1]),
  Actions.up: np.array([-1,0]),
  Actions.down: np.array([1,0]),
  Actions.topleft: np.array([-1,-1]),
  Actions.topright: np.array([-1,1]),
  Actions.botleft: np.array([1,-1]),
  Actions.botright: np.array([1,1]),
}

KEY_ACTION_MAP = {
  "w": Actions.up,
  "a": Actions.left,
  "s": Actions.down,
  "d": Actions.right,
  "q": Actions.topleft,
  "e": Actions.topright,
  "z": Actions.botleft,
  "x": Actions.botright,
}

class Entities(IntEnum):
  empty = 0
  wall = 1
  trail = 2
  agent = 3

"""A set of common utilities used within the environments.

These are not intended as API functions, and will not remain stable over time.
"""

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(
    string: str, color: str, bold: bool = False, highlight: bool = False
) -> str:
    """Returns string surrounded by appropriate terminal colour codes to print colourised text.

    Args:
        string: The message to colourise
        color: Literal values are gray, red, green, yellow, blue, magenta, cyan, white, crimson
        bold: If to bold the string
        highlight: If to highlight the string

    Returns:
        Colourised string
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return f"\x1b[{attrs}m{string}\x1b[0m"

class TrailEnv(gym.Env):
  def __init__(self, width, height, start_pos, trail):
    self.width = width
    self.height = height
    self.start_pos = start_pos
    self.trail = trail
    self.trail_idx = 0

    self.observation_space = spaces.Box(
      low=np.array([0.0, 0.0, 0]),
      high=np.array([height, width, len(trail)]),
      shape=(3,),
      dtype="float32",
    )
    self.action_space = spaces.Discrete(len(Actions))


  def _reset_grid(self):
    self.grid = np.zeros((self.height, self.width), dtype=int)
    # add walls around edges.;w
    self.grid[0, :self.width] = Entities.wall
    self.grid[-1, :self.width] = Entities.wall
    self.grid[:self.height, 0] = Entities.wall
    self.grid[:self.height, -1] = Entities.wall
    for h, w in self.trail:
      self.grid[h, w] = Entities.trail
    self.grid[self.start_pos[0], self.start_pos[1]] = Entities.agent

  def step(self, action):
    # first do the action
    old_pos = self.curr_pos
    new_pos = self.curr_pos + ACTION_COORDS[action]

    # if agent is out of bounds or inside a wall, revert back.
    within_bounds =  (0 <= new_pos[0] < self.height) and (0 <= new_pos[1] < self.width)
    if not within_bounds or (self.grid[new_pos[0], new_pos[1]] == Entities.wall):
      new_pos = self.curr_pos

    # update the grid
    self.grid[old_pos[0], old_pos[1]] = Entities.empty
    self.grid[new_pos[0], new_pos[1]] = Entities.agent

    # update curr_pos for next timestep.
    self.curr_pos = new_pos

    # Reward computation. check if agent is on the trail.
    reward = 0
    if self.trail_idx < len(self.trail) and new_pos[0] == self.trail[self.trail_idx][0] and new_pos[1] == self.trail[self.trail_idx][1]:
      reward += 1
      # print('finished trail idx', self.trail_idx)
      self.trail_idx += 1

    # terminated = self.trail_idx >= len(self.trail)
    terminated = False
    truncated = False
    obs = self.gen_obs()
    # return obs, reward, terminated, truncated, {}
    return obs, reward, terminated, {}

  def gen_obs(self):
    # two one-hot vectors, one for each dimension.
    # width_vec = np.zeros(self.width, dtype=float)
    # height_vec = np.zeros(self.height, dtype=float)
    # width_vec[self.curr_pos[1]] = 1.0
    # height_vec[self.curr_pos[0]] = 1.0
    # # concat them into one vector.
    # obs = np.concatenate([width_vec, height_vec])
    # obs = np.array(self.curr_pos)
    obs = np.concatenate([self.curr_pos, [self.trail_idx]])
    return obs

  def reset(self, *, seed=None, options=None):
    self.curr_pos = np.array(self.start_pos)
    self.trail_idx = 0
    self._reset_grid()
    init_obs = self.gen_obs()
    return init_obs

  @property
  def ascii(self):
    """
    Produce a pretty string of the environment's grid along with the agent.
    """
    grid_str = ""
    ENTITY_TO_STRING = {
      Entities.empty: " ",
      Entities.wall: "x",
      Entities.trail: colorize(".", "blue", highlight=True),
      Entities.agent: colorize("A", "yellow", highlight=True),
    }
    for j in range(self.height):
      for i in range(self.width):
        grid_str += ENTITY_TO_STRING[self.grid[j,i]]
      if j < self.height - 1:
          grid_str += "\n"
    return grid_str

class NoisyTrailEnv(TrailEnv):
  # FO: height, width
  # PO: noisy height, width
  def __init__(self, width, height, start_pos, trail, y_std):
    super().__init__(width, height, start_pos, trail)
    self.observation_space = spaces.Box(
      low=np.array([0.0, 0.0, 0.0, 0.0, 0.0,]),
      high=np.array([height, width, height, width, len(trail)]),
      shape=(5,),
      dtype="float32",
    )
    self.y_std = y_std
    self.rng = np.random.default_rng(12345)

  def gen_obs(self):
    noisy_pos = np.array(self.curr_pos, dtype=np.float32)
    noisy_pos[0] += self.rng.normal(scale=self.y_std)
    obs = np.concatenate([self.curr_pos, noisy_pos, [self.trail_idx]])
    return obs


if __name__ == "__main__":
  width = height = 6
  start_pos = [1,1]
  trail = [[2,2], [3,3]]
  env = TrailEnv(width, height, start_pos, trail)
  env.reset()
  ipdb.set_trace()
  print(env)
  done = False
  while not done:
    key = input("type in wasdqezx")
    if key in KEY_ACTION_MAP:
      obs, rew, done, trunc, info = env.step(KEY_ACTION_MAP[key])
      print(env.ascii)