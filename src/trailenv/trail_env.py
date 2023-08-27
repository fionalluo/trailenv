from enum import IntEnum

# import gymnasium as gym
import ipdb
import numpy as np
from gymnasium import spaces
import gymnasium as gym


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
      dtype="int64",
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

    terminated = self.trail_idx >= len(self.trail)
    # terminated = False
    truncated = False
    obs = self.gen_obs()
    return obs, reward, terminated, truncated, {}

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
    return init_obs, {}

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

class NoisyFootballEnv(TrailEnv):
  # FO: h, w, noisy h, w, trail_idx
  # PO: 0, 0, noisy h, w, trail_idx
  def __init__(self, width, height, start_pos, trail, y_std):
    super().__init__(width, height, start_pos, trail)
    self.observation_space = spaces.Box(
      low=np.array([0.0, 0.0, 0.0, 0.0, 0]),
      high=np.array([height, height, width, height, len(trail) ]),
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

  def step(self, action):
    # first do the action
    old_pos = self.curr_pos
    new_pos = self.curr_pos + ACTION_COORDS[action]

    # if agent is out of bounds or inside a wall, revert back.
    within_bounds =  (0 <= new_pos[0] < self.height) and (0 <= new_pos[1] < self.width)
    if not within_bounds or (self.grid[new_pos[0], new_pos[1]] == Entities.wall):
      new_pos = self.curr_pos

    # or if agent is off trail, move agent back to start.
    on_trail = self.trail_idx < len(self.trail) and new_pos[0] == self.trail[self.trail_idx][0] and new_pos[1] == self.trail[self.trail_idx][1]
    # print("on_trail", on_trail, " trail_idx", self.trail_idx)
    if on_trail:
      # print("on trail, incrementing.")
      self.trail_idx += 1
    else:
      new_pos = np.array(self.start_pos)

    # move the agent to the new position.
    self.grid[old_pos[0], old_pos[1]] = Entities.empty
    self.grid[new_pos[0], new_pos[1]] = Entities.agent

    # update curr_pos for next timestep.
    self.curr_pos = new_pos

    terminated = False
    obs = self.gen_obs()
    # give reward if agent is on other side of the field and done with trail.
    reward = float(self.trail_idx >= len(self.trail) and obs[-2] == self.width - 2)
    return obs, reward, terminated, {}

class ObsDictTrailEnv(gym.Env):
  def __init__(self, width, height, start_pos, trail, observation_type="FO"):
    self.width = width
    self.height = height
    self.start_pos = start_pos
    self.trail = trail
    self.trail_idx = 0
    assert observation_type in ["FO", "PO"]
    self.observation_type = observation_type
    _obs_dict = {}    
    _obs_dict["x"] = spaces.Box(
      low=np.array([0]),
      high=np.array([width]),
      shape=(1,),
      dtype="int64",
    )
    if observation_type == "FO":
      _obs_dict["y"] = spaces.Box(
        low=np.array([0]),
        high=np.array([height]),
        shape=(1,),
        dtype="int64",
      )
    self.observation_space = spaces.Dict(_obs_dict)
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

    terminated = self.trail_idx >= len(self.trail)
    # terminated = False
    truncated = False
    obs = self.gen_obs()
    return obs, reward, terminated, truncated, {}

  def gen_obs(self):
    # obs = {
    #   "x_r": np.array([self.curr_pos[1], self.trail_idx]),
    # }
    obs = {
      "x": np.array([self.curr_pos[1]]),
    }
    if self.observation_type == "FO":
      obs["y"] = np.array([self.curr_pos[0]])
    return obs

  def reset(self, *, seed=None, options=None):
    self.curr_pos = np.array(self.start_pos)
    self.trail_idx = 0
    self._reset_grid()
    init_obs = self.gen_obs()
    return init_obs, {}

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

class GridBlindPickEnv(gym.Env):
  def __init__(self, width, height, start_pos):
    self.width = width
    self.height = height
    self.start_pos = start_pos # should be at center somewhere.
    self.goal = [0, 0]

    self.observation_space = spaces.Dict(
      {
        "robot": spaces.Box(
            low=np.array([0.0, 0.0]), # x, on goal or not.
            high=np.array([width,1]),
            shape=(2,),
            dtype="int64",
        ),
        "priv_info": spaces.Box(
            low=np.array([0.0]), #  y
            high=np.array([height]),
            shape=(1,),
            dtype="int64",
        )
      }
    )
    self.action_space = spaces.Discrete(len(Actions))


  def _reset_grid(self):
    self.grid = np.zeros((self.height, self.width), dtype=int)
    # add walls around edges.;w
    self.grid[0, :self.width] = Entities.wall
    self.grid[-1, :self.width] = Entities.wall
    self.grid[:self.height, 0] = Entities.wall
    self.grid[:self.height, -1] = Entities.wall
    self.goal = [np.random.randint(1, self.height-1),  np.random.randint(1, self.width-1)]
    self.grid[self.goal[0],  self.goal[1]] = Entities.trail
    self.grid[self.start_pos[0], self.start_pos[1]] = Entities.agent

  def step(self, action):
    # first do the action
    old_pos = self.curr_pos
    new_pos = self.curr_pos + ACTION_COORDS[action]
    reward = 0
    terminated = truncated = False
    # if at the goal already, stay in absorbing state.
    if np.all(old_pos == self.goal):
      obs = {"robot": np.array([old_pos[1], 1],dtype=np.int64), "priv_info": np.array([old_pos[0]], dtype="int64")}
      reward = 1
      return obs, reward, terminated, truncated, {}

    # if agent is out of bounds or inside a wall, revert back.
    within_bounds =  (0 <= new_pos[0] < self.height) and (0 <= new_pos[1] < self.width)
    if not within_bounds or (self.grid[new_pos[0], new_pos[1]] == Entities.wall):
      new_pos = self.curr_pos
    # if the agent is at the goal, give reward 1. 
    at_goal = False
    if np.all(new_pos == self.goal):
      reward = 1
      at_goal = True

    # update the grid
    self.grid[old_pos[0], old_pos[1]] = Entities.empty
    self.grid[new_pos[0], new_pos[1]] = Entities.agent

    # update curr_pos for next timestep.
    self.curr_pos = new_pos

    obs = {"robot": np.array([self.curr_pos[1], at_goal],dtype=np.int64), "priv_info": np.array([self.curr_pos[0]], dtype="int64")}

    return obs, reward, terminated, truncated, {}

  def reset(self, *, seed=None, options=None):
    self.curr_pos = np.array(self.start_pos)
    self.trail_idx = 0
    self._reset_grid()
    at_goal = np.all(self.curr_pos == self.goal)
    obs = {"robot": np.array([self.curr_pos[1], at_goal],dtype=np.int64), "priv_info": np.array([self.curr_pos[0]], dtype="int64")}
    return obs, {}

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

if __name__ == "__main__":
  width = height = 29 + 2
  start_pos = [15,15]
  env = GridBlindPickEnv(width, height, start_pos)
  env.reset()
  print(env.ascii)
  done = False
  while not done:
    key = input("type in wasdqezx")
    if key in KEY_ACTION_MAP:
      obs, rew, terminated, truncated, info = env.step(KEY_ACTION_MAP[key])
      print("done", done, "| rew", rew)
      print(env.ascii)

  # trail = [[2,2], [3,3], [2,4]]
  # env = NoisyFootballEnv(width, height, start_pos, trail, 2)
  # env.reset()
  # # ipdb.set_trace()
  # print(env.ascii)
  # done = False
  # while not done:
  #   key = input("type in wasdqezx")
  #   if key in KEY_ACTION_MAP:
  #     obs, rew, done, info = env.step(KEY_ACTION_MAP[key])
  #     print("done", done, "| rew", rew)
  #     print(env.ascii)