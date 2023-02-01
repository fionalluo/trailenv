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

class TrailEnv(gym.Env):
  def __init__(self, width, height, start_pos, trail):
    self.width = width
    self.height = height
    self.start_pos = start_pos
    self.trail = trail
    self.trail_idx = 0

    # self.observation_space = spaces.Box(
    #   low=0.0,
    #   high=1.0,
    #   shape=(self.width+self.height,),
    #   dtype="float32",
    # )
    self.observation_space = spaces.Box(
      low=np.array([0.0, 0.0]),
      high=np.array([width, height]),
      shape=(2,),
      dtype="float32",
    )
    self.action_space = spaces.Discrete(len(Actions))

  # def seed(self, seed):
  #   pass

  def _reset_grid(self):
    self.grid = np.zeros((self.width, self.height), dtype=int)
    # add walls around edges.;w
    self.grid[0, :self.width] = Entities.wall
    self.grid[-1, :self.width] = Entities.wall
    self.grid[:self.height, 0] = Entities.wall
    self.grid[:self.height, -1] = Entities.wall
    for tx, ty in self.trail:
      self.grid[ty, tx] = Entities.trail
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
    obs = np.array(self.curr_pos)
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
      Entities.empty: "o",
      Entities.wall: "x",
      Entities.trail: ".",
      Entities.agent: "a",
    }
    for j in range(self.height):
      for i in range(self.width):
        grid_str += ENTITY_TO_STRING[self.grid[j,i]]
      if j < self.height - 1:
          grid_str += "\n"
    return grid_str



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