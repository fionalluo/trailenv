from copy import deepcopy
from enum import IntEnum
import random

import numpy as np
from gymnasium import spaces
import gymnasium as gym


class Actions(IntEnum):
  left = 0
  right = 1
  up = 2
  down = 3

ACTION_COORDS = {
  Actions.left: np.array([0,-1]),
  Actions.right: np.array([0,1]),
  Actions.up: np.array([-1,0]),
  Actions.down: np.array([1,0]),
}

KEY_ACTION_MAP = {
  "w": Actions.up,
  "a": Actions.left,
  "s": Actions.down,
  "d": Actions.right,
}

class Entities(IntEnum):
  empty = 0
  trail = 1
  agent = 2
  target = 3
  wall = 4
  lava = 5

"""A set of common utilities used within the environments."""

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


class LavaTrailEnv(gym.Env):
  def __init__(self, size, trail_seed=0, priv=True):
    self.size = size
    self.trail_seed = trail_seed
    self.margin = 3 # margin of empty squares around the lava

    # Initialize empty grid
    self._reset_grid()

    # Define observation space
    self.observation_space = spaces.Dict({
      "distance": spaces.Box(low=0, high=size * 2, shape=(), dtype=np.int32),  # Manhattan distance
      "neighbors": spaces.MultiBinary([4, len(Entities)]),  # One-hot encoded neighbors
      "neighbors_unprivileged": spaces.MultiBinary([4, len(Entities)]),  # One-hot encoded. Lava and trail appear the same
      "last_action": spaces.MultiBinary(len(Actions)),  # One-hot encoded last action
    })

    self.action_space = spaces.Discrete(len(Actions))
  
  def _reset_grid(self):
    self.grid = np.zeros((self.size, self.size), dtype=int)

    # Set lava center
    size = self.size
    lava_region = self.grid[self.margin: size - self.margin, self.margin: size - self.margin]
    lava_region[:] = Entities.lava

    # Set target position (center top, above lava)
    self.grid[self.margin - 1, size // 2] = Entities.target
    self.target_pos = np.array([self.margin - 1, size // 2])

    self.robot_pos = np.array([size - self.margin, size // 2])

    # Set trail
    self._set_trail()
    self.visited_trail = set()  # set of visited trail positions

    # Make a copy of the initial grid for resetting states when the agent moves
    self.initial_grid = deepcopy(self.grid)

    # Set robot position (center bottom, below lava)
    self.grid[size - self.margin, size // 2] = Entities.agent
  
  def _set_trail(self):
    # Set trail seed
    random.seed(self.trail_seed)

    # Start and end positions
    start_row, start_col = self.robot_pos[0] - 1, self.robot_pos[1]
    end_row, end_col = self.target_pos[0] + 1, self.target_pos[1]
    current_row, current_col = start_row, start_col

    # Define basic movement directions: up, left, right
    directions = [(-1, 0), (0, -1), (0, 1)]
    
    # Mark the starting position
    self.grid[current_row, current_col] = Entities.trail

    # Generate the trail
    while (current_row, current_col) != (end_row, end_col):
        # Attempt to move forward or turn left/right
        moved = False
        for dr, dc in random.sample(directions, len(directions)):
            next_row, next_col = current_row + dr, current_col + dc
            
            # Ensure movement stays within the lava region and avoids overlapping
            if (
                self.margin + 1 <= next_row < self.size - self.margin - 1
                and self.margin + 1 < next_col < self.size - self.margin - 1
                and self.grid[next_row, next_col] != Entities.trail
            ):
                current_row, current_col = next_row, next_col
                self.grid[current_row, current_col] = Entities.trail
                moved = True
                break
        
        # If no valid move is possible, break out (unlikely but safe)
        if not moved:
            break
        if current_row == end_row + 1:
          break
    
    # Move from the last trail position to the target position
    col_dir = 1 if current_col < end_col else -1
    while current_col != end_col:
        current_col += col_dir
        self.grid[current_row, current_col] = Entities.trail

    # Mark the target position as part of the trail
    self.grid[end_row, end_col] = Entities.trail
  
  def reset(self, *, seed=None, options=None):
    self._reset_grid()
    init_obs = self.gen_obs()
    return init_obs, {}

  def step(self, action):
    # first do the action
    old_pos = self.robot_pos
    new_pos = self.robot_pos + ACTION_COORDS[action]
    self.last_action = action

    # if agent is out of bounds or inside a wall, revert back.
    within_bounds =  (0 <= new_pos[0] < self.size) and (0 <= new_pos[1] < self.size)
    if not within_bounds or (self.grid[new_pos[0], new_pos[1]] == Entities.wall):
      new_pos = self.curr_pos

    # update the grid
    self.grid[old_pos[0], old_pos[1]] = self.initial_grid[old_pos[0], old_pos[1]]
    self.grid[new_pos[0], new_pos[1]] = Entities.agent

    # update curr_pos for next timestep.
    self.robot_pos = new_pos

    # Reward computation. check if agent is on the trail.
    reward = 0
    terminated = False
    curr_cell = self.initial_grid[new_pos[0], new_pos[1]]
    self.visited_trail.add(tuple(new_pos))

    if curr_cell == Entities.trail:
      if tuple(new_pos) not in self.visited_trail:
        reward = 0.5
      else:
        reward = -0.5
    elif curr_cell == Entities.target:
      reward = 10.0
      terminated = True
    elif curr_cell == Entities.lava:
      reward = -0.1

    truncated = False
    obs = self.gen_obs()
    return obs, reward, terminated, truncated, {}
  
  def _manhattan_distance(self):
    return abs(self.robot_pos[0] - self.target_pos[0]) + abs(self.robot_pos[1] - self.target_pos[1])
  
  def _get_neighbors(self):
    # Get the neighbors of the agent.
    neighbors = []
    for dr, dc in (-1, 0), (1, 0), (0, -1), (0, 1): # up down left right
      r, c = self.robot_pos[0] + dr, self.robot_pos[1] + dc
      if 0 <= r < self.size and 0 <= c < self.size:
        neighbors.append(self.grid[r, c])
      else:
        neighbors.append(Entities.wall) # use wall for out of bounds
    return neighbors

  def gen_obs(self):
    # Compute Manhattan distance to the target
    distance = self._manhattan_distance()

    # Get neighbors and one-hot encode them
    neighbors_raw = self._get_neighbors()
    neighbors = np.zeros((4, len(Entities)), dtype=np.int32)
    for i, entity in enumerate(neighbors_raw):
        neighbors[i, entity] = 1  # One-hot encode the neighbor type
    
    # Get neighbors where lava and trail are the same (both marked)
    neighbors_unprivileged = np.zeros((4, len(Entities)), dtype=np.int32)
    for i, entity in enumerate(neighbors_raw):
      if entity == Entities.lava or entity == Entities.trail:
        neighbors_unprivileged[i, Entities.lava] = 1
        neighbors_unprivileged[i, Entities.trail] = 1
      else:
        neighbors_unprivileged[i, entity] = 1

    # One-hot encode the last action
    last_action = np.zeros(len(Actions), dtype=np.int32)
    if hasattr(self, "last_action"):
        last_action[self.last_action] = 1

    # Combine into the observation dictionary
    obs = {
        "distance": distance,
        "neighbors": neighbors,
        "neighbors_unprivileged": neighbors_unprivileged,
        "last_action": last_action
    }
    return obs
  
  @property
  def ascii(self):
    """
    Produce a pretty string of the environment's grid along with the agent.
    """
    grid_str = ""

    ENTITY_TO_STRING = {
      Entities.empty: colorize("■", "white"),
      Entities.wall: colorize("■", "gray"),
      Entities.trail: colorize("■", "blue"),
      Entities.agent: colorize("■", "yellow"),
      Entities.lava: colorize("■", "red"),
      Entities.target: colorize("■", "green"),
    }
    for j in range(self.size):
      for i in range(self.size):
        if self.grid[j,i] == Entities.trail and (j,i) in self.visited_trail:
          grid_str += colorize("■", "cyan") # visited trail on the trail is cyan
        elif self.grid[j, i] != Entities.agent and (j, i) in self.visited_trail:
          grid_str += colorize("■", "magenta") # visited trail anywhere else is magenta
        else:
          grid_str += ENTITY_TO_STRING[self.grid[j,i]]
      if j < self.size - 1:
          grid_str += "\n"
          
    return grid_str

  @property
  def ascii_initial(self):
    """
    Produce a pretty string of the environment's initial grid along with the agent.
    """
    grid_str = ""

    ENTITY_TO_STRING = {
      Entities.empty: colorize("■", "white"),
      Entities.wall: colorize("■", "gray"),
      Entities.trail: colorize("■", "blue"),
      Entities.agent: colorize("■", "yellow"),
      Entities.lava: colorize("■", "red"),
      Entities.target: colorize("■", "green"),
    }
    for j in range(self.size):
      for i in range(self.size):
        grid_str += ENTITY_TO_STRING[self.initial_grid[j,i]]
      if j < self.size - 1:
          grid_str += "\n"
          
    return grid_str

if __name__ == "__main__":
  size = height = 32
  env = LavaTrailEnv(size, trail_seed=0)
  # env.reset()
  # print(env.ascii)
  # exit()

  # Interactive mode
  env.reset()
  print(env.ascii)
  done = False
  while not done:
    key = input("type in wasd")
    if key in KEY_ACTION_MAP:
      obs, rew, terminated, truncated, info = env.step(KEY_ACTION_MAP[key])
      print("rew", rew)
      print("obs", obs)
      print(env.ascii)
