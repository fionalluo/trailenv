from copy import deepcopy
from enum import IntEnum
import random

import numpy as np
from gymnasium import spaces
import gymnasium as gym
import cv2


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
  def __init__(self, size, trail_seed=None, off_center=False):
    self.size = size
    self.trail_seed = trail_seed
    self.off_center = off_center
    self.margin = 3 # margin of empty squares around the lava
    self.trail_forward = []

    # Initialize empty grid
    self._reset_grid()

    # Define observation space
    self.observation_space = spaces.Dict({
      "distance": spaces.Box(low=0, high=size * 2, shape=(1,), dtype=np.int32),  # Manhattan distance
      "neighbors": spaces.Box(low=0, high=1, shape=(4 * len(Entities),), dtype=np.int32),  # One-hot encoded
      "neighbors_unprivileged": spaces.Box(low=0, high=1, shape=(4 * len(Entities),), dtype=np.int32),  # One-hot encoded. Lava and trail appear the same
      "last_action": spaces.Box(low=0, high=1, shape=(len(Actions),), dtype=np.int32),  # One-hot encoded last action
      "image": spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8),  # RGB image of the grid
      "large_image": spaces.Box(low=0, high=255, shape=(self.size * 20, self.size * 20, 3), dtype=np.uint8),  # Larger RGB image of the grid
      # Multibinary is not compatible with dreamer...?
      # "neighbors": spaces.MultiBinary(4 * len(Entities)),  # One-hot encoded neighbors
      # "neighbors_unprivileged": spaces.MultiBinary(4 * len(Entities)),  # One-hot encoded. Lava and trail appear the same
      # "last_action": spaces.MultiBinary(len(Actions)),  # One-hot encoded last action
    })

    self.action_space = spaces.Discrete(len(Actions))
  
  def _reset_grid(self):
    self.grid = np.zeros((self.size, self.size), dtype=int)

    # Set lava center
    size = self.size
    lava_region = self.grid[self.margin: size - self.margin, self.margin: size - self.margin]
    lava_region[:] = Entities.lava

    # Set target position (center top, above lava)
    shift = random.choice([-1, 1]) if self.off_center else 0
    self.grid[self.margin - 1, size // 2 + shift] = Entities.target
    self.target_pos = np.array([self.margin - 1, size // 2 + shift])

    self.robot_pos = np.array([size - self.margin, size // 2])

    # Set trail
    self._set_trail()
    self.visited_trail = set()  # set of visited trail positions

    # Make a copy of the initial grid for resetting states when the agent moves
    self.initial_grid = deepcopy(self.grid)

    # Set robot position (center bottom, below lava)
    self.grid[size - self.margin, size // 2] = Entities.agent
  
  def _set_trail(self):
    reward = 0

    while reward >= 0:

      # Set trail seed
      if self.trail_seed is not None:
        random.seed(self.trail_seed)

      # Start and end positions
      shift = random.choice([1, 0]) if self.off_center else 0
      start_row, start_col = self.robot_pos[0] - 1, self.robot_pos[1] + shift
      end_row, end_col = self.target_pos[0] + 1, self.target_pos[1] - shift
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
              for _ in range(2):  # by doing 2 steps, we ensure paths will wind with margin of at least 1
                next_row, next_col = current_row + dr, current_col + dc
                
                # Ensure movement stays within the lava region and avoids overlapping
                if (
                    self.margin + 1 <= next_row < self.size - self.margin - 1
                    and self.margin + 1 < next_col < self.size - self.margin - 1
                    and self.grid[next_row, next_col] != Entities.trail
                    and not (next_row == start_row - 2 and next_col == start_col)
                ):
                    current_row, current_col = next_row, next_col
                    self.grid[current_row, current_col] = Entities.trail
                    moved = True
                    # break
            
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

      # Get the reward from going up
      # Make sure the reward from going in a straight line up to the target is NEGATIVE
      reward = 0
      for r in range(start_row - 1, end_row, -1):
        if self.grid[r][start_col] == Entities.trail:
          reward += 0.5
        else:
          reward -= 0.2
      # print("Forward reward", reward)
      if reward >= 0:
        size = self.size
        lava_region = self.grid[self.margin: size - self.margin, self.margin: size - self.margin]
        lava_region[:] = Entities.lava
      # print("Trail forward average", sum(self.trail_forward) / len(self.trail_forward))
  
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
      new_pos = old_pos

    # update the grid
    self.grid[old_pos[0], old_pos[1]] = self.initial_grid[old_pos[0], old_pos[1]]
    self.grid[new_pos[0], new_pos[1]] = Entities.agent

    # update curr_pos for next timestep.
    self.robot_pos = new_pos

    # Reward computation. check if agent is on the trail.
    reward = 0
    terminated = False
    curr_cell = self.initial_grid[new_pos[0], new_pos[1]]

    if curr_cell == Entities.trail:
      if tuple(new_pos) not in self.visited_trail:
        reward = 0.5
      else:
        reward = -0.5
    elif curr_cell == Entities.target:
      reward = 10.0
      terminated = True
    elif curr_cell == Entities.lava:
      reward = -0.2

    self.visited_trail.add(tuple(new_pos))
    truncated = False
    obs = self.gen_obs()

    # # Show the grid image in a separate window using OpenCV
    # grid_image = obs["large_image"]  # Get the grid image from observation
    # cv2.imshow("Lava Trail Environment", grid_image)  # Display the image
    # cv2.waitKey(1)  # Wait for a key press for 1 ms, to update the window
    
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
    distance = np.array([distance], dtype=np.int32)

    # Get neighbors and one-hot encode them
    neighbors_raw = self._get_neighbors()
    neighbors = np.zeros((4 * len(Entities),), dtype=np.int32)
    for i, entity in enumerate(neighbors_raw):
        neighbors[len(Entities) * i + entity] = 1  # One-hot encode the neighbor type
    
    # Get neighbors where lava and trail are the same (both marked)
    neighbors_unprivileged = np.zeros((4 * len(Entities),), dtype=np.int32)
    for i, entity in enumerate(neighbors_raw):
      if entity == Entities.lava or entity == Entities.trail:
        neighbors_unprivileged[len(Entities) * i + Entities.lava] = 1
        neighbors_unprivileged[len(Entities) * i + Entities.trail] = 1
      else:
        neighbors_unprivileged[len(Entities) * i + entity] = 1

    # One-hot encode the last action
    last_action = np.zeros(len(Actions), dtype=np.int32)
    if hasattr(self, "last_action"):
        last_action[self.last_action] = 1
    
    # Render the grid as an image
    image = self.render_as_image()
    large_image = self.render_as_large_image()

    # Combine into the observation dictionary
    obs = {
        "distance": distance,
        "neighbors": neighbors,
        "neighbors_unprivileged": neighbors_unprivileged,
        "last_action": last_action,
        "image": image,
        "large_image": large_image,
    }

    # Assert the correct shapes
    assert obs["distance"].shape == (1,)
    assert obs["neighbors"].shape == (4 * len(Entities),)
    assert obs["neighbors_unprivileged"].shape == (4 * len(Entities),)
    assert obs["last_action"].shape == (len(Actions),)
    assert obs["image"].shape == (self.size, self.size, 3)
    assert obs["large_image"].shape == (self.size * 20, self.size * 20, 3)

    return obs

  def render_as_image(self):
    # Generate an image of the grid with visited trail as darker
    img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
    
    ENTITY_COLORS = {
      Entities.empty: [255, 255, 255],  # White
      Entities.wall: [128, 128, 128],   # Gray
      Entities.trail: [0, 0, 255],      # Blue
      Entities.agent: [255, 255, 0],    # Yellow
      Entities.lava: [255, 0, 0],       # Red
      Entities.target: [0, 255, 0],     # Green
    }

    # Update image with grid data and apply darker shades to visited trail
    for i in range(self.size):
      for j in range(self.size):
        entity = self.grid[i, j]
        if entity != Entities.agent and (i, j) in self.visited_trail:
          # Darken the trail if it was visited
          img[i, j] = np.array(ENTITY_COLORS[entity]) * 0.7
        else:
          img[i, j] = ENTITY_COLORS[entity]

    return img

  def render_as_large_image(self, cell_size=20):
    """
    Render the grid as a higher resolution image with borders between cells.
    Each cell is enlarged by `cell_size` pixels, and cells will have black borders.
    """
    # Create an empty image with enough space to expand each cell by `cell_size` pixels
    img = np.zeros((self.size * cell_size, self.size * cell_size, 3), dtype=np.uint8)

    ENTITY_COLORS = {
        Entities.empty: [230, 230, 230],  # White
        Entities.wall: [128, 128, 128],   # Gray
        Entities.trail: [0, 0, 255],      # Blue
        Entities.agent: [255, 255, 0],    # Yellow
        Entities.lava: [255, 0, 0],       # Red
        Entities.target: [0, 255, 0],     # Green
    }

    # Loop over the grid and draw each cell at a larger scale
    for i in range(self.size):
      for j in range(self.size):
        entity = self.grid[i, j]
        # Determine the color for the entity
        color = ENTITY_COLORS[entity]

        # If the cell is part of the visited trail, darken the color
        if (i, j) in self.visited_trail:
          color = np.array(color) * 0.7  # Darken the trail color

        # Scale the position of each cell
        start_x = j * cell_size
        start_y = i * cell_size
        end_x = start_x + cell_size
        end_y = start_y + cell_size

        # Fill the corresponding cell area with the correct color
        img[start_y:end_y, start_x:end_x] = color

        # Draw black outline for each cell (optional, but enhances grid clarity)
        # Only draw borders on the right and bottom to avoid duplicate borders at edges
        if j < self.size - 1:  # right border
          img[start_y:end_y, end_x] = [0, 0, 0]
        if i < self.size - 1:  # bottom border
          img[end_y, start_x:end_x] = [0, 0, 0]

    return img

  
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
  size = height = 16
  env = LavaTrailEnv(size, trail_seed=None)
  # env.reset()
  # print(env.ascii)
  # exit()

  # Interactive mode
  env.reset()
  print(env.ascii)
  done = False
  while not done:
    key = input("type in wasd")
    env.reset()
    print(env.ascii)
    continue
    if key in KEY_ACTION_MAP:
      obs, rew, terminated, truncated, info = env.step(KEY_ACTION_MAP[key])
      print("rew", rew)
      # print("obs", obs)
      dirs = ["up", "down", "left", "right"]
      for i in range(4):
        print(obs["neighbors"][i*len(Entities): (i+1)*len(Entities)], dirs[i])
        print(obs["neighbors_unprivileged"][i*len(Entities): (i+1)*len(Entities)], dirs[i])
      print(env.ascii)