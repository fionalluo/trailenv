from copy import deepcopy
from enum import IntEnum
import random

import numpy as np
from gymnasium import spaces
import gymnasium as gym
import cv2
import jax.numpy as jnp
import jax


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
  agent = 1
  target = 2
  lava = 3
  small_target = 4
  wall = 5

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


class BanditPathEnv(gym.Env):
  def __init__(self, path_length=5, reward_scale=1):
    self.steps = 0
    self.rows = 2 + path_length
    self.cols = 3 + 2 * path_length
    self.path_length = path_length
    self.reward_scale = reward_scale

    # Initialize empty grid
    self._reset_grid()

    # Define observation space
    self.observation_space = spaces.Dict({
      "distance": spaces.Box(low=0, high=self.rows + self.cols, shape=(1,), dtype=np.int32),
      "neighbors": spaces.Box(low=0, high=1, shape=(4 * len(Entities),), dtype=np.int32),  # One-hot encoded
      "neighbors_unprivileged": spaces.Box(low=0, high=1, shape=(4 * len(Entities),), dtype=np.int32),  # One-hot encoded. Lava and trail appear the same
      "target": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
      "target_unprivileged": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
      "image": spaces.Box(low=0, high=255, shape=(self.rows, self.cols, 3), dtype=np.uint8),  # RGB image of the grid
      "large_image": spaces.Box(low=0, high=255, shape=(self.rows * 20, self.cols * 20, 3), dtype=np.uint8),  # Larger RGB image of the grid
      "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),  # 1 if is terminal
      "position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.int32),  # Position of the agent
    })

    self.action_space = spaces.Discrete(len(Actions))
  
  def _reset_grid(self, target_up=None):
    self.grid = np.zeros((self.rows, self.cols), dtype=int)
    self.grid[:-1, :] = Entities.wall
    self.grid[:, self.path_length + 1] = Entities.empty
    self.steps = 0

    self.grid[0, 0] = Entities.wall
    self.grid[0, 2] = Entities.wall
    if random.random() < 0.5 and not target_up or target_up is True:
      self.target_up = True
      self.grid[0, self.path_length + 1] = Entities.target
      self.grid[-1, -1] = Entities.lava
      self.target_pos = np.array([0, self.path_length + 1])
    else:
      self.target_up = False
      self.grid[0, self.path_length + 1] = Entities.lava
      self.grid[-1, -1] = Entities.target
      self.target_pos = np.array([self.rows - 1, self.cols - 1])
    self.grid[-1, 0] = Entities.small_target

    self.robot_pos = np.array([self.rows - 1, self.path_length + 1])

    # Make a copy of the initial grid for resetting states when the agent moves
    self.initial_grid = deepcopy(self.grid)

    # Set robot position (center bottom, below lava)
    self.grid[-1, self.path_length + 1] = Entities.agent

    self.visited_trail = set()
  
  def reset(self, *, seed=None, options=None, target_up=None):
    self._reset_grid(target_up=target_up)
    init_obs = self.gen_obs()
    return init_obs, {}

  def step(self, action):
    # first do the action
    old_pos = self.robot_pos
    new_pos = self.robot_pos + ACTION_COORDS[action]
    self.last_action = action

    # if agent is out of bounds or inside a wall, revert back.
    within_bounds =  (0 <= new_pos[0] < self.rows) and (0 <= new_pos[1] < self.cols)
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
    self.steps += 1

    if curr_cell == Entities.target:
      reward = 1.0
      terminated = True
    elif curr_cell == Entities.lava:
      reward = -1.0
      terminated = True
    elif curr_cell == Entities.empty and tuple(new_pos) not in self.visited_trail:
      reward = 0.01
      self.visited_trail.add(tuple(new_pos))
    elif curr_cell == Entities.empty and tuple(new_pos) in self.visited_trail:
      reward = -0.02
    elif curr_cell == Entities.small_target:
      reward = 0.3
      terminated = True

    truncated = False
    obs = self.gen_obs()

    reward *= self.reward_scale

    # # # Show the grid image in a separate window using OpenCV
    # grid_image = obs["large_image"]  # Get the grid image from observation
    # cv2.imshow("Lava Trail Environment", grid_image)  # Display the image
    # cv2.waitKey(1)  # Wait for a key press for 1 ms, to update the window
    
    return obs, reward, terminated, truncated, {}
  
  def _get_neighbors(self):
    # Get the neighbors of the agent.
    neighbors = []
    for dr, dc in (-1, 0), (1, 0), (0, -1), (0, 1): # up down left right
      r, c = self.robot_pos[0] + dr, self.robot_pos[1] + dc
      if 0 <= r < self.rows and 0 <= c < self.cols:
        neighbors.append(self.grid[r, c])
      else:
        neighbors.append(Entities.wall) # use wall for out of bounds
    return neighbors

  def gen_obs(self):
    # Get manhattan distance to target
    distance = np.abs(self.robot_pos[0] - self.target_pos[0]) + np.abs(self.robot_pos[1] - self.target_pos[1])

    # Get neighbors and one-hot encode them
    neighbors_raw = self._get_neighbors()
    neighbors = np.zeros((4 * len(Entities),), dtype=np.int32)
    for i, entity in enumerate(neighbors_raw):
        neighbors[len(Entities) * i + entity] = 1  # One-hot encode the neighbor type
    
    # Get neighbors where lava and trail are the same (both marked)
    neighbors_unprivileged = np.zeros((4 * len(Entities),), dtype=np.int32)
    for i, entity in enumerate(neighbors_raw):
      if entity == Entities.lava or entity == Entities.target:
        neighbors_unprivileged[len(Entities) * i + Entities.lava] = 1
        neighbors_unprivileged[len(Entities) * i + Entities.target] = 1
      else:
        neighbors_unprivileged[len(Entities) * i + entity] = 1
    
    # Render the grid as an image
    image = self.render_as_image()
    large_image = self.render_as_large_image()

    if self.target_up:
      target = np.array([1])
      target_unprivileged = np.array([0])
    else:
      target = np.array([0])
      target_unprivileged = np.array([0])
    
    terminal = 0
    if self.grid[self.robot_pos[0], self.robot_pos[1]] in [Entities.small_target, Entities.target, Entities.lava]:
      terminal = 1

    position = np.array([self.robot_pos[0], self.robot_pos[1]])

    # Combine into the observation dictionary
    obs = {
        "distance": np.array([distance]),
        "neighbors": neighbors,
        "neighbors_unprivileged": neighbors_unprivileged,
        "target": target,
        "target_unprivileged": target_unprivileged,
        "image": image,
        "large_image": large_image,
        "is_terminal": terminal,
        "position": position,
    }

    # Assert the correct shapes
    assert obs["distance"].shape == (1,)
    assert obs["neighbors"].shape == (4 * len(Entities),)
    assert obs["neighbors_unprivileged"].shape == (4 * len(Entities),)
    assert obs["target"].shape == (1,)
    assert obs["target_unprivileged"].shape == (1,)
    assert obs["image"].shape == (self.rows, self.cols, 3)
    assert obs["large_image"].shape == (self.rows * 20, self.cols * 20, 3)
    assert obs["position"].shape == (2,)

    return obs

  def get_fixed_trail(self, action: int):
    """
    Perform the specified action until the environment is finished.
    Returns the trajectory of all actions and observations.
    Always have the target spawn up.
    
    Parameters:
      action (int): The fixed action to perform at each step.

    Returns:
      dict: A dictionary containing:
        - 'actions': A list of actions taken (including an initial all-zero action).
        - 'observations': A list of observations collected.
    """
    # Reset the environment and initialize trajectory
    observation, _ = self.reset(target_up=True)  # Reset the environment
    observations = [observation]  # Store the first observation
    actions = [np.zeros_like(action)]  # First "action" is an all-zero placeholder
    
    done = False
    while not done:
      observation, reward, done, truncated, info = self.step(action)  # Perform the fixed action
      observations.append(observation)  # Append the observation
      actions.append(action)  # Append the fixed action

    # Return the trajectory as a dictionary
    return {
      "actions": actions,
      "observations": observations,
    }

  def render_reward_penalty(self, kl_div_lru: list[list[float]], cell_size=50):
    """Render the reward penalty for the current state."""
    rows = 2 + self.path_length
    cols = 3 + 2 * self.path_length
    grid = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Calculate the max KL divergence for normalization
    max_kl_div = max([max(kl_div) for kl_div in kl_div_lru])
    kl_div_left, kl_div_right, kl_div_up = kl_div_lru

    # Fill the grid with walls (gray)
    grid[:, :] = [128, 128, 128]  # Gray for walls

    # Define colors
    base_purple = np.array([200, 0, 200])  # Vivid purple
    def get_color(value):
      intensity = value / max_kl_div
      return ((1 - intensity) * 255 + intensity * base_purple).astype(np.uint8)

    # Color left
    for c, i in zip(range(self.path_length + 1, -1, -1), range(self.path_length + 2)):
      grid[rows - 1, c] = get_color(kl_div_left[i])

    # Color right
    for c, i in zip(range(self.path_length + 1, cols), range(self.path_length + 2)):
      grid[rows - 1, c] = get_color(kl_div_right[i])

    # Color up
    for r, i in zip(range(rows - 1, -1, -1), range(self.path_length + 2)):
      grid[r, self.path_length + 1] = get_color(kl_div_up[i])

    # Define special grid positions
    grid[0, self.path_length + 1] = [0, 255, 0]  # Green for target up
    grid[rows - 1, 0] = [255, 255, 0]  # Yellow for target small
    grid[rows - 1, cols - 1] = [255, 0, 0]  # Red for lava

    # Scale up the grid for visualization
    large_grid = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    for i in range(rows):
      for j in range(cols):
        start_x = j * cell_size
        start_y = i * cell_size
        end_x = start_x + cell_size
        end_y = start_y + cell_size
        large_grid[start_y:end_y, start_x:end_x] = grid[i, j]

    # Add KL divergence text to the scaled grid
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55  # Larger font scale for clarity
    thickness = 1  # Thickness for better visibility

    # Add KL divergences (left, right, up)
    for c, i in zip(range(self.path_length, -1, -1), range(1, self.path_length + 1)):  # Left
      text = f"{kl_div_left[i]:.2f}"
      pos = (c * cell_size + cell_size // 10, (rows - 1) * cell_size + cell_size // 2 + 5)  # Slightly shifted
      cv2.putText(large_grid, text, pos, font, font_scale, (0, 0, 0), thickness)

    for c, i in zip(range(self.path_length + 2, cols), range(1, self.path_length + 1)):  # Right
      text = f"{kl_div_right[i]:.2f}"
      pos = (c * cell_size + cell_size // 10, (rows - 1) * cell_size + cell_size // 2 + 5)  # Slightly shifted
      cv2.putText(large_grid, text, pos, font, font_scale, (0, 0, 0), thickness)

    for r, i in zip(range(rows - 1, -1, -1), range(self.path_length + 1)):  # Up
      text = f"{kl_div_up[i]:.2f}"
      pos = ((self.path_length + 1) * cell_size + cell_size // 10, r * cell_size + cell_size // 2 + 5)  # Slightly shifted
      cv2.putText(large_grid, text, pos, font, font_scale, (0, 0, 0), thickness)

    return large_grid

  def render_as_image(self):
    # Generate an image of the grid with visited trail as darker
    img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
    
    ENTITY_COLORS = {
      Entities.empty: [255, 255, 255],  # White
      Entities.wall: [128, 128, 128],   # Gray
      Entities.agent: [0, 255, 255],   # Cyan
      Entities.lava: [255, 0, 0],       # Red
      Entities.target: [0, 255, 0],     # Green
      Entities.small_target: [255, 255, 0],   # Yellow
    }

    # Update image with grid data and apply darker shades to visited trail
    for i in range(self.rows):
      for j in range(self.cols):
        entity = self.grid[i, j]
        img[i, j] = ENTITY_COLORS[entity]

    return img

  def render_as_large_image(self, cell_size=20):
    """
    Render the grid as a higher resolution image with borders between cells.
    Each cell is enlarged by `cell_size` pixels, and cells will have black borders.
    """
    # Create an empty image with enough space to expand each cell by `cell_size` pixels
    img = np.zeros((self.rows * cell_size, self.cols * cell_size, 3), dtype=np.uint8)

    ENTITY_COLORS = {
        Entities.empty: [230, 230, 230],  # White
        Entities.wall: [128, 128, 128],   # Gray
        Entities.agent: [0, 255, 255],   # Cyan
        Entities.lava: [255, 0, 0],       # Red
        Entities.target: [0, 255, 0],     # Green
        Entities.small_target: [255, 255, 0],   # Yellow
    }

    # Loop over the grid and draw each cell at a larger scale
    for i in range(self.rows):
      for j in range(self.cols):
        entity = self.grid[i, j]
        # Determine the color for the entity
        color = ENTITY_COLORS[entity]

        # Scale the position of each cell
        start_x = j * cell_size
        start_y = i * cell_size
        end_x = start_x + cell_size
        end_y = start_y + cell_size

        # Fill the corresponding cell area with the correct color
        img[start_y:end_y, start_x:end_x] = color

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
      Entities.agent: colorize("■", "cyan"),
      Entities.lava: colorize("■", "red"),
      Entities.target: colorize("■", "green"),
      Entities.small_target: colorize("■", "yellow"),
    }
    for j in range(self.rows):
      for i in range(self.cols):
        grid_str += ENTITY_TO_STRING[self.grid[j,i]]
      if j < self.cols - 1:
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
      Entities.agent: colorize("■", "cyan"),
      Entities.lava: colorize("■", "red"),
      Entities.target: colorize("■", "green"),
      Entities.small_target: colorize("■", "yellow"),
    }
    for j in range(self.rows):
      for i in range(self.cols):
        grid_str += ENTITY_TO_STRING[self.initial_grid[j,i]]
      if j < self.cols - 1:
          grid_str += "\n"
          
    return grid_str

if __name__ == "__main__":
  env = BanditPathEnv()
  # env.reset()
  # print(env.ascii)
  # exit()

  # Interactive mode
  env.reset()
  print(env.ascii)
  
  done = False
  total = 0
  up = 0
  while not done:
    key = input("type in wasd")
    # env.reset()
    # print(env.ascii)
    # print("target", env.gen_obs()["target"])
    # total += 1
    # if env.gen_obs()["target"][0] == 1:
    #   up += 1
    # print("target_unprivileged", env.gen_obs()["target_unprivileged"])
    # print(up / total)
    # continue
    if key in KEY_ACTION_MAP:
      obs, rew, terminated, truncated, info = env.step(KEY_ACTION_MAP[key])
      print("rew", rew)
      print("target", obs["target"])
      print("target_unprivileged", obs["target_unprivileged"])
      # dirs = ["up", "down", "left", "right"]
      # for i in range(4):
      #   print(obs["neighbors"][i*len(Entities): (i+1)*len(Entities)], dirs[i])
      #   print(obs["neighbors_unprivileged"][i*len(Entities): (i+1)*len(Entities)], dirs[i])
      print(env.ascii)