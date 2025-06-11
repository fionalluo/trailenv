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
    Actions.left: np.array([0, -1]),
    Actions.right: np.array([0, 1]),
    Actions.up: np.array([-1, 0]),
    Actions.down: np.array([1, 0]),
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
    door = 2
    tiger = 3
    button = 4
    wall = 5

# Define colors for visualization
COLORS = {
    'agent': (0, 255, 255),    # cyan
    'button': (255, 192, 203), # pink
    'tiger': (255, 0, 0),      # red
    'door': (0, 255, 0),       # green
    'wall': (128, 128, 128),   # gray
    'empty': (255, 255, 255)   # white
}

class TigerDoorEnv(gym.Env):
    """Tiger Door Environment
    
    A grid-based environment where an agent must navigate to find a door while avoiding a tiger.
    The agent starts in the top-left corner and can press a button at the bottom-left to reveal
    the door's position. The environment features a vertical path in the first column, a horizontal
    path in the middle that forks into two paths, with the tiger and door randomly placed at the
    ends of these forks.

    Observation Space:
        The environment provides both privileged (teacher) and unprivileged (student) observations:

        Privileged (Teacher) Observations:
        - distance: Manhattan distance to the door
        - neighbors: One-hot encoded neighbors (4 directions × 6 entity types)
        - door: +1 if door is above tiger, -1 otherwise
        - image: Full grid visualization
        - is_terminal: 1 if agent reached door/tiger, 0 otherwise
        - position: Current agent position

        Unprivileged (Student) Observations:
        - neighbors_unprivileged: One-hot encoded neighbors where tiger and door appear identical
        - door_unprivileged: Initially 0, becomes ±1 after button press to reveal door position
        - is_terminal: 1 if agent reached door/tiger, 0 otherwise
        - position: Current agent position

    Action Space:
        Discrete(4): up, right, down, left

    Rewards:
        +10: Reaching the door
        -10: Reaching the tiger
        0: All other actions

    Grid Elements:
        - Agent (A): Cyan
        - Button (B): Pink
        - Tiger (T): Red
        - Door (D): Green
        - Wall (#): Gray
        - Empty ( ): White
    """
    
    def __init__(self, row_size=7, col_size=7):
        super().__init__()
        self.row_size = row_size
        self.col_size = col_size
        self.steps = 0
        
        # Initialize empty grid
        self._reset_grid()
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "distance": spaces.Box(low=0, high=row_size + col_size, shape=(1,), dtype=np.int32),
            "neighbors": spaces.Box(low=0, high=1, shape=(4 * len(Entities),), dtype=np.int32),
            "neighbors_unprivileged": spaces.Box(low=0, high=1, shape=(4 * len(Entities),), dtype=np.int32),
            "door": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32),
            "door_unprivileged": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32),
            "image": spaces.Box(low=0, high=255, shape=(row_size, col_size, 3), dtype=np.uint8),
            "large_image": spaces.Box(low=0, high=255, shape=(row_size * 20, col_size * 20, 3), dtype=np.uint8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            "position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.int32),
        })
        
        self.action_space = spaces.Discrete(len(Actions))
        
    def _reset_grid(self):
        """Create the grid with walls, paths, and special cells."""
        self.grid = np.zeros((self.row_size, self.col_size), dtype=int)
        self.grid.fill(Entities.wall)
        
        # Create vertical path in first column
        for i in range(self.row_size):
            self.grid[i, 0] = Entities.empty
            
        # Create horizontal path in middle
        mid_row = self.row_size // 2
        fork_col = self.col_size // 2
        
        # Create horizontal path up to fork
        for j in range(fork_col + 1):  # +1 to include the fork column
            self.grid[mid_row, j] = Entities.empty
            
        # Create fork paths
        for j in range(fork_col, self.col_size):
            self.grid[mid_row - 1, j] = Entities.empty
            self.grid[mid_row + 1, j] = Entities.empty
            
        # Place button at bottom left
        self.button_pos = np.array([self.row_size - 1, 0])
        self.grid[self.button_pos[0], self.button_pos[1]] = Entities.button
        
        # Randomly place tiger and door at the ends of the forks
        end_col = self.col_size - 1
        if random.random() < 0.5:
            self.tiger_pos = np.array([mid_row - 1, end_col])
            self.door_pos = np.array([mid_row + 1, end_col])
        else:
            self.tiger_pos = np.array([mid_row + 1, end_col])
            self.door_pos = np.array([mid_row - 1, end_col])
            
        self.grid[self.tiger_pos[0], self.tiger_pos[1]] = Entities.tiger
        self.grid[self.door_pos[0], self.door_pos[1]] = Entities.door
        
        # Set agent position (top left)
        self.agent_pos = np.array([0, 0])
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entities.agent
        
        # Make a copy of the initial grid for resetting states
        self.initial_grid = deepcopy(self.grid)
        
        self.button_pressed = False
        self.steps = 0
        
    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self._reset_grid()
        return self.gen_obs(), {}
        
    def step(self, action):
        """Take a step in the environment."""
        old_pos = self.agent_pos
        new_pos = self.agent_pos + ACTION_COORDS[action]
        
        # Check if move is valid
        within_bounds = (0 <= new_pos[0] < self.row_size) and (0 <= new_pos[1] < self.col_size)
        if not within_bounds or (self.grid[new_pos[0], new_pos[1]] == Entities.wall):
            new_pos = old_pos
            
        # Update the grid
        self.grid[old_pos[0], old_pos[1]] = Entities.empty  # Clear old position
        self.grid[new_pos[0], new_pos[1]] = Entities.agent
        
        # Update agent position
        self.agent_pos = new_pos
        
        # Check if button is pressed
        if tuple(self.agent_pos) == tuple(self.button_pos):
            self.button_pressed = True
            
        # Calculate reward
        reward = 0
        terminated = False
        
        if tuple(self.agent_pos) == tuple(self.door_pos):
            reward = 10
            terminated = True
        elif tuple(self.agent_pos) == tuple(self.tiger_pos):
            reward = -10
            terminated = True
            
        self.steps += 1
        truncated = False
        obs = self.gen_obs()
        
        return obs, reward, terminated, truncated, {}
        
    def _get_neighbors(self):
        """Get the neighbors of the agent."""
        neighbors = []
        for dr, dc in (-1, 0), (1, 0), (0, -1), (0, 1):  # up down left right
            r, c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if 0 <= r < self.row_size and 0 <= c < self.col_size:
                neighbors.append(self.grid[r, c])
            else:
                neighbors.append(Entities.wall)  # use wall for out of bounds
        return neighbors
        
    def gen_obs(self):
        """Generate the observation dictionary."""
        # Get manhattan distance to door
        distance = np.abs(self.agent_pos[0] - self.door_pos[0]) + np.abs(self.agent_pos[1] - self.door_pos[1])
        
        # Get neighbors and one-hot encode them
        neighbors_raw = self._get_neighbors()
        neighbors = np.zeros((4 * len(Entities),), dtype=np.int32)
        for i, entity in enumerate(neighbors_raw):
            neighbors[len(Entities) * i + entity] = 1
            
        # Get neighbors where tiger and door appear the same
        neighbors_unprivileged = np.zeros((4 * len(Entities),), dtype=np.int32)
        for i, entity in enumerate(neighbors_raw):
            if entity == Entities.tiger or entity == Entities.door:
                neighbors_unprivileged[len(Entities) * i + Entities.tiger] = 1
                neighbors_unprivileged[len(Entities) * i + Entities.door] = 1
            else:
                neighbors_unprivileged[len(Entities) * i + entity] = 1
                
        # Door position indicator
        door_indicator = 1 if self.door_pos[0] < self.tiger_pos[0] else -1
        
        # Door unprivileged indicator
        door_unprivileged = 0
        if self.button_pressed:
            door_unprivileged = door_indicator
            
        # Render the grid as an image
        image = self.render_as_image()
        large_image = self.render_as_large_image()
        
        # Check if terminal
        terminal = 1 if tuple(self.agent_pos) in [tuple(self.door_pos), tuple(self.tiger_pos)] else 0
        
        position = np.array([self.agent_pos[0], self.agent_pos[1]])
        
        return {
            "distance": np.array([distance]),
            "neighbors": neighbors,
            "neighbors_unprivileged": neighbors_unprivileged,
            "door": np.array([door_indicator]),
            "door_unprivileged": np.array([door_unprivileged]),
            "image": image,
            "large_image": large_image,
            "is_terminal": np.array([terminal]),
            "position": position,
        }
        
    def render_as_image(self):
        """Generate an image of the grid."""
        image = np.zeros((self.row_size, self.col_size, 3), dtype=np.uint8)
        for i in range(self.row_size):
            for j in range(self.col_size):
                entity = self.grid[i, j]
                if entity == Entities.agent:
                    image[i, j] = COLORS['agent']
                elif entity == Entities.button:
                    image[i, j] = COLORS['button']
                elif entity == Entities.tiger:
                    image[i, j] = COLORS['tiger']
                elif entity == Entities.door:
                    image[i, j] = COLORS['door']
                elif entity == Entities.wall:
                    image[i, j] = COLORS['wall']
                else:  # empty
                    image[i, j] = COLORS['empty']
        return image
        
    def render_as_large_image(self, cell_size=20):
        """Generate a larger image of the grid."""
        small_image = self.render_as_image()
        rows, cols = small_image.shape[:2]
        large_image = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
        
        for i in range(rows):
            for j in range(cols):
                large_image[i*cell_size:(i+1)*cell_size, 
                          j*cell_size:(j+1)*cell_size] = small_image[i, j]
        return large_image
        
    @property
    def ascii(self):
        """Return ASCII representation of the current grid state."""
        entity_to_char = {
            Entities.empty: ' ',
            Entities.agent: 'A',
            Entities.door: 'D',
            Entities.tiger: 'T',
            Entities.button: 'B',
            Entities.wall: '#',
        }
        
        grid_str = []
        for i in range(self.row_size):
            row = []
            for j in range(self.col_size):
                entity = self.grid[i, j]
                row.append(entity_to_char[entity])
            grid_str.append(''.join(row))
        return '\n'.join(grid_str)
        
    @property
    def ascii_initial(self):
        """Return ASCII representation of the initial grid state."""
        entity_to_char = {
            Entities.empty: ' ',
            Entities.agent: 'A',
            Entities.door: 'D',
            Entities.tiger: 'T',
            Entities.button: 'B',
            Entities.wall: '#',
        }
        
        grid_str = []
        for i in range(self.row_size):
            row = []
            for j in range(self.col_size):
                entity = self.initial_grid[i, j]
                row.append(entity_to_char[entity])
            grid_str.append(''.join(row))
        return '\n'.join(grid_str)

def colorize(string: str, color: str, bold: bool = False, highlight: bool = False) -> str:
    """Returns string surrounded by appropriate terminal colour codes to print colourised text.

    Args:
        string: The message to colourise
        color: Literal values are gray, red, green, yellow, blue, magenta, cyan, white, crimson
        bold: If to bold the string
        highlight: If to highlight the string

    Returns:
        Colourised string
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
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return f"\x1b[{attrs}m{string}\x1b[0m"

def main():
    """Main function to test the environment with keyboard controls."""
    import sys
    
    env = TigerDoorEnv()
    obs, _ = env.reset()
    
    print(colorize("\nTiger Door Environment", "cyan", bold=True))
    print(colorize("Use WASD keys to move the agent.", "white"))
    print(colorize("Press 'q' to quit.", "white"))
    print(colorize("Press 'r' to reset the environment.", "white"))
    print("\nInitial state:")
    print(env.ascii)
    
    while True:
        # Get key press
        key = input().lower()
        
        # Convert key to action
        if key == 'q':
            break
        elif key == 'r':
            obs, _ = env.reset()
            print(colorize("\nEnvironment reset!", "green"))
            print("\nCurrent state:")
            print(env.ascii)
            continue
        elif key == 'w':
            action = Actions.up
        elif key == 's':
            action = Actions.down
        elif key == 'a':
            action = Actions.left
        elif key == 'd':
            action = Actions.right
        else:
            continue
            
        # Take step
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Print current state
        print("\nCurrent state:")
        print(env.ascii)
        print(colorize(f"Reward: {reward}", "yellow"))
        print(colorize(f"Steps: {env.steps}", "blue"))
        print(colorize(f"Button pressed: {env.button_pressed}", "magenta"))
        
        # Print observations
        print("\nObservations:")
        print(colorize(f"Distance to door: {obs['distance'][0]}", "cyan"))
        print(colorize(f"Neighbors: {obs['neighbors']}", "green"))
        print(colorize(f"Neighbors (unprivileged): {obs['neighbors_unprivileged']}", "blue"))
        print(colorize(f"Door position indicator: {obs['door'][0]}", "yellow"))
        print(colorize(f"Door unprivileged indicator: {obs['door_unprivileged'][0]}", "magenta"))
        print(colorize(f"Position: {obs['position']}", "white"))
        
        if terminated:
            if reward > 0:
                print(colorize("\nSuccess! You found the door!", "green", bold=True))
            else:
                print(colorize("\nGame Over! You encountered the tiger!", "red", bold=True))
            obs, _ = env.reset()
            print("\nNew episode started:")
            print(env.ascii)

if __name__ == "__main__":
    main()
