"""
Clean Environment

A grid-based environment where an agent must clean all dirty cells in the grid.
The agent starts at a random position and leaves clean cells in its path.
The goal is to clean all cells in the grid.

Observation Space:
    The environment provides both privileged (teacher) and unprivileged (student) observations:

    Privileged (Teacher) Observations:
    - neighbors: One-hot encoded neighbors (4 directions × 3 entity types)
    - image: Full grid visualization
    - is_terminal: 1 if all cells are clean, 0 otherwise
    - position: Current agent position (row, col)

    Unprivileged (Student) Observations:
    - neighbors: One-hot encoded neighbors (4 directions × 3 entity types)
    - is_terminal: 1 if all cells are clean, 0 otherwise
    - position: Current agent position (row, col)

Action Space:
    Discrete(4): up, right, down, left

Rewards:
    +100: Cleaning all cells
    +0.5: Cleaning a new cell
    0: Moving to an already clean cell

Grid Elements:
    - Agent (A): Cyan
    - Dirty (D): Light Green
    - Clean ( ): White
"""

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
    clean = 0
    agent = 1
    dirty = 2
    wall = 3  # New entity type for out of bounds

# Define colors for visualization
COLORS = {
    'agent': (0, 255, 255),    # cyan
    'dirty': (144, 238, 144),  # light green
    'clean': (255, 255, 255),  # white
    'wall': (128, 128, 128)    # gray
}

class CleanEnv(gym.Env):
    def __init__(self, size=7):
        super().__init__()
        self.size = size
        self.steps = 0
        
        # Initialize grid
        self._reset_grid()
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "neighbors": spaces.Box(low=0, high=1, shape=(4 * len(Entities),), dtype=np.int32),
            "image": spaces.Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            "position": spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32),
        })
        
        self.action_space = spaces.Discrete(len(Actions))
        
    def _reset_grid(self):
        """Create the grid with all cells dirty and random agent position."""
        # Initialize grid with dirty cells
        self.grid = np.ones((self.size, self.size), dtype=int) * Entities.dirty
        
        # Randomly place agent
        self.agent_pos = np.array([
            random.randint(0, self.size-1),
            random.randint(0, self.size-1)
        ])
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entities.agent
        
        # Make a copy of the initial grid for resetting states
        self.initial_grid = deepcopy(self.grid)
        self.steps = 0
        
    def _get_neighbors(self):
        """Get the neighbors of the agent."""
        neighbors = []
        for dr, dc in (-1, 0), (1, 0), (0, -1), (0, 1):  # up down left right
            r, c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if 0 <= r < self.size and 0 <= c < self.size:
                neighbors.append(self.grid[r, c])
            else:
                neighbors.append(Entities.wall)  # treat out of bounds as wall
        return neighbors
        
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
        within_bounds = (0 <= new_pos[0] < self.size) and (0 <= new_pos[1] < self.size)
        if not within_bounds:
            new_pos = old_pos
            
        # Calculate reward
        reward = 0
        if within_bounds:
            # Check if we're cleaning a new cell
            if self.grid[new_pos[0], new_pos[1]] == Entities.dirty:
                reward = 0.5
                
        # Update the grid
        self.grid[old_pos[0], old_pos[1]] = Entities.clean
        self.grid[new_pos[0], new_pos[1]] = Entities.agent
        
        # Update agent position
        self.agent_pos = new_pos
        
        # Check if all cells are clean
        terminated = np.all(self.grid != Entities.dirty)
        if terminated:
            reward = 100
            
        self.steps += 1
        truncated = False
        obs = self.gen_obs()
        
        return obs, reward, terminated, truncated, {}
        
    def gen_obs(self):
        """Generate the observation dictionary."""
        # Get neighbors and one-hot encode them
        neighbors_raw = self._get_neighbors()
        neighbors = np.zeros((4 * len(Entities),), dtype=np.int32)
        for i, entity in enumerate(neighbors_raw):
            neighbors[len(Entities) * i + entity] = 1
            
        # Render the grid as an image
        image = self.render_as_image()
        
        # Check if terminal
        terminal = 1 if np.all(self.grid != Entities.dirty) else 0
        
        return {
            "neighbors": neighbors,
            "image": image,
            "is_terminal": np.array([terminal]),
            "position": self.agent_pos,
        }
        
    def render_as_image(self):
        """Generate an image of the grid."""
        image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for i in range(self.size):
            for j in range(self.size):
                entity = self.grid[i, j]
                if entity == Entities.agent:
                    image[i, j] = COLORS['agent']
                elif entity == Entities.dirty:
                    image[i, j] = COLORS['dirty']
                else:  # clean
                    image[i, j] = COLORS['clean']
        return image
        
    @property
    def ascii(self):
        """Return ASCII representation of the current grid state."""
        entity_to_char = {
            Entities.clean: ' ',
            Entities.agent: 'A',
            Entities.dirty: 'D',
            Entities.wall: '#',
        }
        
        grid_str = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                entity = self.grid[i, j]
                row.append(entity_to_char[entity])
            grid_str.append(''.join(row))
        return '\n'.join(grid_str)

def main():
    """Main function to test the environment with keyboard controls."""
    import sys
    
    env = CleanEnv()
    obs, _ = env.reset()
    
    print("\nClean Environment")
    print("Use WASD keys to move the agent.")
    print("Press 'q' to quit.")
    print("Press 'r' to reset the environment.")
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
            print("\nEnvironment reset!")
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
        print(f"Reward: {reward}")
        print(f"Steps: {env.steps}")
        
        # Print observations
        print("\nObservations:")
        print(f"Neighbors: {obs['neighbors']}")
        print(f"Position: {obs['position']}")
        
        if terminated:
            print("\nSuccess! All cells are clean!")
            obs, _ = env.reset()
            print("\nNew episode started:")
            print(env.ascii)

if __name__ == "__main__":
    main()
