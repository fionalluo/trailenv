"""
Search Environment

A simple grid-based environment where an agent must navigate to find a goal.
The environment consists of a square grid where the agent and goal are randomly placed.
The agent must navigate to reach the goal while avoiding going out of bounds.

Observation Space:
    The environment provides both privileged (teacher) and unprivileged (student) observations:

    Privileged (Teacher) Observations:
    - distance: Manhattan distance to the goal
    - neighbors: One-hot encoded neighbors (4 directions × 3 entity types)
    - image: Full grid visualization
    - is_terminal: 1 if agent reached goal, 0 otherwise
    - goal_position: Current goal position (row, col)
    - agent_position: Current agent position (row, col)

    Unprivileged (Student) Observations:
    - neighbors: One-hot encoded neighbors (4 directions × 3 entity types)
    - image: Full grid visualization
    - is_terminal: 1 if agent reached goal, 0 otherwise
    - agent_position: Current agent position (row, col)

Action Space:
    Discrete(4): up, right, down, left

Rewards:
    +100: Reaching the goal
     0: All other actions

Grid Elements:
    - Agent (A): Cyan
    - Goal (G): Green
    - Empty ( ): White
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

class Entities(IntEnum):
    empty = 0
    agent = 1
    goal = 2

class SearchEnv(gym.Env):
    """Search Environment
    
    A simple grid-based environment where an agent must navigate to find a goal.
    The agent and goal are randomly placed on a square grid, and the agent must
    navigate to reach the goal.

    Observation Space:
        - distance: Manhattan distance to the goal
        - neighbors: One-hot encoded neighbors (4 directions × 3 entity types)
        - image: Full grid visualization
        - is_terminal: 1 if agent reached goal, 0 otherwise
        - goal_position: Current goal position
        - agent_position: Current agent position

    Action Space:
        Discrete(4): up, right, down, left

    Rewards:
        +100: Reaching the goal
         0: All other actions

    Grid Elements:
        - Agent (A): Cyan
        - Goal (G): Green
        - Empty ( ): White
    """
    
    # Define colors for visualization
    COLORS = {
        'agent': (0, 255, 255),    # cyan
        'goal': (0, 255, 0),       # green
        'empty': (255, 255, 255)   # white
    }
    
    def __init__(self, size=7):
        super().__init__()
        self.size = size
        self.steps = 0
        
        # Initialize empty grid
        self._reset_grid()
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "distance": spaces.Box(low=0, high=2*size, shape=(1,), dtype=np.int32),
            "neighbors": spaces.Box(low=0, high=1, shape=(4 * len(Entities),), dtype=np.int32),
            "image": spaces.Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            "goal_position": spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32),
            "agent_position": spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32),
        })
        
        self.action_space = spaces.Discrete(len(Actions))
        
    def _reset_grid(self):
        """Create the grid with random agent and goal positions."""
        # Initialize grid with empty cells
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # Randomly place agent and goal
        positions = random.sample([(i, j) for i in range(self.size) for j in range(self.size)], 2)
        self.agent_pos = np.array(positions[0])
        self.goal_pos = np.array(positions[1])
        
        # Update grid
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entities.agent
        self.grid[self.goal_pos[0], self.goal_pos[1]] = Entities.goal
        
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
                neighbors.append(Entities.empty)  # treat out of bounds as empty
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
            
        # Update the grid
        self.grid[old_pos[0], old_pos[1]] = Entities.empty
        self.grid[new_pos[0], new_pos[1]] = Entities.agent
        
        # Update agent position
        self.agent_pos = new_pos
        
        # Calculate reward
        reward = 0
        terminated = False
        
        if tuple(self.agent_pos) == tuple(self.goal_pos):
            reward = 100
            terminated = True
            
        self.steps += 1
        truncated = False
        obs = self.gen_obs()
        
        return obs, reward, terminated, truncated, {}
        
    def gen_obs(self):
        """Generate the observation dictionary."""
        # Get manhattan distance to goal
        distance = np.abs(self.agent_pos[0] - self.goal_pos[0]) + np.abs(self.agent_pos[1] - self.goal_pos[1])
        
        # Get neighbors and one-hot encode them
        neighbors_raw = self._get_neighbors()
        neighbors = np.zeros((4 * len(Entities),), dtype=np.int32)
        for i, entity in enumerate(neighbors_raw):
            neighbors[len(Entities) * i + entity] = 1
            
        # Render the grid as an image
        image = self.render_as_image()
        
        # Check if terminal
        terminal = 1 if tuple(self.agent_pos) == tuple(self.goal_pos) else 0
        
        return {
            "distance": np.array([distance]),
            "neighbors": neighbors,
            "image": image,
            "is_terminal": np.array([terminal]),
            "goal_position": self.goal_pos,
            "agent_position": self.agent_pos,
        }
        
    def render_as_image(self):
        """Generate an image of the grid."""
        image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for i in range(self.size):
            for j in range(self.size):
                entity = self.grid[i, j]
                if entity == Entities.agent:
                    image[i, j] = self.COLORS['agent']
                elif entity == Entities.goal:
                    image[i, j] = self.COLORS['goal']
                else:  # empty
                    image[i, j] = self.COLORS['empty']
        return image
        
    @property
    def ascii(self):
        """Return ASCII representation of the current grid state."""
        entity_to_char = {
            Entities.empty: ' ',
            Entities.agent: 'A',
            Entities.goal: 'G',
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
    
    env = SearchEnv()
    obs, _ = env.reset()
    
    print("\nSearch Environment")
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
        print(f"Distance to goal: {obs['distance'][0]}")
        print(f"Neighbors: {obs['neighbors']}")
        print(f"Goal position: {obs['goal_position']}")
        print(f"Agent position: {obs['agent_position']}")
        
        if terminated:
            print("\nSuccess! You reached the goal!")
            obs, _ = env.reset()
            print("\nNew episode started:")
            print(env.ascii)

if __name__ == "__main__":
    main()
