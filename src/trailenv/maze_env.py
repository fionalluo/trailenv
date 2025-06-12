"""
Maze Environment

A procedurally generated maze environment where an agent must navigate from the bottom-left
corner to one of three possible goal positions (upper-left, upper-right, or lower-right).
The maze is generated using a depth-first search algorithm ensuring single-width paths
and proper connectivity to all corners.

Observation Space:
    The environment provides both privileged (teacher) and unprivileged (student) observations:

    Privileged (Teacher) Observations:
    - distance: Manhattan distance to the goal
    - neighbors: 8 × len(entities) one-hot encoded neighbors (3x3 or 5x5 region around agent)
    - image: Full maze visualization
    - is_terminal: 1 if agent reached goal, 0 otherwise
    - goal_position: One-hot encoded goal position (upper-left, upper-right, lower-right)

    Unprivileged (Student) Observations:
    - neighbors: 8 × len(entities) one-hot encoded neighbors (3x3 or 5x5 region around agent)
    - image: Full maze visualization
    - is_terminal: 1 if agent reached goal, 0 otherwise

Action Space:
    Discrete(4): up, right, down, left

Rewards:
    +100: Reaching the goal
     0: All other actions

Grid Elements:
    - Agent (A): Cyan
    - Goal (G): Green
    - Wall (#): Gray
    - Empty ( ): White
    - Visited (V): Light Yellow
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
    wall = 3
    visited = 4  # New entity type for visited cells

class MazeEnv(gym.Env):
    # Define colors for visualization
    COLORS = {
        'agent': (0, 255, 255),    # cyan
        'goal': (0, 255, 0),       # green
        'wall': (128, 128, 128),   # gray
        'empty': (255, 255, 255),  # white
        'visited': (255, 255, 128) # light yellow
    }
    
    def __init__(self, size=7, large_view=False):
        super().__init__()
        self.size = size
        self.large_view = large_view
        self.steps = 0
        
        # Initialize empty grid
        self._reset_grid()
        
        # Define observation space
        neighbor_size = 24 if large_view else 8  # 5x5-1 or 3x3-1 neighbors
        self.observation_space = spaces.Dict({
            "distance": spaces.Box(low=0, high=2*size, shape=(1,), dtype=np.int32),
            "neighbors": spaces.Box(low=0, high=1, shape=(neighbor_size * len(Entities),), dtype=np.int32),
            "image": spaces.Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            "goal_position": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int32),
        })
        
        self.action_space = spaces.Discrete(len(Actions))
        
    def _reset_grid(self):
        """Create a procedurally generated maze."""
        # Initialize grid with walls
        self.grid = np.ones((self.size, self.size), dtype=int) * Entities.wall
        
        # Start DFS from bottom-left corner
        start_pos = (self.size - 1, 0)
        self._generate_maze(start_pos)
        
        # Set agent position
        self.agent_pos = np.array(start_pos)
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entities.agent
        
        # Randomly choose goal position
        possible_goals = [(0, 0), (0, self.size-1), (self.size-1, self.size-1)]
        self.goal_pos = np.array(random.choice(possible_goals))
        self.grid[self.goal_pos[0], self.goal_pos[1]] = Entities.goal
        
        # Store goal position index (0: upper-left, 1: upper-right, 2: lower-right)
        self.goal_idx = possible_goals.index(tuple(self.goal_pos))
        
        # Make a copy of the initial grid for resetting states
        self.initial_grid = deepcopy(self.grid)
        self.steps = 0
        
    def _generate_maze(self, start_pos):
        """Generate maze using DFS algorithm."""
        def get_neighbors(pos):
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                new_pos = (pos[0] + dr, pos[1] + dc)
                if (0 <= new_pos[0] < self.size and 
                    0 <= new_pos[1] < self.size and 
                    self.grid[new_pos[0], new_pos[1]] == Entities.wall):
                    neighbors.append(new_pos)
            return neighbors
        
        def carve_path(pos):
            self.grid[pos[0], pos[1]] = Entities.empty
            neighbors = get_neighbors(pos)
            random.shuffle(neighbors)
            
            for next_pos in neighbors:
                if self.grid[next_pos[0], next_pos[1]] == Entities.wall:
                    # Carve path between current and next position
                    mid_pos = ((pos[0] + next_pos[0]) // 2, (pos[1] + next_pos[1]) // 2)
                    self.grid[mid_pos[0], mid_pos[1]] = Entities.empty
                    carve_path(next_pos)
        
        # Start carving from the start position
        carve_path(start_pos)
        
    def _get_neighbors(self):
        """Get the neighbors of the agent in a 3x3 or 5x5 region."""
        neighbors = []
        view_size = 5 if self.large_view else 3
        offset = view_size // 2
        
        for dr in range(-offset, offset + 1):
            for dc in range(-offset, offset + 1):
                if dr == 0 and dc == 0:  # Skip agent's position
                    continue
                r, c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
                if 0 <= r < self.size and 0 <= c < self.size:
                    neighbors.append(self.grid[r, c])
                else:
                    neighbors.append(Entities.wall)
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
        if not within_bounds or (self.grid[new_pos[0], new_pos[1]] == Entities.wall):
            new_pos = old_pos
            
        # Update the grid
        self.grid[old_pos[0], old_pos[1]] = Entities.visited  # Mark old position as visited
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
        neighbors = np.zeros((len(neighbors_raw) * len(Entities),), dtype=np.int32)
        for i, entity in enumerate(neighbors_raw):
            neighbors[len(Entities) * i + entity] = 1
            
        # Render the grid as an image
        image = self.render_as_image()
        
        # Check if terminal
        terminal = 1 if tuple(self.agent_pos) == tuple(self.goal_pos) else 0
        
        # One-hot encode goal position
        goal_position = np.zeros(3, dtype=np.int32)
        goal_position[self.goal_idx] = 1
        
        return {
            "distance": np.array([distance]),
            "neighbors": neighbors,
            "image": image,
            "is_terminal": np.array([terminal]),
            "goal_position": goal_position,
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
                elif entity == Entities.wall:
                    image[i, j] = self.COLORS['wall']
                elif entity == Entities.visited:
                    image[i, j] = self.COLORS['visited']
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
            Entities.wall: '#',
            Entities.visited: 'V',
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
    
    env = MazeEnv()
    obs, _ = env.reset()
    
    print("\nMaze Environment")
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
        
        if terminated:
            print("\nSuccess! You reached the goal!")
            obs, _ = env.reset()
            print("\nNew episode started:")
            print(env.ascii)

if __name__ == "__main__":
    main()
