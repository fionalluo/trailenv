"""
Maze Environment

A procedurally generated maze environment where an agent must navigate from the bottom-left
corner to one of three possible goal positions (upper-left, upper-right, or lower-right).
The maze is generated using a depth-first search algorithm ensuring single-width paths
and proper connectivity to all corners.
Note that the recursive maze generation algorithm only works with odd side-length mazes.

Observation Space:
    - image: Full maze visualization.
    - neighbors_5x5: 24 × len(entities) one-hot encoded neighbors in a 5x5 region.
    - neighbors_3x3: 8 × len(entities) one-hot encoded neighbors in a 3x3 region.
    - goal_position: One-hot encoded goal position (0: top-left, 1: top-right, 2: bottom-right).
    - distance: Manhattan distance to the goal.
    - position: Current position of the agent (row, column).
    - is_terminal: 1 if agent reached goal, 0 otherwise.

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
    
    def __init__(self, size=12):
        super().__init__()
        # The maze generation algorithm requires odd dimensions to ensure all cells are reachable.
        self.size = size if size % 2 != 0 else size + 1
        self.steps = 0
        
        # Initialize empty grid
        self._reset_grid()
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "distance": spaces.Box(low=0, high=2*self.size, shape=(1,), dtype=np.int32),
            "neighbors_3x3": spaces.Box(low=0, high=1, shape=(3, 3, len(Entities)), dtype=np.int32),
            "neighbors_5x5": spaces.Box(low=0, high=1, shape=(5, 5, len(Entities)), dtype=np.int32),
            "image": spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            "goal_position": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int32),
            "position": spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=np.int32),
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
        
    def _get_neighbors(self, view_size=3):
        """Get the neighbors of the agent in a view_size x view_size region."""
        neighbors = []
        offset = view_size // 2
        
        # Return neighbors in row-major order (top-left to bottom-right)
        for dr in range(-offset, offset + 1):
            for dc in range(-offset, offset + 1):
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
        
        # Get 3x3 neighbors and one-hot encode them
        neighbors_3x3_raw = self._get_neighbors(view_size=3)
        neighbors_3x3 = np.zeros((3, 3, len(Entities)), dtype=np.int32)
        for i, entity in enumerate(neighbors_3x3_raw):
            neighbors_3x3[i // 3, i % 3, entity] = 1

        # Get 5x5 neighbors and one-hot encode them
        neighbors_5x5_raw = self._get_neighbors(view_size=5)
        neighbors_5x5 = np.zeros((5, 5, len(Entities)), dtype=np.int32)
        for i, entity in enumerate(neighbors_5x5_raw):
            neighbors_5x5[i // 5, i % 5, entity] = 1
            
        # Render the grid as an image
        image = self.render_as_image()
        
        # Check if terminal
        terminal = 1 if tuple(self.agent_pos) == tuple(self.goal_pos) else 0
        
        # One-hot encode goal position
        goal_position = np.zeros(3, dtype=np.int32)
        goal_position[self.goal_idx] = 1
        
        return {
            "distance": np.array([distance]),
            "neighbors_3x3": neighbors_3x3,
            "neighbors_5x5": neighbors_5x5,
            "image": image,
            "is_terminal": np.array([terminal]),
            "goal_position": goal_position,
            "position": self.agent_pos.astype(np.int32),
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

    def display_neighbors(self, neighbors_array, title="Neighbors"):
        """Display neighbor array in a readable format."""
        entity_to_char = {
            Entities.empty: ' ',
            Entities.agent: 'A',
            Entities.goal: 'G',
            Entities.wall: '#',
            Entities.visited: 'V',
        }
        
        size = neighbors_array.shape[0]
        print(f"\n{title} ({size}x{size}):")
        print("  " + " ".join([f"{i}" for i in range(size)]))
        for i in range(size):
            row = [f"{i}"]
            for j in range(size):
                # Find which entity is present (1) in this cell
                entity_idx = np.argmax(neighbors_array[i, j])
                row.append(entity_to_char[entity_idx])
            print(" ".join(row))

def colorize(string: str, color: str, bold: bool = False, highlight: bool = False) -> str:
    """Returns string surrounded by appropriate terminal colour codes to print colourised text."""
    color2num = dict(
        gray=30, red=31, green=32, yellow=33, blue=34,
        magenta=35, cyan=36, white=37, crimson=38
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
    from gymnasium.utils.env_checker import check_env
    
    env = MazeEnv(size=12)
    
    print("\n--- Running Environment Checker ---")
    try:
        check_env(env.unwrapped)
        print(colorize("Environment check passed!", "green", bold=True))
    except Exception as e:
        print(colorize(f"Environment check failed:\n{e}", "red", bold=True))
    print("---------------------------------\n")

    obs, _ = env.reset()
    
    print(colorize("Maze Environment", "cyan", bold=True))
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
        print(colorize(f"Reward: {reward}", "yellow"))
        print(colorize(f"Steps: {env.steps}", "blue"))
        
        # Print observations
        print("\nObservations:")
        print(f"Distance to goal: {obs['distance'][0]}")
        print(f"Neighbors (3x3) shape: {obs['neighbors_3x3'].shape}")
        print(f"Neighbors (5x5) shape: {obs['neighbors_5x5'].shape}")
        print(f"Goal position: {obs['goal_position']}")
        print(f"Agent position: {obs['position']}")
        
        # Show a sample of the 3x3 neighbors (center cell)
        print(f"3x3 neighbors center: {obs['neighbors_3x3'][1, 1]}")
        print(f"5x5 neighbors center: {obs['neighbors_5x5'][2, 2]}")
        
        # Display neighbor crops in readable format
        env.display_neighbors(obs['neighbors_3x3'], "3x3 Neighbors")
        env.display_neighbors(obs['neighbors_5x5'], "5x5 Neighbors")
        
        if terminated:
            print(colorize("\nSuccess! You reached the goal!", "green", bold=True))
            obs, _ = env.reset()
            print("\nNew episode started:")
            print(env.ascii)

if __name__ == "__main__":
    main()
