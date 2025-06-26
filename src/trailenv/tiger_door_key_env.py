"""
Tiger-Door-Key Environment

A grid-based environment where an agent must find a key to identify two unlocked doors
out of five, then press a button to determine the relative location of a treasure and a
tiger behind those doors, and finally navigate to the treasure while avoiding the tiger.

Observation Space:
    The environment provides both privileged (teacher) and unprivileged (student) observations:

    Conditional Observations:
    - neighbors/neighbors_unprivileged: One-hot encoded neighbors, distinguishing between treasure, tiger, and locked doors.
    - door/door_unprivileged: Initially 0, reveals relative treasure/tiger position after button press.
    - doors_unlocked/doors_unlocked_unprivileged: Initially all 0s, reveals unlocked doors after key collection.
    - distance_to_treasure: Manhattan distance to the treasure door.
    
    Universal Observations:
    - is_terminal: 1 if agent reached treasure or tiger, 0 otherwise.
    - position: Current agent position.
    - has_key: 1 if the agent has collected the key, 0 otherwise. Not necessary since redundant with doors_unlocked_unprivileged.
    
    Visual Observations:
    - image: Full grid visualization. For logging purposes.

Action Space:
    Discrete(4): up, right, down, left

Rewards:
    +10: Reaching the treasure door.
    -10: Reaching the tiger door.
    -1: Trying to open a locked door.
    0: All other actions.

Grid Elements:
    - Agent (A): Cyan
    - Button (B): Pink
    - Key (K): Yellow
    - Door (D): Brown
    - Wall (#): Gray
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

KEY_ACTION_MAP = {
    "w": Actions.up,
    "a": Actions.left,
    "s": Actions.down,
    "d": Actions.right,
}

class Entities(IntEnum):
    empty = 0
    agent = 1
    wall = 2
    button = 3
    key = 4
    door = 5

# For privileged observations
class PrivilegedEntities(IntEnum):
    empty = 0
    agent = 1
    wall = 2
    button = 3
    key = 4
    treasure_door = 5
    tiger_door = 6
    locked_door = 7
    unknown_door = 8

# Define colors for visualization
COLORS = {
    'agent': (0, 255, 255),       # cyan
    'button': (255, 192, 203),    # pink
    'key': (255, 255, 0),         # yellow
    'door': (165, 42, 42),        # brown
    'treasure': (0, 255, 0),      # green (for rendering image)
    'tiger': (255, 0, 0),         # red (for rendering image)
    'wall': (128, 128, 128),      # gray
    'empty': (255, 255, 255),     # white
}


class TigerDoorKeyEnv(gym.Env):
    def __init__(self, row_size=11, col_size=9):
        super().__init__()
        self.row_size = row_size
        self.col_size = col_size
        self.steps = 0
        
        # Initialize empty grid
        self._reset_grid()
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "distance_to_treasure": spaces.Box(low=0, high=row_size + col_size, shape=(1,), dtype=np.int32),
            "neighbors": spaces.Box(low=0, high=1, shape=(4 * len(PrivilegedEntities),), dtype=np.int32),
            "neighbors_unprivileged": spaces.Box(low=0, high=1, shape=(4 * len(PrivilegedEntities),), dtype=np.int32),
            "door": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32),
            "door_unprivileged": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int32),
            "doors_unlocked": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int32),  # [known, door0, door1, door2, door3, door4]
            "doors_unlocked_unprivileged": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int32),
            "has_key": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            "image": spaces.Box(low=0, high=255, shape=(row_size, col_size, 3), dtype=np.uint8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
            "position": spaces.Box(low=0, high=max(row_size, col_size), shape=(2,), dtype=np.int32),
        })
        
        self.action_space = spaces.Discrete(len(Actions))

    def _reset_grid(self):
        """Create the grid to match the screenshot layout: 7 rows x 6 cols, walls on the border, with random treasure/tiger door positions."""
        self.row_size = 7
        self.col_size = 6
        self.grid = np.full((self.row_size, self.col_size), Entities.wall)

        # Place empty cells in the inner area
        for r in range(1, self.row_size-1):
            for c in range(1, self.col_size-1):
                self.grid[r, c] = Entities.empty

        # Place key (top-left inner)
        self.key_pos = (1, 1)
        self.grid[self.key_pos] = Entities.key

        # Place agent (middle-left inner)
        self.agent_pos = np.array([3, 1])
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entities.agent

        # Place button (bottom-left inner)
        self.button_pos = (5, 1)
        self.grid[self.button_pos] = Entities.button

        # Place doors in the rightmost inner column (col 4)
        self.door_positions = [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4)]
        for pos in self.door_positions:
            self.grid[pos] = Entities.door

        # Randomly assign treasure and tiger positions among the 5 doors
        door_indices = list(range(5))
        treasure_door_idx, tiger_door_idx = random.sample(door_indices, 2)
        self.treasure_door_idx = treasure_door_idx
        self.tiger_door_idx = tiger_door_idx
        self.treasure_pos = self.door_positions[treasure_door_idx]
        self.tiger_pos = self.door_positions[tiger_door_idx]
        self.locked_door_indices = [i for i in door_indices if i not in [treasure_door_idx, tiger_door_idx]]

        # Initialize state
        self.key_collected = False
        self.button_pressed = False
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self._reset_grid()
        return self.gen_obs(), {}
        
    def step(self, action):
        """Take a step in the environment."""
        old_pos = self.agent_pos.copy()
        
        # Handle movement
        new_pos = self.agent_pos + ACTION_COORDS[action]
        if not (0 <= new_pos[0] < self.row_size and 0 <= new_pos[1] < self.col_size and self.grid[new_pos[0], new_pos[1]] != Entities.wall):
            new_pos = self.agent_pos

        self.agent_pos = new_pos
        self.grid[old_pos[0], old_pos[1]] = Entities.empty
        self.grid[self.agent_pos[0], self.agent_pos[1]] = Entities.agent
        
        # Handle interactions
        reward = 0
        terminated = False
        current_pos_tuple = tuple(self.agent_pos)

        if current_pos_tuple == self.key_pos:
            self.key_collected = True
        elif current_pos_tuple == self.button_pos:
            self.button_pressed = True
        elif current_pos_tuple in self.door_positions:
            if current_pos_tuple == self.treasure_pos:
                reward = 10
                terminated = True
            elif current_pos_tuple == self.tiger_pos:
                reward = -10
                terminated = True
            else: # Locked door
                reward = -1
        
        self.steps += 1
        obs = self.gen_obs()
        
        return obs, reward, terminated, False, {}

    def _get_neighbors(self):
        """Get the entities at the four adjacent cells."""
        neighbors = []
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            pos = self.agent_pos + np.array([dr, dc])
            if 0 <= pos[0] < self.row_size and 0 <= pos[1] < self.col_size:
                neighbors.append((self.grid[pos[0], pos[1]], tuple(pos)))
            else:
                neighbors.append((Entities.wall, None))
        return neighbors

    def gen_obs(self):
        """Generate the observation dictionary."""
        # Privileged observations
        distance_to_treasure = np.linalg.norm(self.agent_pos - self.treasure_pos, ord=1)
        
        # Unprivileged observations
        neighbors_raw = self._get_neighbors()
        
        neighbors_unprivileged = np.zeros((4 * len(PrivilegedEntities),), dtype=np.int32)
        for i, (entity, pos) in enumerate(neighbors_raw):
            if entity == Entities.door:
                # For unprivileged, all doors look the same (unknown_door)
                neighbors_unprivileged[i * len(PrivilegedEntities) + PrivilegedEntities.unknown_door] = 1
            else:
                # Map regular entities directly
                neighbors_unprivileged[i * len(PrivilegedEntities) + entity] = 1

        neighbors_privileged = np.zeros((4 * len(PrivilegedEntities),), dtype=np.int32)
        for i, (entity, pos) in enumerate(neighbors_raw):
            p_entity = entity
            if entity == Entities.door:
                if pos == self.treasure_pos:
                    p_entity = PrivilegedEntities.treasure_door
                elif pos == self.tiger_pos:
                    p_entity = PrivilegedEntities.tiger_door
                else:
                    p_entity = PrivilegedEntities.locked_door
            neighbors_privileged[i * len(PrivilegedEntities) + p_entity] = 1

        door_relative_pos = 1 if self.treasure_pos[0] < self.tiger_pos[0] else -1
        door_unprivileged = door_relative_pos if self.button_pressed else 0

        # Privileged: always known, doors_unlocked[0] = 1, doors_unlocked[1:6] = door states
        doors_unlocked = np.zeros(6, dtype=np.int32)
        doors_unlocked[0] = 1  # known = True
        doors_unlocked[1 + self.treasure_door_idx] = 1  # treasure door
        doors_unlocked[1 + self.tiger_door_idx] = 1     # tiger door
        
        # Unprivileged: known only if key collected
        doors_unlocked_unprivileged = np.zeros(6, dtype=np.int32)
        if self.key_collected:
            doors_unlocked_unprivileged[0] = 1  # known = True
            doors_unlocked_unprivileged[1 + self.treasure_door_idx] = 1  # treasure door
            doors_unlocked_unprivileged[1 + self.tiger_door_idx] = 1     # tiger door
        # else: known = False (default), all door states = 0

        has_key = 1 if self.key_collected else 0
        is_terminal = 1 if tuple(self.agent_pos) in [self.treasure_pos, self.tiger_pos] else 0

        return {
            "distance_to_treasure": np.array([distance_to_treasure], dtype=np.int32),
            "neighbors": neighbors_privileged,
            "neighbors_unprivileged": neighbors_unprivileged,
            "door": np.array([door_relative_pos], dtype=np.int32),
            "door_unprivileged": np.array([door_unprivileged], dtype=np.int32),
            "doors_unlocked": doors_unlocked,
            "doors_unlocked_unprivileged": doors_unlocked_unprivileged,
            "has_key": np.array([has_key], dtype=np.int32),
            "image": self.render_as_image(),
            "is_terminal": np.array([is_terminal], dtype=np.int32),
            "position": self.agent_pos,
        }

    def render_as_image(self, show_hidden=False):
        """Render the environment to an image."""
        image = np.zeros((self.row_size, self.col_size, 3), dtype=np.uint8)
        for r in range(self.row_size):
            for c in range(self.col_size):
                entity = self.grid[r, c]
                if entity == Entities.wall:
                    image[r, c] = COLORS['wall']
                elif entity == Entities.empty:
                    image[r, c] = COLORS['empty']
                elif entity == Entities.button:
                    image[r, c] = COLORS['button']
                elif entity == Entities.key:
                    image[r, c] = COLORS['key']
                elif entity == Entities.door:
                    # Always show treasure/tiger colors for visualization
                    if (r, c) == self.treasure_pos:
                        image[r, c] = COLORS['treasure']  # Green for treasure
                    elif (r, c) == self.tiger_pos:
                        image[r, c] = COLORS['tiger']     # Red for tiger
                    else:
                        image[r, c] = COLORS['door']      # Brown for locked doors

        image[self.agent_pos[0], self.agent_pos[1]] = COLORS['agent']
        return image

    def test_colors(self):
        """Test method to verify color rendering."""
        print("Testing color rendering...")
        print("Color definitions (BGR format):")
        for entity, color in COLORS.items():
            print(f"  {entity}: {color}")
        
        print("\nColor behavior:")
        print("  - Treasure door: Green (always visible)")
        print("  - Tiger door: Red (always visible)") 
        print("  - Locked doors: Brown")
        print("  - Agent: Cyan")
        print("  - Key: Yellow")
        print("  - Button: Pink")
        print("  - Walls: Gray")
        print("  - Empty: White")
        print("\nNote: Colors are always shown for visualization/debugging purposes.")
        print("The agent's observations still reflect their knowledge state.")

    @property
    def ascii(self):
        """Return ASCII representation of the current grid state."""
        grid_str = []
        for r in range(self.row_size):
            row_str = ""
            for c in range(self.col_size):
                pos = (r, c)
                entity = self.grid[r, c]
                if pos == self.treasure_pos:
                    row_str += 'T'
                elif pos == self.tiger_pos:
                    row_str += 'X'
                elif entity == Entities.door:
                    row_str += 'D'
                elif entity == Entities.empty:
                    row_str += ' '
                elif entity == Entities.agent:
                    row_str += 'A'
                elif entity == Entities.wall:
                    row_str += '#'
                elif entity == Entities.button:
                    row_str += 'B'
                elif entity == Entities.key:
                    row_str += 'K'
                else:
                    row_str += '?'
            grid_str.append(row_str)
        return "\n".join(grid_str)
        
def main():
    """Main function to test the environment with keyboard controls."""
    import sys
    
    env = TigerDoorKeyEnv()
    
    # Test color rendering
    print(colorize("\n--- Color Test ---", "cyan", bold=True))
    env.test_colors()
    print("--- End Color Test ---\n")
    
    obs, _ = env.reset()
    
    print(colorize("\nTiger-Door-Key Environment", "cyan", bold=True))
    print(colorize("Use WASD keys to move the agent.", "white"))
    print(colorize("Press 'q' to quit.", "white"))
    print(colorize("Press 'r' to reset the environment.", "white"))
    print("\nInitial state:")
    print(env.ascii)
    print(f"Treasure door position: {env.treasure_pos}")
    print(f"Tiger door position: {env.tiger_pos}")
    print(f"Locked door positions: {[env.door_positions[i] for i in env.locked_door_indices]}")
    
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
        elif key in KEY_ACTION_MAP:
            action = KEY_ACTION_MAP[key]
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
        print(f"Position: {obs['position']}")
        print(f"Has key: {obs['has_key'][0]}")
        print(f"Button pressed: {env.button_pressed}")
        print(f"Door privileged: {obs['door'][0]}")
        print(f"Door unprivileged: {obs['door_unprivileged'][0]}")
        print(f"Doors unlocked privileged: [known={obs['doors_unlocked'][0]}, doors={obs['doors_unlocked'][1:]}]")
        print(f"Doors unlocked unprivileged: [known={obs['doors_unlocked_unprivileged'][0]}, doors={obs['doors_unlocked_unprivileged'][1:]}]")
        
        # Show door color information
        print(colorize("Door colors (always visible for visualization):", "green", bold=True))
        print(f"  Treasure door (green): {env.treasure_pos}")
        print(f"  Tiger door (red): {env.tiger_pos}")
        print(f"  Locked doors (brown): {[env.door_positions[i] for i in env.locked_door_indices]}")
        print(colorize("Note: Agent observations still reflect knowledge state (key/button).", "yellow"))
        
        if terminated:
            if reward > 0:
                print(colorize("\nSuccess! You found the treasure!", "green", bold=True))
            else:
                print(colorize("\nGame Over! You found the tiger!", "red", bold=True))
            obs, _ = env.reset()
            print("\nNew episode started:")
            print(env.ascii)

if __name__ == "__main__":
    # Add colorize function for main to run
    def colorize(string: str, color: str, bold: bool = False, highlight: bool = False) -> str:
        color2num = dict(gray=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37, crimson=38)
        attr = []
        num = color2num[color]
        if highlight: num += 10
        attr.append(str(num))
        if bold: attr.append("1")
        return f"\x1b[{';'.join(attr)}m{string}\x1b[0m"
    main()
