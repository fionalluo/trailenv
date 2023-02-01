# Frontier Maze
A maze with sparsely populated rewards.
- Agent is green point mass.
- Rewards are yellow point mass.
- Goal is the red point mass.

The action space is continuous velocity. The observation is the agent's (x,y) coordinate and velocity.

<img src="imgs/frontier_maze.png" width="400px">


# Installation
```bash
# install frontier_maze package
pip install -e .

# run a visualization of the maze
python -m src.frontier_maze.visualize_maze
```

# Usage
You can import it through gym.
```py
from gym import gym
gym.make("frontiermaze2d-v0")
```

Alternatively, you can import the environment class.

```py
from frontier_maze import FrontierMaze
env = FrontierMaze()
```

# Action Space
It is 2 dimensional velocity control. The sign directions follow conventional directions.
- 1nd dim: -1 is go up, +1 is go down
- 2st dim: -1 is go left, +1 is go right

# Observation Space
Position of the point max.
- 1st dim is vertical axis. Negative is upwards, positive is downwards.
- 2nd dim is horizontal axis. Negative is left, positive is right.