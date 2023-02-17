# from gymnasium.envs.registration import register
from gym.envs.registration import register
from trailenv.lava_env import make_lava_env
print("registering trail env")
register(
    id="TrailEnv-v0",
    entry_point="trailenv.trail_env:TrailEnv",
    max_episode_steps=100,
    kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[2,2],[3,3],[2,4],[3,5],[2,6],[1,7],[2,8],[3,9],[4,10]])
)

register(
    id="TrailEnv-v1",
    entry_point="trailenv.trail_env:TrailEnv",
    max_episode_steps=100,
    kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[2,2],[4,2],[6,2],[8,2],[10,4]])
)

register(
    id="TrailEnv-v2",
    entry_point="trailenv.trail_env:TrailEnv",
    max_episode_steps=100,
    kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[6,2],[10,8]])
)

register(
    id="NoisyTrailEnv-v2",
    entry_point="trailenv.trail_env:NoisyTrailEnv",
    max_episode_steps=100,
    kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[6,2],[10,8]], y_std=2.0)
)

register(
    id="TrailEnv-v3",
    entry_point="trailenv.trail_env:TrailEnv",
    max_episode_steps=100,
    kwargs=dict(width=4,height=4,start_pos=[1,1],trail=[[2,2]])
)
