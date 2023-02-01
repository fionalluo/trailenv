# from gymnasium.envs.registration import register
from gym.envs.registration import register
print("registering trail env")
register(
    id="TrailEnv-v0",
    entry_point="trailenv.trail_env:TrailEnv",
    max_episode_steps=100,
    kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[2,2],[3,3],[4,2],[5,3],[6,2],[7,1],[8,2],[9,3],[10,4]])
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
    kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[6,2],[10,4]])
)