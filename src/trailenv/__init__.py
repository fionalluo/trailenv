from gymnasium.envs.registration import register as register2
# from gym.envs.registration import register
from trailenv.lava_env import make_lava_env, make_lava_cliff_env
print("Registering TrailEnv")
# register(
#     id="TrailEnv-v0",
#     entry_point="trailenv.trail_env:TrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[2,2],[3,3],[2,4],[3,5],[2,6],[1,7],[2,8],[3,9],[4,10]])
# )

# register2(
#     id="TrailEnv-v0",
#     entry_point="trailenv.trail_env:TrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[2,2],[3,3],[2,4],[3,5],[2,6],[1,7],[2,8],[3,9],[4,10]])
# )

# register2(
#     id="TrailEnv-v1",
#     entry_point="trailenv.trail_env:TrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=20,height=20,start_pos=[1,1],trail=[[2,2],[3,3],[2,4],[3,5],[2,6],[1,7],[2,8],[3,9],[4,10], [5,11], [6,12], [7,13], [6,14], [5,15], [4,16],[5,17],[6, 18], [7, 19]])
# )
# register2(
#     id="GridBlindPick7x7Env-v0",
#     entry_point="trailenv.trail_env:GridBlindPickEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=7,height=7,start_pos=[3,3])
# )

# register2(
#     id="GridBlindPick15x15Env-v0",
#     entry_point="trailenv.trail_env:GridBlindPickEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=15,height=15,start_pos=[7,7])
# )

# register2(
#     id="GridBlindPick31x31Env-v0",
#     entry_point="trailenv.trail_env:GridBlindPickEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=31,height=31,start_pos=[15,15])
# )

# # Centered
# register2(
#     id="GridBlindPick31x31EnvCenter-v0",
#     entry_point="trailenv.trail_env:GridBlindPickEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=31,height=31,start_pos=[15,15],centered=True)
# )
# register2(
#     id="GridBlindPick100x100EnvCenter-v0",
#     entry_point="trailenv.trail_env:GridBlindPickEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=100,height=100,start_pos=[50,50],centered=True)
# )
# for curriculum in [2, 3, 4, 5, 10, 15, 50]:
#     for dim in [7, 15, 31, 100]:
#         for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
#             register2(
#                 id=f"GridBlindPick{dim}x{dim}EnvC{curriculum}Threshold{threshold}-v0",
#                 entry_point="trailenv.trail_env:GridBlindPickEnv",
#                 max_episode_steps=100,
#                 kwargs=dict(width=dim,height=dim,start_pos=[dim//2, dim//2],curriculum=curriculum, threshold=threshold)
#             )
# for curriculum in [2, 3, 4, 5, 10, 15, 50]:
#     for dim in [7, 15, 31, 100]:
#         for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
#             register2(
#                 id=f"GridBlindPick{dim}x{dim}EnvC{curriculum}Threshold{threshold}Center-v0",
#                 entry_point="trailenv.trail_env:GridBlindPickEnv",
#                 max_episode_steps=100,
#                 kwargs=dict(width=dim,height=dim,start_pos=[dim//2, dim//2],curriculum=curriculum, threshold=threshold, centered=True)
#             )

# Register Tiger Door Environment
for size in [5, 6, 7, 8, 12, 16, 20, 32, 64, 128]:
    register2(
        id=f"TigerDoor{size}x{size}-v0",
        entry_point="trailenv.tiger_door_env:TigerDoorEnv",
        max_episode_steps=100,
        kwargs=dict(size=size)
    )

# Register Tiger Door Environment
register2(
    id=f"TigerDoorKey-v0",
    entry_point="trailenv.tiger_door_key_env:TigerDoorKeyEnv",
    max_episode_steps=100,
)  # default tiger door environment (11 rows, 9 cols)
# custom sizes can be defined as needed later.

# Register Maze Envs
for size in [7, 9, 11, 13, 15, 17, 21, 31]:
    # Give the agent a large view (5x5 local crop) if the size is at least 15
    if size < 15:
        # allocate more steps for larger mazes
        register2(
            id=f"Maze{size}x{size}-v0",
            entry_point="trailenv.maze_env:MazeEnv", 
            max_episode_steps=100,
            kwargs=dict(size=size)
        )
    else:
        # allocate less steps for smaller mazes
        register2(
            id=f"Maze{size}x{size}-v0",
            entry_point="trailenv.maze_env:MazeEnv",
            max_episode_steps=200,
            kwargs=dict(size=size)
        )

# Register Search Envs (simple)
for size in [7, 8, 10, 12, 16, 20, 32, 64, 128]:
    register2(
        id=f"Search{size}x{size}-v0",
        entry_point="trailenv.search_env:SearchEnv",
        max_episode_steps=100,
        kwargs=dict(size=size)
    )

# Register Clean Envs
for size in [7, 8, 10, 12, 16, 20, 32, 64, 128]:
    register2(
        id=f"Clean{size}x{size}-v0",
        entry_point="trailenv.clean_env:CleanEnv",
        max_episode_steps=100,
        kwargs=dict(size=size)
    )

# Register Lava Trail Envs
for size in [7, 8, 12, 16, 20, 32, 64, 128]:
    # for seed in range(10):
    #     register2(
    #         id=f"LavaTrail{size}x{size}Seed{seed}-v0",
    #         entry_point="trailenv.lava_trail_env:LavaTrailEnv",
    #         max_episode_steps=100,
    #         kwargs=dict(size=size, trail_seed=seed)
    #     )
    register2(
        id=f"LavaTrail{size}x{size}-v0",
        entry_point="trailenv.lava_trail_env:LavaTrailEnv",
        max_episode_steps=100,
        kwargs=dict(size=size)
    )

# Registering Bandit Envs
register2(
    id="BanditEnv-v0",
    entry_point="trailenv.bandit_env:BanditEnv",
    max_episode_steps=1,
    kwargs=dict()
)

for path_length in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    register2(
        id=f"BanditPathEnv{path_length}-v0",
        entry_point="trailenv.bandit_path_env:BanditPathEnv",
        max_episode_steps=100,
        kwargs=dict(path_length=path_length)
    )
    for reward_scale in [2, 10, 100, 1000]:
        register2(
            id=f"BanditPathEnv{path_length}scale{reward_scale}-v0",
            entry_point="trailenv.bandit_path_env:BanditPathEnv",
            max_episode_steps=100,
            kwargs=dict(path_length=path_length, reward_scale=reward_scale)
        )

# register2(
#     id="ObsDictTrailEnv-v0",
#     entry_point="trailenv.trail_env:ObsDictTrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[2,2],[3,3],[2,4],[3,5],[2,6],[1,7],[2,8],[3,9],[4,10]])
# )

# register2(
#     id="POObsDictTrailEnv-v0",
#     entry_point="trailenv.trail_env:ObsDictTrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(observation_type="PO",width=12,height=12,start_pos=[1,1],trail=[[2,2],[3,3],[2,4],[3,5],[2,6],[1,7],[2,8],[3,9],[4,10]])
# )

# register2(
#     id="ObsDictTrailEnv-v0",
#     entry_point="trailenv.trail_env:ObsDictTrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=7,height=24,start_pos=[1,1],trail=[[14,2],[2,3],[21,4],[3,5]])
# )
# register2(
#     id="POObsDictTrailEnv-v0",
#     entry_point="trailenv.trail_env:ObsDictTrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(observation_type="PO",width=7,height=24,start_pos=[1,1],trail=[[14,2],[2,3],[21,4],[3,5]])
# )

# register(
#     id="TrailEnv-v1",
#     entry_point="trailenv.trail_env:TrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[2,2],[4,2],[6,2],[8,2],[10,4]])
# )

# register(
#     id="TrailEnv-v2",
#     entry_point="trailenv.trail_env:TrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[6,2],[10,8]])
# )

# register(
#     id="NoisyTrailEnv-v2",
#     entry_point="trailenv.trail_env:NoisyTrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=12,height=12,start_pos=[1,1],trail=[[6,2],[10,8]], y_std=2.0)
# )

# register(
#     id="TrailEnv-v3",
#     entry_point="trailenv.trail_env:TrailEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=4,height=4,start_pos=[1,1],trail=[[2,2]])
# )

# register(
#     id="NoisyFootballEnv-v0",
#     entry_point="trailenv.trail_env:NoisyFootballEnv",
#     max_episode_steps=100,
#     kwargs=dict(width=6,height=6,start_pos=[1,1],trail=[[2,2], [3,3],[2,4]], y_std=5.0)
# )