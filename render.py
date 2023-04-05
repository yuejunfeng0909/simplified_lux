import time
from multiagent_env import GridWorldEnv

env = GridWorldEnv()

env.reset()
env.record()
for _ in range(10):
    
    example_action = {
        'player_1': {
            0: 0,
            1: 1,
            2: 6,
        },
        'player_2': {
            0: 3,
            1: 5,
            2: 1,
        }
    }
    
    random_sample_action = {
        'player_1': {
            0: env.action_space.sample(),
            1: env.action_space.sample(),
            2: env.action_space.sample(),
        },
        'player_2': {
            0: env.action_space.sample(),
            1: env.action_space.sample(),
            2: env.action_space.sample(),
        }
    }
    obs, rewards, dones, infos = env.step(random_sample_action)
    env.record()
env.render(framerate=2)

# enter to continue
input()