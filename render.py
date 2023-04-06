import time
from multiagent_env import GridWorldEnv

env = GridWorldEnv()

env.reset()
env.record()

simulation_time = 100
assert simulation_time > 1

for _ in range(simulation_time-1):
    
    random_sample_action = {
        'team_0': env.action_space.sample(),
        'team_1': env.action_space.sample(),
    }
    new_obs, reward, _, _, _ = env.step(random_sample_action)
    env.record()
env.render(framerate=2)

# enter to continue
input()