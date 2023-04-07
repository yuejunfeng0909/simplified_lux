import time
from multiagent_env import GridWorldEnv

env = GridWorldEnv()

env.reset()
env.record()

simulation_time = 1000
assert simulation_time > 1

for _ in range(simulation_time-1):
    
    random_sample_action = {
        agent: env.action_space.sample() for agent in env.agents
    }
    new_obs, reward, _, _, _ = env.step(random_sample_action)
    env.record()
env.render(framerate=10)

# enter to continue
input()