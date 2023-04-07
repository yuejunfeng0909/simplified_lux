import time
import pprint
from multiagent_env import GridWorldEnv
from ray.rllib.algorithms.algorithm import Algorithm

path_to_checkpoint = "logdir/IMPALA/IMPALA_GridWorldEnv_edcf4_00000_0_2023-04-07_22-31-48/checkpoint_000140"

agent = Algorithm.from_checkpoint(path_to_checkpoint)

env = GridWorldEnv()

# agent = Algorithm.from_checkpoint(
#     checkpoint=path_to_checkpoint,
#     policy_ids={"policy_1"},
#     policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "policy_1",
# )

obs = env.reset()
env.record()

simulation_time = 1000
assert simulation_time > 1

for _ in range(simulation_time-1):
    
    # random_sample_action = {
    #     'team_0': env.action_space.sample(),
    #     'team_1': env.action_space.sample(),
    # }
    
    action = agent.compute_actions(obs)
    
    obs, reward, _, _, _ = env.step(action)
    env.record()
env.render(framerate=2)

# enter to continue
input()