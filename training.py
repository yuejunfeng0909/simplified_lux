import numpy as np
import pprint
import ray
ray.init()

from multiagent_env import GridWorldEnv
# Import a Trainable (one of RLlib's built-in algorithms):
# We use the PPO algorithm here b/c its very flexible wrt its supported
# action spaces and model types and b/c it learns well almost any problem.
from ray.rllib.agents.ppo import PPOTrainer, PPOTF1Policy, PPOTF2Policy, PPOTorchPolicy

env = GridWorldEnv()

policies = {
    "policy_1": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
}

policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "policy_1"

config = {
    "env": GridWorldEnv,  # "my_env" <- if we previously have registered the env with `tune.register_env("[name]", lambda config: [returns env object])`.
    "env_config": {},
    "framework": "torch",  # If users have chosen to install torch instead of tf.
    "create_env_on_driver": True,
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "policy_1",
    },
}
pprint.pprint(config)
# print()
# for agent in env.agents:
#     print(f"{agent} is now mapped to {policy_mapping_fn(agent)}")

rllib_trainer = PPOTrainer(config=config)

# 4) Run `train()` n times. Repeatedly call `train()` now to see rewards increase.
# Move on once you see (agent1 + agent2) episode rewards of 10.0 or more.
for _ in range(10):
    results = rllib_trainer.train()
    print(f"Iteration={rllib_trainer.iteration}: R(\"return\")={results['episode_reward_mean']}")