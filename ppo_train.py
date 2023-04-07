import numpy as np
import pprint
import ray
ray.init()

from multiagent_env import GridWorldEnv
from ray import air, tune
from ray.tune.stopper.experiment_plateau import ExperimentPlateauStopper
# from ray.rllib.agents.ppo import PPOTrainer, PPOTF1Policy, PPOTF2Policy, PPOTorchPolicy
from ray.rllib.algorithms.ppo import PPOConfig, PPOTF1Policy

checkpoint_path = "model_checkpoint/PPO/best_model"
logdir = "logdir"

env = GridWorldEnv()

policies = {
    "policy_1": (PPOTF1Policy, env.observation_space, env.action_space, {}),
}

policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "policy_1"

config = PPOConfig()
config = config.environment(env=GridWorldEnv)
config = config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
config = config.framework("tf")
config = config.resources(num_gpus=1)

# config = {
#     "env": GridWorldEnv,  # "my_env" <- if we previously have registered the env with `tune.register_env("[name]", lambda config: [returns env object])`.
#     "env_config": {},
#     "framework": "tf",  # If users have chosen to install torch instead of tf.
#     "create_env_on_driver": True,
#     "multiagent": {
#         "policies": policies,
#         "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "policy_1",
#     },
#     # "exploration_config": {
#     #     "type": "UpperConfidenceBound",
#     # },
#     # "num_workers": 4,
# }

# rllib_trainer = PPOTrainer(config=config)

stoper = ExperimentPlateauStopper(
    metric="episode_reward_mean",
    mode="max",
    patience=10,
)

tune.Tuner(  
    "PPO",
    run_config=air.RunConfig(
        stop=stoper,
        local_dir=logdir,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_score_attribute="episode_reward_mean",
            checkpoint_score_order="max",
            checkpoint_frequency=1,
            num_to_keep=1
        )
    ),
    param_space=config.to_dict(),
).fit()