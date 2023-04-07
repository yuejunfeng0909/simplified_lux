import numpy as np
import pprint
import ray
ray.init()
checkpoint_path = "model_checkpoint/Impala/best_model"
logdir = "logdir"

from multiagent_env import GridWorldEnv
# import tune
from ray import air, tune
from ray.tune.stopper.experiment_plateau import ExperimentPlateauStopper
from ray.rllib.algorithms.impala import ImpalaConfig, ImpalaTF1Policy, ImpalaTF2Policy, ImpalaTorchPolicy
config = ImpalaConfig()
config = config.training(lr=0.0003, train_batch_size=512)  
# config = config.resources(num_gpus=1)
config = config.framework("torch")  
print(config.to_dict())  

config = config.environment(env=GridWorldEnv)
config = config.rollouts(num_rollout_workers=12)
# config["exploration_config"] = {
#     "type": "UCB",
#     "ucb_coeff": 2.0
# }

env = GridWorldEnv()
policies = {
    "policy_1": (ImpalaTorchPolicy, env.observation_space, env.action_space, {}),
}
policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "policy_1"

# rllib_trainer = config.build()
# best_reward = 0
# for _ in range(1000):
#     results = rllib_trainer.train()
#     print(f"Iteration={rllib_trainer.iteration}: R(\"return\")={results['episode_reward_mean']}")
#     if results['episode_reward_mean'] > best_reward:
#         best_reward = results['episode_reward_mean']
#         rllib_trainer.save(checkpoint_path)
#     if results['episode_reward_mean'] > 1000:
#         break

stoper = ExperimentPlateauStopper(
    metric="episode_reward_mean",
    mode="max",
    patience=10,
)

tune.Tuner(  
    "IMPALA",
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
