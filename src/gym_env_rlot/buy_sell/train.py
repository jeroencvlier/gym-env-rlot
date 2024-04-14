import os
import warnings
import shutil
from collections import deque
import wandb
import ray
import logging
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from ray.rllib.algorithms import PPOConfig
from ray.tune.registry import register_env
from src.gym_env_rlot.buy_sell.gym_env import BuySellUndEnv
from src.gym_env_rlot.buy_sell.utils import backtest_proba, data_artifact_download
import json


warnings.filterwarnings("ignore", category=DeprecationWarning)


def env_creator(env_config):
    return BuySellUndEnv(env_config)


class MultiEnv(gym.Env):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        self.env = gym.make(
            choose_env_for(env_config.worker_index, env_config.vector_index)
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed, options):
        return self.env.reset(seed, options)

    def step(self, action):
        return self.env.step(action)


check_point_path = os.path.join(os.getcwd(), "models", "PPO")
artifact_file_name = "jeroencvlier/rlot-data-pipeline/desc_stats_nounderlying:latest"
project_name = "PPO-tune-v3"

artifact_path = data_artifact_download(artifact_file_name)
ray.init(ignore_reinit_error=True, num_cpus=11, num_gpus=0)
register_env("buysellmulti_env", lambda config: MultiEnv(config))


algo = (
    PPOConfig()
    .environment(
        BuySellUndEnv,
        env_config={"artifact_path": artifact_path, "test_train": "train"},
    )
    .framework("torch")
    .rollouts(num_rollout_workers=11, num_envs_per_worker=1)
    .build()
)

wandb.init(project=project_name, job_type="PPO-Training", name="Iteration BuyHold Tune")

# train the agent
for test_it in tqdm(range(1, 1000)):
    logging.info(f"Starting Iteration {test_it}...")
    wandb.init(project=project_name, job_type="PPO", name=f"Iteration {test_it}")

    for i in range(10):
        result = algo.train()


ray.shutdown()
wandb.finish()
