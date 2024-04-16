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

logging.basicConfig(level=logging.INFO)
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


def test_env(algo, config):
    env = BuySellUndEnv(config)
    obs = env.reset()
    done = False
    while not done:
        action = algo.compute_single_action(obs)
        help(algo)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break

    return info["total_pl"]


check_point_path = os.path.join(os.getcwd(), "models", "PPO")
artifact_file_name = "jeroencvlier/rlot-data-pipeline/desc_stats_nounderlying:latest"
project_name = "PPO-tune-v0"

artifact_path = data_artifact_download(artifact_file_name)


try:
    ray.init(ignore_reinit_error=True, num_cpus=11, num_gpus=0)
    register_env("buysellmulti_env", lambda config: MultiEnv(config))

    algo = (
        PPOConfig()
        .environment(
            BuySellUndEnv,
            env_config={
                "artifact_path": artifact_path,
                "test_train": "train",
                "ticker": "SPY",
            },
        )
        .framework("torch")
        .rollouts(num_rollout_workers=10, num_envs_per_worker=1)
        .build()
    )
    wandb.init(project=project_name, job_type="PPO-Training")

    # train the agent
    for test_it in tqdm(range(1, 10000)):

        result = algo.train()
        test1_pl = test_env(
            algo,
            {"artifact_path": artifact_path, "test_train": "test1", "ticker": "SPY"},
        )
        test2_pl = test_env(
            algo,
            {"artifact_path": artifact_path, "test_train": "test2", "ticker": "SPY"},
        )
        wandb.log(
            {
                "reward": result["episode_reward_mean"],
                "test1_pl": test1_pl,
                "test2_pl": test2_pl,
            }
        )
except Exception as err:
    print(err)

finally:
    ray.shutdown()
    wandb.finish()
