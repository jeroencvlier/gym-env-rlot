import os
import warnings
from collections import deque
import wandb
import ray
import logging
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
from ray.rllib.algorithms import PPOConfig
from ray.tune.registry import register_env
from src.gym_env_rlot.buy_sell.gym_env import BuySellUndEnv
from src.gym_env_rlot.buy_sell.utils import backtest_proba

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="Forking a process while a parallel region is active is potentially unsafe",
)


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
    obs, _ = env.reset()
    done = False
    while not done:
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break

    return info


def run_env(config, algo=None):
    env = BuySellUndEnv(config)
    obs, _ = env.reset()
    done = False
    while not done:
        if algo is None:
            action = 1
        else:
            action = algo.compute_single_action(obs, explore=False)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
    return env.running_pl_list


def plot_test(algo, config):
    env = BuySellUndEnv(config)
    obs, _ = env.reset()
    done = False
    while not done:
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break

    alog_pl = run_env(config, algo)
    bh_pl = run_env(config)

    # plot the P&L
    plt.plot(alog_pl, label="algo")
    plt.plot(bh_pl, label="benchmark")
    # add titale and labels
    plt.title(config["test_train"])
    plt.legend()

    return plt.show()


check_point_path = os.path.join(os.getcwd(), "models", "PPO")
artifact_file_name = "jeroencvlier/rlot-data-pipeline/desc_stats_nounderlying:latest"
project_name = "PPO-tune-v1"


try:
    # Initialize ray and wandb
    ray.init(ignore_reinit_error=True, num_cpus=11, num_gpus=0)
    wandb.init(project=project_name, job_type="PPO-Training")
    artifact_path = wandb.use_artifact(artifact_file_name).download()

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
        .rollouts(num_rollout_workers=11, num_envs_per_worker=5)
        .build()
    )

    # train the agent
    for test_it in tqdm(range(10000)):
        result = algo.train()
        test1_info = test_env(
            algo,
            {"artifact_path": artifact_path, "test_train": "test1", "ticker": "SPY"},
        )
        test2_info = test_env(
            algo,
            {"artifact_path": artifact_path, "test_train": "test2", "ticker": "SPY"},
        )
        # add suffix "test1" and "test2" to dict keys
        test1_info = {f"test1_{k}": v for k, v in test1_info.items()}
        test2_info = {f"test2_{k}": v for k, v in test2_info.items()}

        wandb.log(
            {
                "reward": result["episode_reward_mean"],
            }
        )
        wandb.log(test1_info)
        wandb.log(test2_info)

        if test_it % 25 == 0:
            plot_test(
                algo,
                {
                    "artifact_path": artifact_path,
                    "test_train": "test1",
                    "ticker": "SPY",
                },
            )
            plot_test(
                algo,
                {
                    "artifact_path": artifact_path,
                    "test_train": "test2",
                    "ticker": "SPY",
                },
            )
except Exception as err:
    print(err)

finally:
    ray.shutdown()
    wandb.finish()
