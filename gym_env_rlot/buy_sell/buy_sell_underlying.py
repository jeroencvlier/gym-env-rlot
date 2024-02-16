import wandb
import pandas as pd
import ray
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from ray.rllib.algorithms import PPOConfig

import matplotlib.pyplot as plt
import seaborn as sns
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.tune.registry import register_env

from gym_env_rlot.buy_sell.gym_env import BuySellUndEnv


# env = BuySellUndEnv({"data_path": "artifacts/delta_bin:v1", "test_train": "test"})
# env.reset()
# for _ in range(10000):
#     action = env.action_space.sample()
#     observation, reward, done, truncated, info = env.step(action)
#     # wandb.log({"position": observation[0], "reward": reward})
#     env.render()
#     if done:
#         # env.reset()
#         break


# e = BuySellUndEnv({"data_path": "artifacts/delta_bin:v1", "test_train": "test"})
# a,b =e.reset()




def env_creator(env_config):
    return BuySellUndEnv(env_config)


def back_test(
    algo,
    pfx: str,
    test_it: int,
    artifact_path: str,
    it_backtests: list,
) -> list:
    if pfx == "unseen":
        tt = "test"
    if pfx == "all":
        tt = "all"

    env_backtest = BuySellUndEnv({"data_path": artifact_path, "test_train": tt})
    observation, _ = env_backtest.reset()
    while not env_backtest.done:
        action = algo.compute_single_action(observation, explore=False)
        observation, reward, done, truncated, info = env_backtest.step(action)

    wandb.log({f"{k}_{pfx}": v for k, v in env_backtest.custom_logs().items()})

    back_test_results = pd.DataFrame(env_backtest.running_pl_list).rename(
        columns={0: f"PL_{test_it}"}
    )
    it_backtests.append(back_test_results)
    if len(it_backtests) > 10:
        it_backtest = pd.concat([it_backtests[0]] + it_backtests[-9:], axis=1)
    else:
        it_backtest = pd.concat(it_backtests, axis=1)

    df_melted = it_backtest.reset_index().melt(
        id_vars="index", var_name="Trade", value_name="Value"
    )

    plt.figure(figsize=(13, 6))
    sns.lineplot(data=df_melted, x="index", y="Value", hue="Trade", dashes=False)
    plt.axhline(y=0, color="black", linestyle="-")
    plt.title(f"Backtest Results {test_it}")
    plt.xlabel("Steps")
    plt.ylabel("Points")
    wandb.log({f"Backtest Results {pfx.capitalize()}": plt})

    return it_backtests


def baseline_backtest(artifact_path: str, test_train: str) -> list:
    env_test = BuySellUndEnv({"data_path": artifact_path, "test_train": test_train})
    env_test.reset()
    while not env_test.done:
        _ = env_test.step(1)
    it_backtests = [
        pd.DataFrame(env_test.running_pl_list).rename(columns={0: f"PL_BuyHold"})
    ]
    return it_backtests


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


try:
    
    ray.init(ignore_reinit_error=True, num_cpus=11, num_gpus=0)
    wandb.init(project="rlot", job_type="PPO")
    artifact_path = wandb.use_artifact("delta_bin:v1").download()
    register_env("buysellmulti_env", lambda config: MultiEnv(config))
    algo = (
        # DQNConfig() # works ok, unstable
        PPOConfig() # best
        # SACConfig() # didn't really learn
        # APPOConfig() # worst
        # ImpalaConfig # couldn't run, keeps trying to latch onto a GPU
        # BCConfig() # didn't really learn
        # MARWILConfig() # couldn't run
        .environment(
            BuySellUndEnv,
            env_config={"data_path": artifact_path, "test_train": "train"},
        )
        .rollouts(num_rollout_workers=11, num_envs_per_worker=20)
        .build()
    )
    # train
    unseen_backtests = baseline_backtest(artifact_path, "test")
    all_backtests = baseline_backtest(artifact_path, "all")
    test_it = 0

    # generate baseline for buy and hold strategy

    for i in tqdm(range(10000)):
        result = algo.train()
        wandb.log(
            {
                "episode_reward_mean": result["episode_reward_mean"],
                "episode_reward_max": result["episode_reward_max"],
                "episode_reward_min": result["episode_reward_min"],
            }
        )

        if i % 100 == 0:
            test_it += 1
            all_backtests = back_test(
                algo, "all", test_it, artifact_path, all_backtests
            )
            unseen_backtests = back_test(
                algo, "unseen", test_it, artifact_path, unseen_backtests
            )
except Exception as err:
    print(err)
finally:
    ray.shutdown()
    wandb.finish()
