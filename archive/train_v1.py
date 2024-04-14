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
from gym_env_rlot.buy_sell.gym_env_v1 import BuySellUndEnv
from gym_env_rlot.buy_sell.utils import backtest_proba, data_artifact_download
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


def save_log_checkpoint(
    algo,
    check_point_path,
    test_metrics_all,
    test_metrics_unseen,
    test_it,
    artifact_file_name,
    env_train,
):
    shutil.rmtree(check_point_path, ignore_errors=True)
    os.makedirs(check_point_path, exist_ok=True)
    algo.save(os.path.join(os.getcwd(), "models", "PPO"))
    # create an artifact for the directiry checkpoint on wandb

    env_info = {
        "feature_names": env_train.feature_names,
        "noise_factor": env_train.noise_factor,
        "observation_noise_std": env_train.observation_noise_std,
    }
    with open(os.path.join(check_point_path, "gym_env_info.json"), "w") as fn:
        json.dump(env_info, fn)

    artifact = wandb.Artifact(
        f"iteration_{test_it}",
        type="model",
        description="PPO model Iteration Checkpoint",
        metadata={
            "test_metrics_all": test_metrics_all,
            "test_metrics_unseen": test_metrics_unseen,
            "data_artifact": artifact_file_name,
        },
    )

    # log artifact
    artifact.add_dir(check_point_path, name="PPO")
    wandb.log_artifact(artifact)


it_var = {
    "previous_drawdown": deque(maxlen=10),
    "previous_pl": deque(maxlen=10),
    "episode_reward_mean": [],
    "episode_reward_max": [],
    "episode_reward_min": [],
    "checkpoint_reset_counter": 0,
    "checkpoint_reset_limit": 7,
    "it_count_before_starting_reset": 15,
}

check_point_path = os.path.join(os.getcwd(), "models", "PPO")
artifact_file_name = "jeroencvlier/rlot/desc_stats_nounderlying:v5"
project_name = "rlot-PPO-pipeline-v1-noise-sliceddown-stats-v-gym-v0"

try:
    artifact_path = data_artifact_download(artifact_file_name)
    ray.init(ignore_reinit_error=True, num_cpus=11, num_gpus=0)
    register_env("buysellmulti_env", lambda config: MultiEnv(config))
    env_train = BuySellUndEnv({"artifact_path": artifact_path, "test_train": "train"})
    env_all = BuySellUndEnv({"artifact_path": artifact_path, "test_train": "all"})
    env_unseen = BuySellUndEnv({"artifact_path": artifact_path, "test_train": "test"})

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

    wandb.init(project=project_name, job_type="PPO-Training", name="Iteration BuyHold")

    test_metrics_all = backtest_proba(
        algo=algo, all_unseen="all", buyhold=True, env_backtest=env_all
    )
    test_metrics_unseen = backtest_proba(
        algo=algo, all_unseen="unseen", buyhold=True, env_backtest=env_unseen
    )
    goal_pl = test_metrics_unseen["pL"] * 1.25
    goal_drawdown = test_metrics_unseen["drawdown"] * 0.50

    it_var["previous_drawdown"].append(test_metrics_unseen["drawdown"])
    it_var["previous_pl"].append(test_metrics_unseen["pL"])

    wandb.run.summary["it_reward_mean"] = 0
    wandb.run.summary["it_reward_max"] = 0
    wandb.run.summary["it_reward_min"] = 0
    wandb.run.summary["mean_drawdown"] = test_metrics_unseen["drawdown"]
    wandb.run.summary["mean_pl"] = test_metrics_unseen["pL"]
    wandb.finish()

    # train the agent
    for test_it in tqdm(range(1, 1000)):
        logging.info(f"Starting Iteration {test_it}...")
        wandb.init(project=project_name, job_type="PPO", name=f"Iteration {test_it}")

        for i in range(10):
            result = algo.train()
            it_var["episode_reward_max"].append(result["episode_reward_max"])
            it_var["episode_reward_min"].append(result["episode_reward_min"])
            it_var["episode_reward_mean"].append(result["episode_reward_mean"])

        # remove np.nan values from the list
        wandb.run.summary["it_reward_mean"] = np.nanmean(it_var["episode_reward_mean"])
        wandb.run.summary["it_reward_max"] = np.nanmean(it_var["episode_reward_max"])
        wandb.run.summary["it_reward_min"] = np.nanmean(it_var["episode_reward_min"])

        # reset the lists
        it_var["episode_reward_mean"] = []
        it_var["episode_reward_max"] = []
        it_var["episode_reward_min"] = []

        test_metrics_all = backtest_proba(
            algo=algo, all_unseen="all", buyhold=False, env_backtest=env_all
        )
        test_metrics_unseen = backtest_proba(
            algo=algo, all_unseen="unseen", buyhold=False, env_backtest=env_unseen
        )
        it_var["previous_drawdown"].append(test_metrics_unseen["drawdown"])
        it_var["previous_pl"].append(test_metrics_unseen["pL"])

        mean_drawdown = np.mean(it_var["previous_drawdown"])
        mean_pl = np.mean(it_var["previous_pl"])
        wandb.run.summary["mean_drawdown"] = mean_drawdown
        wandb.run.summary["mean_pl"] = mean_pl
        logging.info("Iteration metrics improved. Saving model...")

        save_log_checkpoint(
            algo,
            check_point_path,
            test_metrics_all,
            test_metrics_unseen,
            test_it,
            artifact_file_name,
            env_train,
        )

        if test_it > it_var["it_count_before_starting_reset"]:
            logging.info(
                f"Testing if drawdown (goal: {goal_drawdown}, mean: {mean_drawdown}) and pL (goal: {goal_pl}, mean: {mean_pl}) metrics are reached..."
            )
            if (mean_drawdown < goal_drawdown) and (mean_pl > goal_pl):

                # exit the training loop
                logging.info("Drawdown and pL metrics reached. Exiting...")
                break

        wandb.finish()


except Exception as err:
    print(err)
finally:
    ray.shutdown()
    wandb.finish()
