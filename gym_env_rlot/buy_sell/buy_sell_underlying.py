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
from gym_env_rlot.buy_sell.gym_env import BuySellUndEnv
from gym_env_rlot.buy_sell.utils import backtest_proba, data_artifact_download


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
    algo, check_point_path, test_metrics_all, test_metrics_unseen, test_it
):
    shutil.rmtree(check_point_path, ignore_errors=True)
    os.makedirs(check_point_path, exist_ok=True)
    algo.save(os.path.join(os.getcwd(), "models", "PPO"))
    # create an artifact for the directiry checkpoint on wandb
    artifact = wandb.Artifact(
        f"iteration_{test_it}",
        type="model",
        description="PPO model Iteration Checkpoint",
        metadata={
            "test_metrics_all": test_metrics_all,
            "test_metrics_unseen": test_metrics_unseen,
        },
    )

    # log artifact
    artifact.add_dir(check_point_path, name="PPO")
    wandb.log_artifact(artifact)


it_var = {
    "previous_drawdown": deque(maxlen=15),
    "previous_pl": deque(maxlen=15),
    "episode_reward_mean": [],
    "episode_reward_max": [],
    "episode_reward_min": [],
    "checkpoint_reset_counter": 0,
    "checkpoint_reset_limit": 7,
    "it_count_before_starting_reset": 15,
}

check_point_path = os.path.join(os.getcwd(), "models", "PPO")


try:
    artifact_path = data_artifact_download()
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

    wandb.init(project="rlot-PPO", job_type="PPO-Training", name="Iteration BuyHold")

    test_metrics_all = backtest_proba(
        algo=algo, all_unseen="all", artifact_path=artifact_path, buyhold=True
    )
    test_metrics_unseen = backtest_proba(
        algo=algo, all_unseen="unseen", artifact_path=artifact_path, buyhold=True
    )
    it_var["previous_drawdown"].append(test_metrics_unseen["drawdown"])
    it_var["previous_pl"].append(test_metrics_unseen["pL"])

    wandb.run.summary["it_reward_mean"] = 0
    wandb.run.summary["it_reward_max"] = 0
    wandb.run.summary["it_reward_min"] = 0
    save_log_checkpoint(
        algo,
        check_point_path,
        test_metrics_all,
        test_metrics_unseen,
        test_it=0,
    )
    wandb.run.summary["mean_drawdown"] = test_metrics_unseen["drawdown"]
    wandb.run.summary["mean_pl"] = test_metrics_unseen["pL"]
    wandb.finish()

    # train the agent
    for test_it in tqdm(range(1, 1000)):
        logging.info(f"Starting Iteration {test_it}...")
        wandb.init(project="rlot-PPO-2", job_type="PPO", name=f"Iteration {test_it}")

        for i in range(25):
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
            algo=algo, all_unseen="all", artifact_path=artifact_path, buyhold=False
        )
        test_metrics_unseen = backtest_proba(
            algo=algo, all_unseen="unseen", artifact_path=artifact_path, buyhold=False
        )
        it_var["previous_drawdown"].append(test_metrics_unseen["drawdown"])
        it_var["previous_pl"].append(test_metrics_unseen["pL"])

        mean_drawdown = np.mean(it_var["previous_drawdown"])
        mean_pl = np.mean(it_var["previous_pl"])
        wandb.run.summary["mean_drawdown"] = mean_drawdown
        wandb.run.summary["mean_pl"] = mean_pl
        logging.info("Iteration metrics improved. Saving model...")
        # save the model
        save_log_checkpoint(
            algo,
            check_point_path,
            test_metrics_all,
            test_metrics_unseen,
            test_it,
        )
        if test_it > it_var["it_count_before_starting_reset"]:
            if (test_metrics_unseen["drawdown"] < 7) and (
                test_metrics_unseen["pL"] > 50
            ):

                # exit the training loop
                logging.info("Drawdown and pL metrics reached. Exiting...")
                break

        wandb.finish()


except Exception as err:
    print(err)
finally:
    ray.shutdown()
    wandb.finish()
