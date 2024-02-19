import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from ray.rllib.algorithms import PPOConfig
from gym_env_rlot.buy_sell.gym_env import BuySellUndEnv
from gym_env_rlot.buy_sell.gym_env import calculate_maximum_drawdown
import logging


def backtest_buyhold(artifact_path: str, test_train: str) -> list:
    env_test = BuySellUndEnv({"artifact_path": artifact_path, "test_train": test_train})
    env_test.reset()
    while not env_test.done:
        _ = env_test.step(1)
    it_backtests = [
        pd.DataFrame(env_test.running_pl_list).rename(columns={0: f"PL_BuyHold"})
    ]
    return it_backtests


def plot_wandb(it_backtest: list, pfx: str, test_it: int = 0) -> None:
    if len(it_backtest) > 10:
        it_backtest = pd.concat([it_backtest[0]] + it_backtest[-9:], axis=1)
    else:
        it_backtest = pd.concat(it_backtest, axis=1)

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


def backtest_old(
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

    env_backtest = BuySellUndEnv({"artifact_path": artifact_path, "test_train": tt})
    observation, _ = env_backtest.reset()
    while not env_backtest.done:
        action = algo.compute_single_action(observation, explore=False)
        observation, reward, done, truncated, info = env_backtest.step(action)

    wandb.log({f"{k}_{pfx}": v for k, v in env_backtest.custom_logs().items()})

    back_test_results = pd.DataFrame(env_backtest.running_pl_list).rename(
        columns={0: f"PL_{test_it}"}
    )
    it_backtests.append(back_test_results)
    plot_wandb(it_backtests, pfx, test_it)

    return it_backtests


# Apply softmax function
def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def backtest_proba(
    algo: PPOConfig,
    all_unseen: str,
    artifact_path: str,
    probas: list = [0.5, 0.7, 0.9],
    buyhold: bool = False,
) -> list:
    logging.info(f"Backtesting {all_unseen} data")
    if all_unseen == "unseen":
        tt = "test"
    if all_unseen == "all":
        tt = "all"

    env_backtest = BuySellUndEnv({"artifact_path": artifact_path, "test_train": tt})
    plot_probas_dict = {}
    test_metrics = {}
    for proba in probas:
        observation, _ = env_backtest.reset()
        previous_action = 0
        while not env_backtest.done:
            if buyhold:
                action = 1
            else:
                if previous_action == 1:
                    action = algo.compute_single_action(observation, explore=False)
                else:
                    # TODO: Exit trade at various probabilities
                    # TODO: Experimetn with different probabilities on exit
                    action_probas = softmax(
                        algo.compute_single_action(
                            observation, explore=False, full_fetch=True
                        )[2]["action_dist_inputs"]
                    )
                    action = 1 if action_probas[1] > proba else 0

            observation, reward, done, truncated, info = env_backtest.step(action)
            previous_action = action

        running_pl_list = env_backtest.running_pl_list
        # log merrics
        proba_str = str(int(proba * 100))

        wandb.run.summary[f"drawdown_{all_unseen}_{proba_str}"] = round(
            env_backtest.drawdown, 4
        )
        wandb.run.summary[f"total_trades_{all_unseen}_{proba_str}"] = (
            env_backtest.total_trades
        )
        if proba == 0.5:
            test_metrics.update(
                {"drawdown": round(env_backtest.drawdown, 4), "pL": running_pl_list[-1]}
            )

        wandb.run.summary[f"end_pL_{all_unseen}_{proba_str}"] = running_pl_list[-1]
        plot_probas_dict.update({f"pL_{proba_str}": running_pl_list})

    # log backtest plots
    wandb.log(
        {
            f"backtest_{all_unseen}": wandb.plot.line_series(
                xs=[[*range(env_backtest.position)] for _ in plot_probas_dict],
                ys=[list(v) for k, v in plot_probas_dict.items()],
                keys=list(plot_probas_dict.keys()),
                title=f"Backtest {all_unseen.capitalize()}",
                xname="Steps",
            ),
        }
    )

    return test_metrics


def data_artifact_download():
    wandb.init(project="rlot", job_type="artifact download")
    artifact_version = "delta_bin:v0"
    artifact_path = wandb.use_artifact(artifact_version).download()
    wandb.finish()
    return artifact_path
