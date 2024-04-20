import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from ray.rllib.algorithms import PPOConfig
from src.gym_env_rlot.buy_sell.gym_env import BuySellUndEnv
from src.gym_env_rlot.buy_sell.gym_env import calculate_maximum_drawdown
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
    buyhold: bool = False,
    env_backtest: BuySellUndEnv = None,
) -> list:
    logging.info(f"Backtesting {all_unseen} data")
    plot_probas_dict = {}
    plot_adjusted_pl = {}
    test_metrics = {}
    observation, _ = env_backtest.reset()
    while not env_backtest.done:
        if buyhold:
            action = 1
        else:
            action = algo.compute_single_action(observation, explore=False)

        observation, reward, done, truncated, info = env_backtest.step(action)
        previous_action = action

    running_pl_list = env_backtest.running_pl_list
    running_pl_adjusted = env_backtest.running_pl_adjusted_list
    # log merrics
    wandb.run.summary[f"drawdown_{all_unseen}"] = round(env_backtest.drawdown, 4)
    wandb.run.summary[f"trades_{all_unseen}"] = env_backtest.total_trades
    wandb.run.summary[f"pct_rpL_{all_unseen}"] = env_backtest.pct_pl_running

    test_metrics.update(
        {"drawdown": round(env_backtest.drawdown, 4), "pL": running_pl_list[-1]}
    )

    wandb.run.summary[f"pL_{all_unseen}"] = running_pl_list[-1]
    wandb.run.summary[f"pL_adj_{all_unseen}"] = running_pl_adjusted[-1]

    plot_probas_dict.update({f"pL": running_pl_list})
    plot_adjusted_pl.update({f"pL_adjusted": running_pl_adjusted})

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
    wandb.log(
        {
            f"backtest_adjusted_{all_unseen}": wandb.plot.line_series(
                xs=[[*range(env_backtest.position)] for _ in plot_adjusted_pl],
                ys=[list(v) for k, v in plot_adjusted_pl.items()],
                keys=list(plot_adjusted_pl.keys()),
                title=f"Backtest Adjusted {all_unseen.capitalize()}",
                xname="Steps",
            ),
        }
    )

    return test_metrics

