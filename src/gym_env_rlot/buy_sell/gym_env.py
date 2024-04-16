import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from src.gym_env_rlot.buy_sell.gym_env_utils import calculate_maximum_drawdown
import numpy as np
from tqdm import tqdm
import os
import umap
import pickle

from src.gym_env_rlot.buy_sell.gym_env_utils import (
    calculate_maximum_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
# artifact_path = (
#     '/Users/jeroenvanlier/Documents/Github/gym-env-rlot/artifacts/desc_stats_nounderlying:v0'
# )


def load_data(
    artifact_path: str,
    ticker: str,
    test_train: str,
):
    if test_train not in ["train", "test1", "test2"]:
        raise ValueError("Invalid test_train")

    data_dir = os.path.join(artifact_path, "data", ticker)

    if test_train == "train":
        with open(os.path.join(data_dir, "train_data.pkl"), "rb") as f:
            data = pickle.load(f)

    elif test_train == "test1":
        with open(os.path.join(data_dir, "test_front_data.pkl"), "rb") as f:
            data = pickle.load(f)

    elif test_train == "test2":
        with open(os.path.join(data_dir, "test_back_data.pkl"), "rb") as f:
            data = pickle.load(f)

    return data


def get_feature_names(artifact_path: str, ticker: str):
    model_path = os.path.join(artifact_path, "model", ticker, "columns.pkl")
    with open(model_path, "rb") as f:
        feature_names = pickle.load(f)
    feature_names = list(feature_names)
    return feature_names


class BuySellUndEnv(gym.Env):
    def __init__(self, config):
        super(BuySellUndEnv, self).__init__()

        self.test_train = config["test_train"]
        self.ticker = config["ticker"]
        self.artifact_path = config["artifact_path"]
        self.max_episode_steps = 500

        self.data = load_data(self.artifact_path, self.ticker, self.test_train)
        self.feature_names = get_feature_names(self.artifact_path, self.ticker)
        self.feature_names += ["trade_pl", "in_trade"]  # removed the "in_price" feature
        self.action_space = Discrete(2)

        self.num_features = len(self.feature_names)
        low = np.full((self.num_features,), -1.0, dtype=np.float32)
        high = np.full((self.num_features,), 1.0, dtype=np.float32)
        low[-1] = 0.0  # Adjust the last dimension to range from 0 to 1 for in_trade
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        self.noise_ratio = 0.0
        if self.test_train == "train":
            self.noise_ratio = 0.05

    def obs_(self):
        trade_pl = 0.0
        if self.in_trade == 1:
            trade_pl = self.trade_pl
            if trade_pl != 0.0:
                trade_pl = np.clip((trade_pl / 500) - 1, -1, 1)

        self.obs = self.data[self.idx]["data"]
        if self.test_train == "train":
            self.obs += np.random.normal(0, self.noise_ratio, len(self.obs))
            self.obs = np.clip(self.obs, -1, 1)
        self.obs = np.append(self.obs, [trade_pl, self.in_trade])
        return self.obs

    def trade_pl_(self):
        return self.data[self.idx]["underlyingPrice"] - self.entry_price

    def step(self, action):
        self.steps += 1
        self.idx += 1
        self.done = self.steps >= (self.max_episode_steps)
        self.reward = self.total_pl  # TODO: Revisit this
        self.action = action

        if self.action == 1:  # action 1 means enter/stay in trade
            if self.in_trade == 0:  # Not in a trade, enter the trade
                self.trade_count += 1
                self.in_trade = 1
                entry_noise = np.random.uniform(-self.noise_ratio, self.noise_ratio)
                self.entry_price = self.data[self.idx - 1][
                    "underlyingPrice"
                ]  # + entry_noise # TODO: Uncomment this

            # elif self.in_trade == 1:

            # self.reward += self.trade_pl * 1  # TODO: Revisit this
            # self.reward += self.trade_pl_pct  # TODO: Revisit this

            self.trade_pl = self.trade_pl_()

        if self.action == 0:  # action 0 means exit/stay out trade
            if self.in_trade == 0:  # Not in a trade, no action taken
                pass
            elif self.in_trade == 1:  # In a trade, exit the trade
                self.trade_pl = self.trade_pl_()
                self.trade_pl_pct = self.trade_pl / self.entry_price
                self.total_pl = round(self.total_pl + self.trade_pl, 3)
                # self.reward += self.total_pl + (self.trade_pl * 1)
                # if self.trade_pl < 0:
                #     self.reward += 2 * self.trade_pl
                self.in_trade = 0
                self.trade_pl = 0.0
                self.entry_price = np.nan

        self.running_pl_list.append(self.running_pl)
        running_pl_pct = self.running_pl / self.data[self.idx]["underlyingPrice"]
        self.running_pl_pct_list.append(running_pl_pct)
        self.drawdown = calculate_maximum_drawdown(self.running_pl_list)
        self.drawdown_pct = calculate_maximum_drawdown(self.running_pl_pct_list)

        # self.reward -= self.drawdown
        # self.reward -= self.drawdown_pct

        info = {"total_pl": self.total_pl, "trade_count": self.trade_count}

        self.reward = round(self.reward, 6)
        self.trade_pl = round(self.trade_pl, 3)
        self.running_pl = round(self.total_pl + self.trade_pl, 3)
        return self.obs_(), self.reward, self.done, False, info

    def reset(self, *, seed=42, options=None):
        self.steps = 0
        self.in_trade = 0
        self.idx = 0

        # set the start position to a random position in the data
        if self.test_train == "train":
            self.idx = np.random.randint(0, len(self.data) - self.max_episode_steps)

        self.trade_count = 0
        self.trade_pl = 0.0
        self.total_pl = 0.0
        self.running_pl = 0.0
        self.running_pl_list = []
        self.running_pl_pct_list = []
        self.trade_pl_pct = 1.0
        self.done = False
        self.entry_price = np.nan
        return self.obs_(), {}

    def custom_logs(self):
        log = {
            "trades": self.trade_count,
            "drawdown": self.drawdown,
            "total_pl": self.running_pl,
        }
        return log

    def render(self, mode="human"):
        print(
            f"STEP: {self.steps}, Position: {self.idx}, Action: {self.action}, In-Trade: {self.in_trade}, Trade PL: {self.trade_pl}, Total PL: {self.total_pl}, Total Trades: {self.trade_count}, reward: {self.reward}, Underlying Price: {self.data[self.idx]['underlyingPrice']}"
        )

    def env_state(self):
        # return a dict of values
        render_dict = {
            "step": self.steps,
            "position": self.idx,
            "action": self.action,
            "in_trade": self.in_trade,
            "running_pl": self.running_pl,
            "pct_pl_running": self.trade_pl_pct,
            "trade_pl": self.trade_pl,
            "total_pl": self.total_pl,
            "trade_count": self.trade_count,
            "reward": self.reward,
            "drawdown": self.drawdown,
            "underlying_price": np.round(self.data[self.idx]["underlyingPrice"], 3),
            "entry_price": self.entry_price,
        }
        return render_dict


if __name__ == "__main__":
    config = {
        "artifact_path": "/Users/jeroenvanlier/Documents/Github/gym-env-rlot/artifacts/desc_stats_nounderlying:v0",
        "test_train": "train",
        "ticker": "SPY",
    }

    env = BuySellUndEnv(config)

    pct_pl_running = []
    for i in tqdm(range(1)):
        env_states = []
        obs = env.reset()
        done = env.done
        action = 1
        while not done:
            # action = env.action_space.sample()

            if env.steps % 4 == 0:
                # alternate action between 0 and 1 every 5 steps
                if action == 0:
                    action = 1
                elif action == 1:
                    action = 0

            obs, reward, done, truncated, info = env.step(action)
            env_states.append(env.env_state())
            pct_pl_running.append(env.env_state()["pct_pl_running"])

    min(pct_pl_running), max(pct_pl_running)

    pd.set_option("display.max_columns", None, "display.max_rows", 50)

    df = pd.DataFrame(env_states)
    df["underlying_price_cstual"] = (
        df["underlying_price"] - df["underlying_price"].iloc[0]
    )
    # calculate the difference from each row
    df["row_diff"] = df["underlying_price_cstual"].diff()

    # plot running_pl and underlying_price_adj
    df[["running_pl", "underlying_price_cstual"]].plot()

    # line plot the rewards
    df["reward"].plot()

    df.head(25)
    display(
        df[
            [
                "action",
                "in_trade",
                "trade_count",
                "row_diff",
                "trade_pl",
                "underlying_price",
                "underlying_price_cstual",
            ]
        ].head(25)
    )
