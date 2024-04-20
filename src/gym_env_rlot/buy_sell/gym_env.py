import os
import logging
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt
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

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.in_trade = 0
        self.idx = 0
        self.trade_count = 0
        self.reward = 0
        self.action = 0

        # set the start position to a random position in the data
        if self.test_train == "train":
            # set seed to numpy for reproducibility
            if seed is not None:
                np.random.seed(seed)  # check how to set seed on level of ray
            self.idx = np.random.randint(0, len(self.data) - self.max_episode_steps)

        # set the start position to the beginning of the data
        self.first_underlying_price = self.data[self.idx]["underlyingPrice"]

        # this is the pl of the last trade
        self.trade_pl = 0.0
        self.trade_pl_pct = 0.0

        # these are the cumulitave pl of all trades
        self.running_pl_base = 0.0
        self.running_pl = 0.0
        self.running_pl_pct = 0.0
        self.running_pl_list = []
        self.running_pl_pct_list = []

        self.done = False
        self.entry_price = np.nan
        return self._obs(), {}

    def _obs(self):
        trade_pl = 0.0
        if self.in_trade == 1:
            trade_pl = self.trade_pl
            if trade_pl != 0.0:
                trade_pl = np.clip((trade_pl / 500) - 1, -1, 1)  # TODO: revisit

        self.obs = self.data[self.idx]["data"]
        if self.test_train == "train":
            self.obs += np.random.normal(0, self.noise_ratio, len(self.obs))
            self.obs = np.clip(self.obs, -1, 1)
        self.obs = np.append(self.obs, [trade_pl, self.in_trade])
        # assert everything is a float32
        self.obs = np.array(self.obs, dtype=np.float32)

        return self.obs

    def _trade_calculations(self, trade_reset=False):
        if self.in_trade == 1:
            underlying_price = self.data[self.idx]["underlyingPrice"]
            self.trade_pl = round(underlying_price - self.entry_price, 3)
            self.trade_pl_pct = round(self.trade_pl / self.entry_price, 3)

        self.running_pl = round(self.running_pl_base + self.trade_pl, 3)
        self.running_pl_list.append(self.running_pl)
        self.running_pl_pct = round(
            self.running_pl / self.data[self.idx]["underlyingPrice"], 3
        )
        self.running_pl_pct_list.append(self.running_pl_pct)

        # calculate drawdowns
        self.drawdown = calculate_maximum_drawdown(self.running_pl_list)
        self.drawdown_pct = calculate_maximum_drawdown(self.running_pl_pct_list)
        if trade_reset:
            self.running_pl_base += self.trade_pl
            self.in_trade = 0
            self.exit_pl = self.trade_pl
            self.trade_pl = 0.0
            self.entry_price = np.nan

    def step(self, action):
        self.steps += 1
        self.idx += 1
        self.action = action
        trade_reset = False
        self.done = self.steps >= (self.max_episode_steps)

        if self.action == 0:  # action 0 means exit/stay out trade
            if self.in_trade == 0:  # Not in a trade, no action taken
                pass
            elif self.in_trade == 1:  # In a trade, exit the trade
                trade_reset = True

        if self.action == 1:  # action 1 means enter/stay in trade
            if self.in_trade == 0:  # Not in a trade, enter the trade
                self.trade_count += 1
                self.in_trade = 1
                price_noise = np.random.uniform(-self.noise_ratio, self.noise_ratio)
                self.entry_price = (
                    self.data[self.idx - 1]["underlyingPrice"] + price_noise
                )

        self._trade_calculations(trade_reset)
        self._reward(trade_reset)

        return self._obs(), self.reward, self.done, False, self.info

    def _reward(self, trade_reset):
        self.reward = 0.0

        self.info = {}
        if self.done:
            self.info = {
                "trade_count": self.trade_count,
                "drawdown": self.drawdown,
                "drawdown_pct": self.drawdown_pct,
                "running_pl": self.running_pl,
                "running_pl_pct": self.running_pl_pct,
                "sharpe_ratio": calculate_sharpe_ratio(self.running_pl_list),
                # "sortino_ratio": calculate_sortino_ratio(self.running_pl_list),
                # "calmar_ratio": calculate_calmar_ratio(self.running_pl_list),
            }

        self.reward += self.drawdown
        self.reward += self.running_pl

        # slow penalty for too little trades
        if self.trade_count <= 8:
            self.reward += self.trade_count - 8

        # hard penalty for too many trades
        if self.trade_count > (self.max_episode_steps * 0.2):
            self.reward -= self.trade_count

        # double points on trade reault when exiting a trade
        if trade_reset:
            self.reward += self.exit_pl * 2

        self.price_differece = (
            self.data[self.idx]["underlyingPrice"] - self.first_underlying_price
        )
        # negative reward for not beating the market and increases over time
        if self.running_pl < self.price_differece:
            self.reward -= self.steps * 0.01
        # positive reward for beating the market and increases over time
        elif self.running_pl > self.price_differece:
            self.reward += self.steps * 0.1

        self.reward = round(self.reward, 6)

    def render(self, mode="human"):
        pass

    def env_state(self):
        # return a dict of values
        render_dict = {
            "step": self.steps,
            "position": self.idx,
            "action": self.action,
            "in_trade": self.in_trade,
            "running_pl": self.running_pl,
            "running_pl_pct": self.running_pl_pct,
            "trade_pl": self.trade_pl,
            "trade_pl_pct": self.trade_pl_pct,
            "trade_count": self.trade_count,
            "reward": self.reward,
            "drawdown": self.drawdown,
            "drawdown_pct": self.drawdown_pct,
            "underlying_price": np.round(self.data[self.idx]["underlyingPrice"], 3),
            "entry_price": self.entry_price,
            "price_differece": self.price_differece,
        }

        return render_dict


if __name__ == "__main__":
    config = {
        "artifact_path": "/Users/jeroenvanlier/Documents/Github/gym-env-rlot/artifacts/desc_stats_nounderlying:v1",
        "test_train": "test1",
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
            # ALGO Logic will be placed here
            if env.steps % 4 == 0:
                # alternate action between 0 and 1 every 5 steps
                if action == 0:
                    action = 1
                elif action == 1:
                    action = 0

            # take action
            obs, reward, done, truncated, info = env.step(action)
            env_states.append(env.env_state())
            pct_pl_running.append(env.env_state()["running_pl_pct"])

    min(pct_pl_running), max(pct_pl_running)

    pd.set_option("display.max_columns", None, "display.max_rows", 50)

    df = pd.DataFrame(env_states)
    df["underlying_price_adj"] = df["underlying_price"] - df["underlying_price"].iloc[0]
    # calculate the difference from each row
    df["row_diff"] = df["underlying_price_adj"].diff()

    # plot running_pl and underlying_price_adj
    df[["running_pl", "underlying_price_adj", "reward"]].plot()

    # plot (env.running_pl_pct_list)
    plt.plot(env.running_pl_pct_list)
    plt.show()

    df.head(25)
    display(
        df[
            [
                "action",
                "in_trade",
                "trade_count",
                "row_diff",
                "trade_pl",
                "running_pl",
                "underlying_price",
                "entry_price",
                "underlying_price_adj",
                "reward",
            ]
        ].head(25)
    )

    df.tail(25)
