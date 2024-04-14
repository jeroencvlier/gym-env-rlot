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
#     "/Users/jeroenvanlier/Documents/Github/gym-env-rlot/artifacts/percentage_bin:v3"
# )


def load_data(
    artifact_path,
    ticker,
    test_train,
    test_size,
    noise_factor,
):
    data_dir = os.path.join(artifact_path, "train", ticker)
    # load data from .pkl
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


class BuySellUndEnv(gym.Env):
    def __init__(self, config):
        super(BuySellUndEnv, self).__init__()

        self.test_train = config["test_train"]

        if self.test_train not in ["train", "test", "all"]:
            raise ValueError("Invalid test_train")
        if self.test_train == "train":
            self.noise_factor = 0.05
            self.observation_noise_std = 0.035
        else:
            self.noise_factor = 0.0
            self.observation_noise_std = 0.0

        self.max_episode_steps = 500

        self.data, self.scaler_price, self.feature_names = load_data(
            config["artifact_path"],
            self.test_train,
            test_size=self.max_episode_steps + 1,
            noise_factor=self.noise_factor,
        )
        self.feature_names += ["in_trade", "in_price", "pl"]
        if self.test_train == "all":
            self.max_episode_steps = len(self.data) - 1
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.data[0]["scaled"]) + 3,),
            dtype=np.float32,
        )

    def obs_(
        self,
    ):
        pl = 0.0
        if self.in_trade == 1:
            in_tr = 1
            in_price = self.scaler_price.transform(
                self.entry_market_price.reshape(-1, 1)
            )[0][0]
            pl = self.pl
            if pl != 0.0:
                pl = np.clip((pl / 500) - 1, -1, 1)
        else:
            in_tr = -1
            in_price = 0

        self.observation = self.data[self.position]["scaled"]
        if self.test_train == "train":
            self.observation += np.random.normal(
                0, self.observation_noise_std, len(self.observation)
            )
            self.observation = np.clip(self.observation, -1, 1)
        self.observation = np.append(self.observation, [in_tr, in_price, pl])
        return self.observation

    def step(self, action):
        self.steps += 1
        self.position += 1
        self.done = self.steps >= (self.max_episode_steps)
        self.reward = self.total_pl

        self.action = action

        if self.action == 0:
            # action 0 is to not be in a trade
            if self.in_trade == 0:
                # Not in a trade, no action taken
                pass
            elif self.in_trade == 1:
                # In a trade, exit the trade
                self.trade_pl = round(
                    self.data[self.position]["underlyingPrice"]
                    - self.entry_market_price,
                    3,
                )
                self.pct_pl_running = round(
                    self.pct_pl_running + (self.trade_pl / self.entry_market_price),
                    6,
                )

                self.total_pl = round(self.total_pl + self.trade_pl, 3)
                del self.entry_market_price
                self.reward += self.total_pl + (self.trade_pl * 5)
                if self.trade_pl < 0:
                    self.reward -= 10
                self.in_trade = 0
                self.trade_pl = 0.0
                self.pl = 0.0

            else:
                # this is not possible but for clarity
                raise ValueError("Invalid trade state")

        elif self.action == 1:
            # action 1 is to be in a trade
            if self.in_trade == 0:
                # Not in a trade, enter the trade
                self.total_trades += 1
                # add noise of between -0.05 and 0.05 to entry price
                self.noise = round(
                    np.random.uniform(-self.noise_factor, self.noise_factor), 3
                )
                self.entry_market_price = (
                    self.data[self.position]["underlyingPrice"] + self.noise
                )
                self.in_trade = 1
                self.pl = round(
                    self.data[self.position]["underlyingPrice"]
                    - self.entry_market_price,
                    3,
                )

            elif self.in_trade == 1:
                self.pl = round(
                    self.data[self.position]["underlyingPrice"]
                    - self.entry_market_price,
                    3,
                )
                self.reward += self.total_pl * self.pct_pl_running
                self.reward += self.pl * self.pct_pl_running
            else:
                # this is not possible but for clarity
                raise ValueError("Invalid trade state")
        else:
            # not a valid action, statement for clarity
            raise ValueError("Invalid action")

        if self.done:
            if self.total_trades < 5:
                self.reward = -1000

        self.running_pl = round(self.total_pl + self.pl, 3)
        self.running_pl_adjusted = round(self.running_pl * self.pct_pl_running, 6)
        self.running_pl_list.append(self.running_pl)
        self.running_pl_adjusted_list.append(self.running_pl_adjusted)
        self.drawdown = calculate_maximum_drawdown(self.running_pl_list)
        self.drawdown_adjusted = calculate_maximum_drawdown(
            self.running_pl_adjusted_list
        )
        # self.sharpe_ratio = calculate_sharpe_ratio(self.running_pl_list)
        # self.sortino_ratio = calculate_sortino_ratio(self.running_pl_list)
        # self.calmar_ratio = calculate_calmar_ratio(self.running_pl_list)

        # REWARD TIME

        # self.reward -= self.drawdown
        self.reward -= self.drawdown_adjusted

        # self.reward += self.sharpe_ratio * 30
        # self.reward += self.sortino_ratio
        # self.reward -= self.calmar_ratio
        self.reward = round(self.reward, 6)

        truncated = False
        info = {"total_pl": self.total_pl, "total_trades": self.total_trades}

        return self.obs_(), round(self.reward, 10), self.done, truncated, info

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.in_trade = 0
        if self.test_train == "train":
            self.start_position = np.random.randint(
                0, len(self.data) - self.max_episode_steps
            )
        elif (self.test_train == "test") or (self.test_train == "all"):
            self.start_position = 0
        else:
            raise ValueError("Invalid test_train")

        self.position = self.start_position
        self.total_pl = 0.0
        self.total_trades = 0
        self.trade_pl = 0.0
        self.pl = 0.0
        self.running_pl = 0.0
        self.running_pl_list = []
        self.running_pl_adjusted_list = []
        self.done = False
        self.pct_pl_running = 1.0
        return self.obs_(), {}

    def custom_logs(self):
        log = {
            "trades": self.total_trades,
            "drawdown": self.drawdown,
            "total_pl": self.running_pl,
        }
        return log

    def render(self, mode="human"):
        print(
            f"STEP: {self.steps}, Position: {self.position}, Action: {self.action}, In-Trade: {self.in_trade}, PL: {self.pl}, Trade PL: {self.trade_pl}, Total PL: {self.total_pl}, Total Trades: {self.total_trades}, reward: {self.reward}, Underlying Price: {self.data[self.position]['underlyingPrice']}"
        )

    def env_state(self):
        # return a dict of values
        render_dict = {
            "step": self.steps,
            "position": self.position,
            "action": self.action,
            "in_trade": self.in_trade,
            "pl": self.pl,
            "running_pl": self.running_pl,
            "running_pl_adjusted": self.running_pl_adjusted,
            "pct_pl_running": self.pct_pl_running,
            "trade_pl": self.trade_pl,
            "total_pl": self.total_pl,
            "total_trades": self.total_trades,
            "reward": self.reward,
            "drawdown": self.drawdown,
            "drawdown_adjusted": self.drawdown_adjusted,
            "underlying_price": np.round(
                self.data[self.position]["underlyingPrice"], 3
            ),
        }
        return render_dict


if __name__ == "__main__":
    config = {
        "artifact_path": "/Users/jeroenvanlier/Documents/Github/gym-env-rlot/artifacts/percentage_bin_nounderlying_staggered:v0",
        "test_train": "train",
    }

    env = BuySellUndEnv(config)

    pct_pl_running = []
    for i in tqdm(range(1)):
        env_states = []
        obs = env.reset()
        done = False
        action = 1
        while not done:
            # action = env.action_space.sample()

            if env.steps % 2 == 0:
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

    # plot running_pl and running_pl_adjusted
    df[["running_pl", "running_pl_adjusted"]].plot()

    df.head(25)
    df
