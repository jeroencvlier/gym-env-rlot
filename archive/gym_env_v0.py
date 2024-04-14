import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from gym_env_rlot.buy_sell.gym_env_utils import calculate_maximum_drawdown
import numpy as np
from tqdm import tqdm

import umap

from gym_env_rlot.buy_sell.gym_env_utils import (
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
    test_train,
    test_size,
    noise_factor,
    dim_features: list = ["aS", "bS"],
):

    df = pd.read_parquet(artifact_path)
    df.set_index("humanTime", inplace=True)
    df.sort_index(inplace=True)
    logging.info(f"Loaded data from {artifact_path}")
    na_count = len(df)
    df.dropna(inplace=True)
    na_count = na_count - len(df)
    logging.info(f"Dropped {na_count} rows with NA values")
    # drop underlyingPrice from df as asign to variable
    underlyingPrice = df.pop("tickerPrice").to_numpy()
    assert "tickerPrice" not in df.columns, "tickerPrice still in df"

    scaler_price = MinMaxScaler(feature_range=(-1 + noise_factor, 1 - noise_factor))
    underlyingPriceScaled = scaler_price.fit_transform(underlyingPrice.reshape(-1, 1))
    data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df)
    feature_names = list(df.columns)
    # zip data to the scaled data into an ordered dictionary
    data = {
        en: {
            "underlyingPrice": underlyingPrice[en],
            "scaled": data[en],
            "underlyingPriceScaled": underlyingPriceScaled[en],
        }
        for en, i in enumerate(df.index)
    }

    if test_train == "train":
        data = list(data.values())[:-test_size]
    elif test_train == "test":
        data = list(data.values())[-test_size:]
    elif test_train == "all":
        data = list(data.values())

    logging.info(
        f"Loaded {len(data)} data points for {test_train}, columns: {len(feature_names)}"
    )
    return data, scaler_price, feature_names


class BuySellUndEnv(gym.Env):
    def __init__(self, config):
        super(BuySellUndEnv, self).__init__()

        self.test_train = config["test_train"]

        if self.test_train not in ["train", "test", "all"]:
            raise ValueError("Invalid test_train")
        if self.test_train == "train":
            self.noise_factor = 0.05
            self.observation_noise_std = 0.025
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
        if action == 0:
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

        elif action == 1:
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
                self.reward += self.total_pl
                self.reward += self.pl
            else:
                # this is not possible but for clarity
                raise ValueError("Invalid trade state")
        else:
            # not a valid action, statement for clarity
            raise ValueError("Invalid action")

        if self.done:
            if self.total_trades < 5:
                self.reward = -1000

        self.runnning_pl = round(self.total_pl + self.pl, 3)
        self.running_pl_list.append(self.runnning_pl)
        self.drawdown = calculate_maximum_drawdown(self.running_pl_list)
        # self.sharpe_ratio = calculate_sharpe_ratio(self.running_pl_list)
        # self.sortino_ratio = calculate_sortino_ratio(self.running_pl_list)
        # self.calmar_ratio = calculate_calmar_ratio(self.running_pl_list)

        # REWARD TIME
        self.reward = round(self.reward, 4)
        self.reward -= self.drawdown
        # self.reward += self.sharpe_ratio * 30
        # self.reward += self.sortino_ratio
        # self.reward -= self.calmar_ratio

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
        self.runnning_pl = 0.0
        self.running_pl_list = []
        self.done = False
        return self.obs_(), {}

    def custom_logs(self):
        log = {
            "trades": self.total_trades,
            "drawdown": self.drawdown,
            "total_pl": self.runnning_pl,
        }
        return log

    def render(self, mode="human"):
        print(
            f"STEP: {self.steps}, Position: {self.position}, Action: {self.action}, In-Trade: {self.in_trade}, PL: {self.pl}, Trade PL: {self.trade_pl}, Total PL: {self.total_pl}, Total Trades: {self.total_trades}, reward: {self.reward}, Underlying Price: {self.data[self.position]['underlyingPrice']}"
        )


if __name__ == "__main__":
    config = {
        "artifact_path": "/Users/jeroenvanlier/Documents/Github/gym-env-rlot/artifacts/percentage_bin_nounderlying_staggered:v0",
        "test_train": "train",
    }

    sharpes = []
    # calmer = []
    # sortino = []
    drawdown = []
    b = False
    for i in tqdm(range(100)):
        env = BuySellUndEnv(config)
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            sharpes.append(env.sharpe_ratio)
            # calmer.append(env.calmar_ratio)
            # sortino.append(env.sortino_ratio)
            drawdown.append(env.drawdown)

        # print(env.custom_logs())

    print(max(sharpes), min(sharpes), np.mean(sharpes), np.median(sharpes))
    # print(max(calmer), min(calmer), np.mean(calmer), np.median(calmer))
    # print(max(sortino), min(sortino), np.mean(sortino), np.median(sortino))
    print(max(drawdown), min(drawdown), np.mean(drawdown), np.median(drawdown))

    # calculate q25 and q75 for sharpes and drawdowns
    sharpesq25 = np.percentile(sharpes, 25)
    sharpesq75 = np.percentile(sharpes, 75)
    drawdownq25 = np.percentile(drawdown, 25)
    drawdownq75 = np.percentile(drawdown, 75)

    # calculate scale between the sharpe and drawdown q25 and q25 values
    scale25 = sharpesq25 / drawdownq25
    scale75 = sharpesq75 / drawdownq75

    # scale the sharpes to the drawdown scale
    sharpes = [i / scale25 for i in sharpes]

    # plot the sharpes

    import matplotlib.pyplot as plt

    plt.hist(sharpes, bins=50)
    plt.show()

    # plot
    import matplotlib.pyplot as plt

    plt.hist(sharpes, bins=50)
    plt.show()

    plt.hist(drawdown, bins=50)
    plt.show()
