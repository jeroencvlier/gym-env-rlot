import wandb
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(artifact_path, test_train, test_size, noise_factor):
    df = pd.read_parquet(artifact_path)
    df.set_index("humanTime", inplace=True)
    pd.set_option("display.max_columns", None)
    df.dropna(inplace=True)
    underlyingPrice = df["underlyingPrice"].to_numpy()
    scaler_price = MinMaxScaler(feature_range=(-1 + noise_factor, 1 - noise_factor))
    underlyingPriceScaled = scaler_price.fit_transform(underlyingPrice.reshape(-1, 1))
    data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df)
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
    return data, scaler_price


def calculate_maximum_drawdown(cumulative_pnl):
    # Initialize variables for the peak, maximum drawdown, and current drawdown
    peak = cumulative_pnl[0]
    maximum_drawdown = 0

    for pnl in cumulative_pnl:
        # Update the peak if current P&L is higher
        if pnl > peak:
            peak = pnl
        # Calculate the current drawdown
        drawdown = peak - pnl
        # Update the maximum drawdown if the current drawdown is larger
        if drawdown > maximum_drawdown:
            maximum_drawdown = drawdown

    return maximum_drawdown




class BuySellUndEnv(gym.Env):
    def __init__(self, config):
        super(BuySellUndEnv, self).__init__()

        self.noise_factor = 0.05
        self.test_train = config["test_train"]
        self.max_episode_steps = 500

        self.data, self.scaler_price = load_data(
            config["data_path"],
            self.test_train,
            test_size=self.max_episode_steps + 1,
            noise_factor=self.noise_factor,
        )
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

        self.observation = np.append(
            self.data[self.position]["scaled"], [in_tr, in_price, pl]
        )
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
        if self.drawdown > 1:
            self.reward -= self.drawdown - 1

        self.reward = round(self.reward, 4)
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