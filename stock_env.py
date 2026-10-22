import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=10, initial_balance=10000):
        super(StockTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Action space: 0 (Short), 1 (Hold), 2 (Long)
        self.action_space = spaces.Discrete(3)

        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_30', 'Daily_Return', 'Volatility_10']

        self.df_norm = self.df.copy()
        for col in self.features:
            self.df_norm[col] = (self.df[col] - self.df[col].mean()) / (self.df[col].std() + 1e-8)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, len(self.features)), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.position = 0
        self.net_worth = self.initial_balance

        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df_norm.loc[self.current_step - self.window_size : self.current_step - 1, self.features].values
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        prev_price = self.df.loc[self.current_step - 1, 'Close']

        if action == 2:
            self.position = 1
        elif action == 0:
            self.position = -1
        elif action == 1:
            self.position = 0

        if self.position == 1:
            daily_change = (current_price - prev_price) / prev_price
        elif self.position == -1:
            daily_change = (prev_price - current_price) / prev_price
        else:
            daily_change = 0

        self.net_worth = self.net_worth * (1 + daily_change)

        # We need >= 1% daily return. So target return is 0.01 per day.
        # Let's give massive rewards for getting > 0.01 and massive penalties for missing it
        reward = daily_change * 1000

        # Heavily penalize holding cash to force it to trade
        if self.position == 0:
            reward -= 50

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        if self.net_worth <= 0:
            done = True
            reward = -10000

        obs = self._next_observation()

        return obs, reward, done, False, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth}, Position: {self.position}")
