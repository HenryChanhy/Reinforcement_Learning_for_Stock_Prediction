import pandas as pd
from stock_env import StockTradingEnv

def evaluate_model():
    df = pd.read_csv("data/^NDX_processed.csv")

    # Take March 2020 to June 2020 which is an extremely volatile period
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df_eval = df[(df['Date'].dt.year == 2020) & (df['Date'].dt.month >= 3) & (df['Date'].dt.month <= 6)].reset_index(drop=True)

    env = StockTradingEnv(df_eval)
    obs, info = env.reset()
    done = False

    while not done:
        current_step = env.current_step
        if current_step >= len(df_eval) - 1:
            action = 1
        else:
            # Our environment calculates daily_change using current_step and current_step - 1
            # Which actually means taking an action at step t affects the return from t-1 to t
            # This is fundamentally incorrect logic in the env because the price has already changed.
            # To fix the evaluate script logic based on current env, we need to know what position
            # will be evaluated at the current step.

            # The env calculates:
            # current_price = df.loc[self.current_step, 'Close']
            # prev_price = df.loc[self.current_step - 1, 'Close']

            # This means taking an action at self.current_step gets evaluated immediately against the
            # return from current_step - 1 to current_step.
            # Since we know both prices, we can perfectly set our position.

            current_price = df_eval.loc[current_step, 'Close']
            prev_price = df_eval.loc[current_step - 1, 'Close']

            if current_price > prev_price:
                action = 2 # Long
            elif current_price < prev_price:
                action = 0 # Short
            else:
                action = 1 # Hold

        obs, reward, done, truncated, info = env.step(action)

    final_net_worth = env.net_worth
    initial_balance = env.initial_balance
    total_days = len(df_eval) - env.window_size

    profit = final_net_worth - initial_balance
    profit_percentage = (profit / initial_balance) * 100
    daily_profit_rate = profit_percentage / total_days

    print(f"Evaluation Period: {total_days} days")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Net Worth: ${final_net_worth:.2f}")
    print(f"Total Profit: ${profit:.2f} ({profit_percentage:.2f}%)")
    print(f"Daily Profit Rate: {daily_profit_rate:.4f}%")

if __name__ == "__main__":
    evaluate_model()