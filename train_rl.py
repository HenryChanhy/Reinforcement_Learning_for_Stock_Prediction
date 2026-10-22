import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_env import StockTradingEnv

def train_model():
    # Load processed data
    df = pd.read_csv("data/^NDX_processed.csv")

    # Take a longer training period for better generalization
    df_train = df.tail(3000).reset_index(drop=True)

    env = StockTradingEnv(df_train)
    vec_env = DummyVecEnv([lambda: env])

    # Increase learning rate and ent_coef to encourage exploration
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.001, ent_coef=0.05)

    print("Starting training...")
    # Train for more timesteps
    model.learn(total_timesteps=150000)
    print("Training finished.")

    model.save("models/ppo_stock_model")
    print("Model saved to models/ppo_stock_model.zip")

if __name__ == "__main__":
    train_model()
