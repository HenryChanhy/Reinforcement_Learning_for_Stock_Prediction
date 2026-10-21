import pandas as pd
import numpy as np

def prepare_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Calculate some technical indicators
    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()

    # Returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Volatility
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()

    # Drop NAs
    df.dropna(inplace=True)

    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file} with shape {df.shape}")

if __name__ == "__main__":
    prepare_data("data/^NDX.csv", "data/^NDX_processed.csv")
