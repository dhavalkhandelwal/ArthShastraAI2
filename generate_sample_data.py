import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 3 years of daily trading dates (roughly 252 days/year)
dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='B')

# Define assets with synthetic realistic characteristics (drift and volatility)
assets = {
    'AAPL': {'start': 130.0, 'drift': 0.15, 'vol': 0.25},
    'JPM': {'start': 125.0, 'drift': 0.10, 'vol': 0.20},
    'TLT': {'start': 150.0, 'drift': 0.03, 'vol': 0.10},
    'GLD': {'start': 180.0, 'drift': 0.05, 'vol': 0.15}
}

market_returns = np.random.normal(0.0003, 0.01, len(dates))
data = {'Date': dates}

for ticker, props in assets.items():
    daily_drift = props['drift'] / 252
    daily_vol = props['vol'] / np.sqrt(252)
    idiosyncratic = np.random.normal(0, daily_vol, len(dates))
    
    if ticker in ['AAPL', 'JPM']:
        beta = 1.2 if ticker == 'AAPL' else 0.9
        daily_returns = daily_drift + beta * market_returns + idiosyncratic * 0.7
    elif ticker == 'TLT':
        daily_returns = daily_drift - 0.2 * market_returns + idiosyncratic
    else:
        daily_returns = daily_drift + idiosyncratic
        
    prices = props['start'] * np.exp(np.cumsum(daily_returns))
    data[ticker] = prices

df = pd.DataFrame(data)
output_file = 'sample_portfolio.csv'
df.to_csv(output_file, index=False)
print(f"Successfully generated {output_file} with {len(df)} rows.")
