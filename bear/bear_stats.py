import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import mplfinance as mpf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump, load
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

LATENT_DIM = 50 
SEQ_LENGTH = 30 

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(LATENT_DIM, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.c_out = torch.nn.Linear(64, SEQ_LENGTH)
        self.r1_out = torch.nn.Linear(64, SEQ_LENGTH)
        self.r2_out = torch.nn.Linear(64, SEQ_LENGTH)

    def forward(self, z):
        x = torch.nn.functional.leaky_relu(self.bn1(self.fc1(z)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        c = self.c_out(x)
        r1 = self.r1_out(x)
        r2 = self.r2_out(x)
        out = torch.stack([c, r1, r2], dim=1)  # (batch, 3, SEQ_LENGTH)
        return out

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
generator.load_state_dict(torch.load('bear_generatorv7_vii_len30.pth', map_location=device))
generator.eval()

quantile_transformer_r1 = load('bear_quantile_transformer_r1_len30.joblib')
quantile_transformer_r2 = load('bear_quantile_transformer_r2_len30.joblib')

global_min = np.load('./global_min.npy')
global_max = np.load('./global_max.npy')

def extract_ohlc_from_sample(sample):
    close = sample[0]
    r1 = quantile_transformer_r1.inverse_transform(sample[1].reshape(-1, 1)).flatten()
    r2 = quantile_transformer_r2.inverse_transform(sample[2].reshape(-1, 1)).flatten()
    open_ = close[0:-1]
    low = []
    high = []
    for i in range(1, len(close)):
        l = np.minimum(open_[i-1], close[i]) - (r1[i] * close[i])
        low.append(l)
    for i in range(1, len(close)):
        h = np.maximum(open_[i-1], close[i]) + (r2[i] * close[i])
        high.append(h)
    close_ = close[1:]
    return open_, high, low, close_

def plot_candlestick(open_, high, low, close_, sample_idx):
    dates = pd.date_range("2023-01-01 00:00", periods=len(open_), freq='min')
    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close_,
    }, index=dates)
    mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'Synthetic M1 Candlestick Sample {sample_idx+1}',
        ylabel='Price',
        xlabel='Time',
        datetime_format='%H:%M',
        figratio=(4, 3),
    )

def count_ohlc_violations(open_, high, low, close_):
    violations = 0
    for o, h, l, c in zip(open_, high, low, close_):
        if not (h >= o and h >= c and o >= l and c >= l):
            violations += 1
    return violations

num_violations = 0
num_samples = 595
total = 0
all_candles = []
for i in range(num_samples):
    with torch.no_grad():
        z = torch.rand(1, LATENT_DIM, device=device) * 2 - 1
        sample = generator(z).cpu().numpy()  # shape: (1, 3, SEQ_LENGTH)
        sample = sample * (global_max - global_min) + global_min
    sample = sample[0] 
    open_, high, low, close_ = extract_ohlc_from_sample(sample)
    num_violations += count_ohlc_violations(open_, high, low, close_)
    total += len(open_)
    all_candles.append({
        'Open': list(open_),
        'High': high,
        'Low': low,
        'Close': list(close_),
    })
    if i < 5 and False:
        plot_candlestick(open_, high, low, close_, i)

print(f"Number of OHLC violations: {num_violations} out of {total}")

# Save all candles to feather
all_df = pd.DataFrame({
    'o_chunk': [c['Open'] for c in all_candles],
    'h_chunk': [c['High'] for c in all_candles],
    'l_chunk': [c['Low'] for c in all_candles],
    'c_chunk': [c['Close'] for c in all_candles],
})
all_df.to_csv('synthetic_bear_candles_len30.csv')

def parse_chunk(chunk):
    if isinstance(chunk, list):
        return np.array(chunk)
    elif isinstance(chunk, str):
        return np.array([float(x) for x in chunk.strip('[]').split()])
    else:
        raise TypeError("Unsupported type for chunk")

def parse_np_float_list(s):
    # Remove brackets and np.float64, then convert to float
    s = s.replace('np.float64(', '').replace(')', '')
    return np.array([float(x) for x in s.strip('[]').split(',')])

df = all_df

results = []
for idx, row in df.iterrows():
    o = parse_chunk(row['o_chunk'])
    h = parse_chunk(row['h_chunk'])
    l = parse_chunk(row['l_chunk'])
    c = parse_chunk(row['c_chunk'])

    # Range-to-Body Ratio
    ranges = h - l
    bodies = np.abs(c - o)
    if len(ranges) > 3:
        with np.errstate(divide='ignore', invalid='ignore'):
            range_to_body = np.where(ranges != 0, (ranges - bodies) / ranges, np.nan)
        avg_range_to_body = np.nanmean(range_to_body)
    else:
        range_to_body = np.where(ranges != 0, (ranges - bodies) / ranges, np.nan)
        avg_range_to_body = np.nanmean(range_to_body)

    # Candle Range Distribution
    avg_range = np.mean(ranges)

    if len(c) > 1:
        log_returns = np.diff(np.log(c))
        volatility = np.std(log_returns)
    else:
        volatility = np.nan

    results.append({
        'avg_range_to_body': avg_range_to_body,
        'avg_range': avg_range,
        'volatility': volatility,
    })

# Convert results to DataFrame for further analysis
results_df = pd.DataFrame(results)
# print(results_df)

# Compute and display average and standard deviation for all columns
summary = results_df.agg(['mean', 'std'])
print("\nSummary (mean and std):")
print(summary)