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
SEQ_LENGTH = 20 

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

# --- Device selection ---
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# --- Load generator ---
generator = Generator().to(device)
generator.load_state_dict(torch.load('bearx2_generatorv7_vii.pth', map_location=device))
generator.eval()

# --- Load quantile transformers ---
quantile_transformer_r1 = load('bearx2_quantile_transformer_r1.joblib')
quantile_transformer_r2 = load('bearx2_quantile_transformer_r2.joblib')

df = pd.read_feather('../data/bearx2.feather')
filtered_c = [seq.copy() for seq in np.array(df['c_chunk']) if len(seq) == 20]

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
        style='classic',
        title=f'Synthetic M1 Candlestick Sample {sample_idx+1}',
        ylabel='Price',
        xlabel='Time',
        datetime_format='%H:%M',
        figratio=(4, 3),
    )

num_samples = 595
all_candles = []
all_samples = []
fig, axes = plt.subplots(3, 1, figsize=(3, 6))
if not isinstance(axes, np.ndarray):
    axes = [axes]  # Ensure axes is always iterable

for i in range(3):
    with torch.no_grad():
        z = torch.rand(1, LATENT_DIM, device=device) * 2 - 1
        sample = generator(z).cpu().numpy()  # shape: (1, 3, SEQ_LENGTH)
        sample = sample * (global_max - global_min) + global_min
    sample = sample[0]  # shape: (3, SEQ_LENGTH)
    open_, high, low, close_ = extract_ohlc_from_sample(sample)

    all_samples.append({
        'c_chunk': sample[0],
        'r1': sample[1],
        'r2': sample[2],
    })
    all_candles.append({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close_,
    })

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
        ax=axes[i],
        # title=f'Synthetic M1 Candlestick Sample {i+1}',
        ylabel='Price',
        datetime_format='%H:%M',
    )

plt.tight_layout()
plt.show()