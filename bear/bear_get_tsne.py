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
generator.load_state_dict(torch.load('bear_generatorv7_vii.pth', map_location=device))
generator.eval()

# --- Load quantile transformers ---
quantile_transformer_r1 = load('bear_quantile_transformer_r1.joblib')
quantile_transformer_r2 = load('bear_quantile_transformer_r2.joblib')

df = pd.read_feather('../data/bearv7.feather')
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
        style='charles',
        title=f'Synthetic M1 Candlestick Sample {sample_idx+1}',
        ylabel='Price',
        xlabel='Time',
        datetime_format='%H:%M',
        figratio=(4, 3),
    )

num_samples = 595
all_candles = []
all_samples = []
for i in range(num_samples):
    with torch.no_grad():
        z = torch.rand(1, LATENT_DIM, device=device) * 2 - 1
        sample = generator(z).cpu().numpy()  # shape: (1, 3, SEQ_LENGTH)
        sample = sample * (global_max - global_min) + global_min
    sample = sample[0]  # shape: (3, SEQ_LENGTH)
    open_, high, low, close_ = extract_ohlc_from_sample(sample)

    all_samples.append({
        'c_chunk' : sample[0],
        'r1' : sample[1],
        'r2' : sample[2],
    })
    all_candles.append({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close_, 
    })
    if i < 5 and True:
        plot_candlestick(open_, high, low, close_, i)

# Save all candles to feather
all_df = pd.DataFrame({
    'Open': [c['Open'] for c in all_candles],
    'High': [c['High'] for c in all_candles],
    'Low': [c['Low'] for c in all_candles],
    'Close': [c['Close'] for c in all_candles],
})
all_df.to_feather('synthetic_bear_candles.feather')

def count_ohlc_violations(open_, high, low, close_):
    violations = 0
    for o, h, l, c in zip(open_, high, low, close_):
        if not (h >= o and h >= c and o >= l and c >= l):
            violations += 1
    return violations

num_violations = count_ohlc_violations(open_, high, low, close_)
print(f"Number of OHLC violations: {num_violations}")


df = pd.read_feather('../data/bearv7.feather')
df = df[df['o_chunk'].apply(lambda x: len(x) == 20)].reset_index(drop=True)

# Select a random row
random_idx = np.random.randint(len(df))
row = df.iloc[random_idx]

# Extract OHLC values
open_chunk = row['o_chunk']
high_chunk = row['h_chunk']
low_chunk = row['l_chunk']
close_chunk = row['c_chunk']

# Convert to numpy arrays (if needed)
open_chunk = np.array(open_chunk)
high_chunk = np.array(high_chunk)
low_chunk = np.array(low_chunk)
close_chunk = np.array(close_chunk)

# Reconstruct OHLC by shifting close (if following your logic)
open_ = close_chunk[:-1]
close_ = close_chunk[1:]
high_ = high_chunk[1:]
low_ = low_chunk[1:]

# Create datetime index
dates = pd.date_range(start="2023-01-01 00:00", periods=len(close_) , freq='min')

# Assemble DataFrame
ohlc_df = pd.DataFrame({
    'Open': open_,
    'High': high_,
    'Low': low_,
    'Close': close_,
}, index=dates)

c_chunks_df = []

def prepare_sequences(df, seq_length=20):
    sequences = []
    # Prepare filtered chunks for each type
    filtered_chunks = {}
    for col in ['c_chunk', 'r1', 'r2']:
        raw_dataset = np.array(df[col])
        filtered_chunks[col] = [seq.copy() for seq in raw_dataset if len(seq) == seq_length]

    # c_chunk logic unchanged
    c_filtered = filtered_chunks['c_chunk']
    middle = 55000
    for seq in c_filtered:
        shift = middle - seq[0]
        seq += shift
        c_chunks_df.append(seq)

    # bearten all r1 and r2 values for global quantile transform
    r1_all = np.array(filtered_chunks['r1']).reshape(-1, 1)
    r2_all = np.array(filtered_chunks['r2']).reshape(-1, 1)
    n_quantiles_r1 = min(1000, r1_all.shape[0])
    n_quantiles_r2 = min(1000, r2_all.shape[0])
    
    r1_all_trans = quantile_transformer_r1.fit_transform(r1_all).reshape(-1)
    r2_all_trans = quantile_transformer_r2.fit_transform(r2_all).reshape(-1)

    # Reconstruct chunk structure
    r1_trans_chunks = [r1_all_trans[i*seq_length:(i+1)*seq_length] for i in range(len(filtered_chunks['r1']))]
    r2_trans_chunks = [r2_all_trans[i*seq_length:(i+1)*seq_length] for i in range(len(filtered_chunks['r2']))]

    sequences.append(c_filtered)
    sequences.append(r1_trans_chunks)
    sequences.append(r2_trans_chunks)
    # Combine into array of shape (n_samples, 3, seq_length)
    combined = np.stack([sequences[0], sequences[1], sequences[2]], axis=1)
    return combined

# Prepare dataset
data = prepare_sequences(df)
global_min = np.min(data, axis=(0, 2), keepdims=True)  # Min per sequence type
global_max = np.max(data, axis=(0, 2), keepdims=True)  # Max per sequence type
denom = global_max - global_min
denom[denom == 0] = 1  # Avoid division by zero
normalized_data = (data - global_min) / denom

# --- Generate synthetic normalized sequences (c_chunk, r1, r2) ---
def generate_all_sequences_normalized(generator, n_samples, latent_dim, device):
    z = torch.rand(n_samples, latent_dim, device=device) * 2 - 1
    with torch.no_grad():
        generated = generator(z).cpu().numpy()  # Already normalized output
    return generated  # shape: (n_samples, 3, seq_length)

# --- Prepare real normalized sequences (c_chunk, r1, r2) ---
def extract_real_all_sequences_normalized(normalized_data):
    return normalized_data  # shape: (n_samples, 3, seq_length)

# --- Prepare for t-SNE ---
def prepare_for_tsne(real_all, fake_all):
    X = np.concatenate([real_all, fake_all], axis=0)
    y = np.array([0] * real_all.shape[0] + [1] * fake_all.shape[0])
    return X, y

# --- Run t-SNE ---
def run_tsne(X):
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    return tsne.fit_transform(X_scaled)

# --- Plot t-SNE ---

def plot_tsne_2d(X_tsne, y):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], c='blue', label='Real', alpha=0.6)
    ax.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], c='red', label='Synthetic', alpha=0.6)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend()
    plt.show()

# --- FID implementation for time series ---

# === MAIN CALL ===

def generate_tsne_and_fid_all_sequences_normalized(generator, normalized_data, latent_dim, device):
    # Load quantile transformers
    quantile_transformer_r1 = load('bear_quantile_transformer_r1.joblib')
    quantile_transformer_r2 = load('bear_quantile_transformer_r2.joblib')

    real_all = extract_real_all_sequences_normalized(normalized_data)
    fake_all = generate_all_sequences_normalized(generator, real_all.shape[0], latent_dim, device)

    # Inverse transform r1 and r2 for both real and fake
    def inverse_r1_r2(arr):
        arr = arr.copy()
        arr[:,1,:] = quantile_transformer_r1.inverse_transform(arr[:,1,:].reshape(-1, 1)).reshape(arr[:,1,:].shape)
        arr[:,2,:] = quantile_transformer_r2.inverse_transform(arr[:,2,:].reshape(-1, 1)).reshape(arr[:,2,:].shape)
        return arr

    real_all_inv = inverse_r1_r2(real_all)
    fake_all_inv = inverse_r1_r2(fake_all)

    # t-SNE visualization
    X, y = prepare_for_tsne(real_all_inv.reshape(real_all_inv.shape[0], -1), fake_all_inv.reshape(fake_all_inv.shape[0], -1))
    X_tsne = run_tsne(X)
    plot_tsne_2d(X_tsne, y)

# --- Call your function ---
generate_tsne_and_fid_all_sequences_normalized(generator, normalized_data, LATENT_DIM, device)

# Aggregate all c_chunk values
# all_c_values = np.concatenate([sample['c_chunk'] for sample in all_samples])

# bearten_c_chunks_df = np.concatenate(c_chunks_df)

# plt.figure(figsize=(10, 6))
# plt.hist(all_c_values, bins=50, color='skyblue', edgecolor='black', alpha=0.5)
# plt.hist(bearten_c_chunks_df, bins=50, color='green', edgecolor='black', alpha=0.5)
# plt.title('Distribution of c_chunk values across all_samples')
# plt.xlabel('c_chunk value')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
