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
generator.load_state_dict(torch.load('bullx2_generatorv7_vii.pth', map_location=device))
generator.eval()

# --- Load quantile transformers ---
quantile_transformer_r1 = load('bullx2_quantile_transformer_r1.joblib')
quantile_transformer_r2 = load('bullx2_quantile_transformer_r2.joblib')

df = pd.read_feather('../data/bullx2v7.feather')
filtered_c = [seq.copy() for seq in np.array(df['c_chunk']) if len(seq) == 20]

global_min = np.load('./bullx2_global_minv7.npy')
global_max = np.load('./bullx2_global_maxv7.npy')

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
    if i < 3:
        plot_candlestick(open_, high, low, close_, i)

# Save all candles to feather
all_df = pd.DataFrame({
    'Open': [c['Open'] for c in all_candles],
    'High': [c['High'] for c in all_candles],
    'Low': [c['Low'] for c in all_candles],
    'Close': [c['Close'] for c in all_candles],
})
all_df.to_feather('synthetic_bullx2_candles.feather')

def count_ohlc_violations(open_, high, low, close_):
    violations = 0
    for o, h, l, c in zip(open_, high, low, close_):
        if not (h >= o and h >= c and o >= l and c >= l):
            violations += 1
    return violations

num_violations = count_ohlc_violations(open_, high, low, close_)

df = pd.read_feather('../data/bullx2v7.feather')
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

    # Flatten all r1 and r2 values for global quantile transform
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

def plot_tsne_3d(X_tsne, y):
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], c='blue', label='Real', alpha=0.6)
    plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], c='red', label='Synthetic', alpha=0.6)
    plt.title('2D t-SNE of Denormalized Real vs Synthetic (Close, r1, r2)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- FID implementation for time series ---

# === MAIN CALL ===

def generate_tsne_and_fid_all_sequences_normalized(generator, normalized_data, latent_dim, device):
    # Load quantile transformers
    quantile_transformer_r1 = load('bullx2_quantile_transformer_r1.joblib')
    quantile_transformer_r2 = load('bullx2_quantile_transformer_r2.joblib')

    real_all = extract_real_all_sequences_normalized(normalized_data)
    fake_all = generate_all_sequences_normalized(generator, real_all.shape[0], latent_dim, device)

    # Denormalize c_chunk and inverse transform r1 and r2 for both real and fake
    def denormalize_all(arr, global_min, global_max, quantile_transformer_r1, quantile_transformer_r2):
        arr = arr.copy()
        # Denormalize c_chunk (index 0)
        arr[:,0,:] = arr[:,0,:] * (global_max[0,0,:] - global_min[0,0,:]) + global_min[0,0,:]
        # Inverse transform r1 and r2
        arr[:,1,:] = quantile_transformer_r1.inverse_transform(arr[:,1,:].reshape(-1, 1)).reshape(arr[:,1,:].shape)
        arr[:,2,:] = quantile_transformer_r2.inverse_transform(arr[:,2,:].reshape(-1, 1)).reshape(arr[:,2,:].shape)
        return arr

    # Use global_min and global_max from earlier normalization
    global_min = np.min(normalized_data, axis=(0, 2), keepdims=True)
    global_max = np.max(normalized_data, axis=(0, 2), keepdims=True)

    real_all_denorm = denormalize_all(real_all, global_min, global_max, quantile_transformer_r1, quantile_transformer_r2)
    fake_all_denorm = denormalize_all(fake_all, global_min, global_max, quantile_transformer_r1, quantile_transformer_r2)

    # t-SNE visualization
    # X, y = prepare_for_tsne(real_all_denorm.reshape(real_all_denorm.shape[0], -1), fake_all_denorm.reshape(fake_all_denorm.shape[0], -1))
    # X_tsne = run_tsne(X)
    # plot_tsne_3d(X_tsne, y)  

# --- Compare bearx2v7.feather and bullx2v7.feather ---
def compare_bear_bull_tsne():
    # Generate synthetic bear (not bearx2) data (fake)
    bearv7_df = pd.read_feather('../data/bearv7.feather')
    bearv7_df = bearv7_df[bearv7_df['c_chunk'].apply(lambda x: len(x) == 20)].reset_index(drop=True)
    n_fake_bearv7 = bearv7_df.shape[0]
    bearv7_generator = Generator().to(device)
    bearv7_generator.load_state_dict(torch.load('../bear/bear_generatorv7_vii.pth', map_location=device))
    bearv7_generator.eval()
    z_bearv7 = torch.rand(n_fake_bearv7, LATENT_DIM, device=device) * 2 - 1
    with torch.no_grad():
        fake_bearv7 = bearv7_generator(z_bearv7).cpu().numpy()
    bearv7_data = prepare_sequences(bearv7_df)
    bearv7_global_min = np.min(bearv7_data, axis=(0, 2), keepdims=True)
    bearv7_global_max = np.max(bearv7_data, axis=(0, 2), keepdims=True)
    bearv7_quantile_transformer_r1 = load('../bear/bear_quantile_transformer_r1.joblib')
    bearv7_quantile_transformer_r2 = load('../bear/bear_quantile_transformer_r2.joblib')
    fake_bearv7_denorm = fake_bearv7.copy()
    fake_bearv7_denorm[:,0,:] = fake_bearv7_denorm[:,0,:] * (bearv7_global_max[0,0,:] - bearv7_global_min[0,0,:]) + bearv7_global_min[0,0,:]
    fake_bearv7_denorm[:,1,:] = bearv7_quantile_transformer_r1.inverse_transform(fake_bearv7_denorm[:,1,:].reshape(-1, 1)).reshape(fake_bearv7_denorm[:,1,:].shape)
    fake_bearv7_denorm[:,2,:] = bearv7_quantile_transformer_r2.inverse_transform(fake_bearv7_denorm[:,2,:].reshape(-1, 1)).reshape(fake_bearv7_denorm[:,2,:].shape)
    # Generate synthetic bull (not bullx2) data (fake)
    bullv7_df = pd.read_feather('../data/bullv7.feather')
    bullv7_df = bullv7_df[bullv7_df['c_chunk'].apply(lambda x: len(x) == 20)].reset_index(drop=True)
    n_fake_bullv7 = bullv7_df.shape[0]
    bullv7_generator = Generator().to(device)
    bullv7_generator.load_state_dict(torch.load('../bull/bull_generatorv7_vii.pth', map_location=device))
    bullv7_generator.eval()
    z_bullv7 = torch.rand(n_fake_bullv7, LATENT_DIM, device=device) * 2 - 1
    with torch.no_grad():
        fake_bullv7 = bullv7_generator(z_bullv7).cpu().numpy()
    bullv7_data = prepare_sequences(bullv7_df)
    bullv7_global_min = np.min(bullv7_data, axis=(0, 2), keepdims=True)
    bullv7_global_max = np.max(bullv7_data, axis=(0, 2), keepdims=True)
    bullv7_quantile_transformer_r1 = load('../bull/bull_quantile_transformer_r1.joblib')
    bullv7_quantile_transformer_r2 = load('../bull/bull_quantile_transformer_r2.joblib')
    fake_bullv7_denorm = fake_bullv7.copy()
    fake_bullv7_denorm[:,0,:] = fake_bullv7_denorm[:,0,:] * (bullv7_global_max[0,0,:] - bullv7_global_min[0,0,:]) + bullv7_global_min[0,0,:]
    fake_bullv7_denorm[:,1,:] = bullv7_quantile_transformer_r1.inverse_transform(fake_bullv7_denorm[:,1,:].reshape(-1, 1)).reshape(fake_bullv7_denorm[:,1,:].shape)
    fake_bullv7_denorm[:,2,:] = bullv7_quantile_transformer_r2.inverse_transform(fake_bullv7_denorm[:,2,:].reshape(-1, 1)).reshape(fake_bullv7_denorm[:,2,:].shape)

    # Generate synthetic flat data (fake)
    flat_df = pd.read_feather('../data/flatv7.feather')
    flat_df = flat_df[flat_df['c_chunk'].apply(lambda x: len(x) == 20)].reset_index(drop=True)
    n_fake_flat = flat_df.shape[0]
    flat_generator = Generator().to(device)
    flat_generator.load_state_dict(torch.load('../flat/flat_generatorv7_vii.pth', map_location=device))
    flat_generator.eval()
    z_flat = torch.rand(n_fake_flat, LATENT_DIM, device=device) * 2 - 1
    with torch.no_grad():
        fake_flat = flat_generator(z_flat).cpu().numpy()
    flat_data = prepare_sequences(flat_df)
    flat_global_min = np.min(flat_data, axis=(0, 2), keepdims=True)
    flat_global_max = np.max(flat_data, axis=(0, 2), keepdims=True)
    flat_quantile_transformer_r1 = load('../flat/flat_quantile_transformer_r1.joblib')
    flat_quantile_transformer_r2 = load('../flat/flat_quantile_transformer_r2.joblib')
    fake_flat_denorm = fake_flat.copy()
    fake_flat_denorm[:,0,:] = fake_flat_denorm[:,0,:] * (flat_global_max[0,0,:] - flat_global_min[0,0,:]) + flat_global_min[0,0,:]
    fake_flat_denorm[:,1,:] = flat_quantile_transformer_r1.inverse_transform(fake_flat_denorm[:,1,:].reshape(-1, 1)).reshape(fake_flat_denorm[:,1,:].shape)
    fake_flat_denorm[:,2,:] = flat_quantile_transformer_r2.inverse_transform(fake_flat_denorm[:,2,:].reshape(-1, 1)).reshape(fake_flat_denorm[:,2,:].shape)
    # Helper to load, normalize, and denormalize a dataset
    def process_df(df_path, r1_path, r2_path):
        df = pd.read_feather(df_path)
        df = df[df['c_chunk'].apply(lambda x: len(x) == 20)].reset_index(drop=True)
        qt_r1 = load(r1_path)
        qt_r2 = load(r2_path)
        data = prepare_sequences(df)
        gmin = np.min(data, axis=(0, 2), keepdims=True)
        gmax = np.max(data, axis=(0, 2), keepdims=True)
        norm = (data - gmin) / (gmax - gmin + 1e-8)
        def denorm(arr):
            arr = arr.copy()
            arr[:,0,:] = arr[:,0,:] * (gmax[0,0,:] - gmin[0,0,:]) + gmin[0,0,:]
            arr[:,1,:] = qt_r1.inverse_transform(arr[:,1,:].reshape(-1, 1)).reshape(arr[:,1,:].shape)
            arr[:,2,:] = qt_r2.inverse_transform(arr[:,2,:].reshape(-1, 1)).reshape(arr[:,2,:].shape)
            return arr
        return denorm(norm)

    # Bull x2
    bull_denorm = process_df('../data/bullx2v7.feather', 'bullx2_quantile_transformer_r1.joblib', 'bullx2_quantile_transformer_r2.joblib')
    # Bear x2
    bear_denorm = process_df('../data/bearx2v7.feather', '../bearx2/bearx2_quantile_transformer_r1.joblib', '../bearx2/bearx2_quantile_transformer_r2.joblib')
    # Bull v7
    bullv7_denorm = process_df('../data/bullv7.feather', '../bull/bull_quantile_transformer_r1.joblib', '../bull/bull_quantile_transformer_r2.joblib')
    # Bear v7
    bearv7_denorm = process_df('../data/bearv7.feather', '../bear/bear_quantile_transformer_r1.joblib', '../bear/bear_quantile_transformer_r2.joblib')
    # Flat
    flat_denorm = process_df('../data/flatv7.feather', '../flat/flat_quantile_transformer_r1.joblib', '../flat/flat_quantile_transformer_r2.joblib')

    # Generate synthetic bullx2 data (fake)
    n_fake_bull = bull_denorm.shape[0]
    z_bull = torch.rand(n_fake_bull, LATENT_DIM, device=device) * 2 - 1
    with torch.no_grad():
        fake_bull = generator(z_bull).cpu().numpy()
    bull_df = pd.read_feather('../data/bullx2v7.feather')
    bull_df = bull_df[bull_df['c_chunk'].apply(lambda x: len(x) == 20)].reset_index(drop=True)
    bull_data = prepare_sequences(bull_df)
    bull_global_min = np.min(bull_data, axis=(0, 2), keepdims=True)
    bull_global_max = np.max(bull_data, axis=(0, 2), keepdims=True)
    bull_quantile_transformer_r1 = load('bullx2_quantile_transformer_r1.joblib')
    bull_quantile_transformer_r2 = load('bullx2_quantile_transformer_r2.joblib')
    fake_bull_denorm = fake_bull.copy()
    fake_bull_denorm[:,0,:] = fake_bull_denorm[:,0,:] * (bull_global_max[0,0,:] - bull_global_min[0,0,:]) + bull_global_min[0,0,:]
    fake_bull_denorm[:,1,:] = bull_quantile_transformer_r1.inverse_transform(fake_bull_denorm[:,1,:].reshape(-1, 1)).reshape(fake_bull_denorm[:,1,:].shape)
    fake_bull_denorm[:,2,:] = bull_quantile_transformer_r2.inverse_transform(fake_bull_denorm[:,2,:].reshape(-1, 1)).reshape(fake_bull_denorm[:,2,:].shape)

    # Generate synthetic bearx2 data (fake)
    bearx2_generator = Generator().to(device)
    bearx2_generator.load_state_dict(torch.load('../bearx2/bearx2_generatorv7_vii.pth', map_location=device))
    bearx2_generator.eval()
    n_fake_bear = bear_denorm.shape[0]
    z_bear = torch.rand(n_fake_bear, LATENT_DIM, device=device) * 2 - 1
    with torch.no_grad():
        fake_bear = bearx2_generator(z_bear).cpu().numpy()
    bear_df = pd.read_feather('../data/bearx2v7.feather')
    bear_df = bear_df[bear_df['c_chunk'].apply(lambda x: len(x) == 20)].reset_index(drop=True)
    bear_data = prepare_sequences(bear_df)
    bear_global_min = np.min(bear_data, axis=(0, 2), keepdims=True)
    bear_global_max = np.max(bear_data, axis=(0, 2), keepdims=True)
    bear_quantile_transformer_r1 = load('../bearx2/bearx2_quantile_transformer_r1.joblib')
    bear_quantile_transformer_r2 = load('../bearx2/bearx2_quantile_transformer_r2.joblib')
    fake_bear_denorm = fake_bear.copy()
    fake_bear_denorm[:,0,:] = fake_bear_denorm[:,0,:] * (bear_global_max[0,0,:] - bear_global_min[0,0,:]) + bear_global_min[0,0,:]
    fake_bear_denorm[:,1,:] = bear_quantile_transformer_r1.inverse_transform(fake_bear_denorm[:,1,:].reshape(-1, 1)).reshape(fake_bear_denorm[:,1,:].shape)
    fake_bear_denorm[:,2,:] = bear_quantile_transformer_r2.inverse_transform(fake_bear_denorm[:,2,:].reshape(-1, 1)).reshape(fake_bear_denorm[:,2,:].shape)

    # Prepare for t-SNE: concatenate all
    X = np.concatenate([
        bull_denorm.reshape(bull_denorm.shape[0], -1),
        bear_denorm.reshape(bear_denorm.shape[0], -1),
        bullv7_denorm.reshape(bullv7_denorm.shape[0], -1),
        bearv7_denorm.reshape(bearv7_denorm.shape[0], -1),
        flat_denorm.reshape(flat_denorm.shape[0], -1),
        fake_bull_denorm.reshape(fake_bull_denorm.shape[0], -1),
        fake_bear_denorm.reshape(fake_bear_denorm.shape[0], -1),
        fake_bullv7_denorm.reshape(fake_bullv7_denorm.shape[0], -1),
        fake_bearv7_denorm.reshape(fake_bearv7_denorm.shape[0], -1),
        fake_flat_denorm.reshape(fake_flat_denorm.shape[0], -1)
    ], axis=0)
    y = np.array(
        [0] * bull_denorm.shape[0] +
        [1] * bear_denorm.shape[0] +
        [2] * bullv7_denorm.shape[0] +
        [3] * bearv7_denorm.shape[0] +
        [4] * flat_denorm.shape[0] +
        [5] * fake_bull_denorm.shape[0] +
        [6] * fake_bear_denorm.shape[0] +
        [7] * fake_bullv7_denorm.shape[0] +
        [8] * fake_bearv7_denorm.shape[0] +
        [9] * fake_flat_denorm.shape[0]
    )
    X_tsne = run_tsne(X)
    fig = plt.figure(figsize=(16, 12))
    # Real: dark, Fake: light
    plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], c="#153cc8", label='Bull x2 (real)', alpha=0.8)
    plt.scatter(X_tsne[y == 5, 0], X_tsne[y == 5, 1], c='#7fa6f9', label='Bull x2 (fake)', alpha=0.4)
    plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], c='#a85b00', label='Bear x2 (real)', alpha=0.8)
    plt.scatter(X_tsne[y == 6, 0], X_tsne[y == 6, 1], c='#ffd699', label='Bear x2 (fake)', alpha=0.4)
    plt.scatter(X_tsne[y == 2, 0], X_tsne[y == 2, 1], c='#005c23', label='Bull (real)', alpha=0.8)
    plt.scatter(X_tsne[y == 7, 0], X_tsne[y == 7, 1], c='#a8f7c1', label='Bull (fake)', alpha=0.4)
    plt.scatter(X_tsne[y == 3, 0], X_tsne[y == 3, 1], c="#5b147c", label='Bear (real)', alpha=0.8)
    plt.scatter(X_tsne[y == 8, 0], X_tsne[y == 8, 1], c="#e0b3ff", label='Bear (fake)', alpha=0.4)
    plt.scatter(X_tsne[y == 4, 0], X_tsne[y == 4, 1], c="#8B0000", label='Flat (real)', alpha=0.8)
    plt.scatter(X_tsne[y == 9, 0], X_tsne[y == 9, 1], c="#f47777", label='Flat (fake)', alpha=0.4)
    plt.title('2D t-SNE: Real (dark) vs Fake (light) - Bull x2, Bear x2, Bull, Bear, Flat (Close, r1, r2)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Call your function ---
generate_tsne_and_fid_all_sequences_normalized(generator, normalized_data, LATENT_DIM, device)
compare_bear_bull_tsne()