
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

def analyze_synthetic_candles(
    generator_pth,
    r1_joblib,
    r2_joblib,
    global_min_npy,
    global_max_npy,
    output_csv,
    num_samples=595
):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_pth, map_location=device))
    generator.eval()

    quantile_transformer_r1 = load(r1_joblib)
    quantile_transformer_r2 = load(r2_joblib)

    global_min = np.load(global_min_npy)
    global_max = np.load(global_max_npy)

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

    def count_ohlc_violations(open_, high, low, close_):
        violations = 0
        for o, h, l, c in zip(open_, high, low, close_):
            if not (h >= o and h >= c and o >= l and c >= l):
                violations += 1
        return violations

    num_violations = 0
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

    all_df = pd.DataFrame({
        'o_chunk': [c['Open'] for c in all_candles],
        'h_chunk': [c['High'] for c in all_candles],
        'l_chunk': [c['Low'] for c in all_candles],
        'c_chunk': [c['Close'] for c in all_candles],
    })
    all_df.to_csv(output_csv)

    def parse_chunk(chunk):
        if isinstance(chunk, list):
            return np.array(chunk)
        elif isinstance(chunk, str):
            return np.array([float(x) for x in chunk.strip('[]').split()])
        else:
            raise TypeError("Unsupported type for chunk")

    results = []
    for idx, row in all_df.iterrows():
        o = parse_chunk(row['o_chunk'])
        h = parse_chunk(row['h_chunk'])
        l = parse_chunk(row['l_chunk'])
        c = parse_chunk(row['c_chunk'])

        ranges = h - l
        bodies = np.abs(c - o)
        if len(ranges) > 3:
            with np.errstate(divide='ignore', invalid='ignore'):
                range_to_body = np.where(ranges != 0, (ranges - bodies) / ranges, np.nan)
            avg_range_to_body = np.nanmean(range_to_body)
        else:
            range_to_body = np.where(ranges != 0, (ranges - bodies) / ranges, np.nan)
            avg_range_to_body = np.nanmean(range_to_body)

        avg_range = np.mean(ranges)

        if len(c) > 1:
            log_returns = np.diff(np.log(c))
            volatility = np.std(log_returns)
        else:
            volatility = np.nan

        results.append({
            # 'avg_range_to_body': avg_range_to_body,
            'avg_range': avg_range,
            # 'volatility': volatility,
        })

    results_df = pd.DataFrame(results)
    summary = results_df.agg(['mean', 'std'])
    print(summary)
    print()
    return summary

if __name__ == "__main__":
    # Example usage:
    print('Bull')
    analyze_synthetic_candles(
        generator_pth='./bull_generatorv7_vii.pth',
        r1_joblib='./bull_quantile_transformer_r1.joblib',
        r2_joblib='./bull_quantile_transformer_r2.joblib',
        global_min_npy='./bullx2_global_minv7.npy',
        global_max_npy='./bullx2_global_maxv7.npy',
        output_csv='./synthetic_bull_candles.csv',
        num_samples=595
    )
    print("Bear")
    analyze_synthetic_candles(
        generator_pth='./bear_generatorv7_vii.pth',
        r1_joblib='./bear_quantile_transformer_r1.joblib',
        r2_joblib='./bear_quantile_transformer_r2.joblib',
        global_min_npy='./bear_global_minv7.npy',
        global_max_npy='./bear_global_maxv7.npy',
        output_csv='./synthetic_bear_candles.csv',
        num_samples=595
    )
    print("Bullx2")
    analyze_synthetic_candles(
        generator_pth='./bullx2_generatorv7_vii.pth',
        r1_joblib='./bullx2_quantile_transformer_r1.joblib',
        r2_joblib='./bullx2_quantile_transformer_r2.joblib',
        global_min_npy='./bullx2_global_minv7.npy',
        global_max_npy='./bullx2_global_maxv7.npy',
        output_csv='./synthetic_bullx2_candles.csv',
        num_samples=595
    )
    print("Bearx2")
    analyze_synthetic_candles(
        generator_pth='./bearx2_generatorv7_vii.pth',
        r1_joblib='./bearx2_quantile_transformer_r1.joblib',
        r2_joblib='./bearx2_quantile_transformer_r2.joblib',
        global_min_npy='./bearx2_global_minv7.npy',
        global_max_npy='./bearx2_global_maxv7.npy',
        output_csv='./synthetic_bearx2_candles.csv',
        num_samples=595
    )
    print("Flat")
    analyze_synthetic_candles(
        generator_pth='./flat_generatorv7_vii.pth',
        r1_joblib='./flat_quantile_transformer_r1.joblib',
        r2_joblib='./flat_quantile_transformer_r2.joblib',
        global_min_npy='./flat_global_minv7.npy',
        global_max_npy='./flat_global_maxv7.npy',
        output_csv='./synthetic_flat_candles.csv',
        num_samples=595
    )