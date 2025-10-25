import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import seaborn as sns

csv_path = os.path.join(os.path.dirname(__file__), 'data', 'BTCUSDT_1m_2020_2024_data.csv')
df = pd.read_csv(csv_path)
df = df.drop(['Open time','Volume','Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], axis=1)
og_data = np.array(df['Close'])
data = np.array(df['Close'])
opens = np.array(df['Open'])
highs = np.array(df['High'])
lows = np.array(df['Low'])
print(len(data))

class Filter:
    def butter_lowpass_filter(self, data, Wn, order):
        b, a = butter(order, Wn, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

filt = Filter()

Wn = 2/9
order = 4

filtered_data = filt.butter_lowpass_filter(data, Wn, order)

time = np.arange(len(data))

def find_peaks_troughs(data):
    peaks = []
    troughs = []
    
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks.append(i)
        elif data[i] < data[i - 1] and data[i] < data[i + 1]:
            troughs.append(i)
    
    return peaks, troughs

def chunk_at_extremums(og_data, data, time_data, highs, lows, opens):
    data = np.array(data)
    time_data = np.array(time_data)
    
    maxima, minima = find_peaks_troughs(data)
    
    extremums = sorted(set(minima).union(maxima))
    cut_points = [0] + extremums + [len(data)]
    
    chunks = [data[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points)-1)]
    og_chunks = [og_data[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points)-1)]
    time_chunks = [time_data[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points)-1)]
    high_chunks = [highs[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points)-1)]
    low_chunks = [lows[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points)-1)]
    open_chunks = [opens[cut_points[i]:cut_points[i+1]] for i in range(len(cut_points)-1)]
    return og_chunks, chunks, time_chunks, high_chunks, low_chunks, open_chunks
    
og_data_chunks, filtered_data_chunks, time_chunks, high_chunks, low_chunks, open_chunks = chunk_at_extremums(og_data, filtered_data, time, highs, lows, opens)

def get_rise_run_slope(y_chunk, x_chunk):
    rise = y_chunk[-1] - y_chunk[0]
    run = x_chunk[-1] - x_chunk[0]

    if run == 0:
        return 0.0  
    return rise / run

    
slopes = []
for y_chunk, x_chunk in zip(filtered_data_chunks,time_chunks):
    slope = get_rise_run_slope(y_chunk,x_chunk)
    slopes.append(slope)

slope_values = np.array(slopes)
percentiles = np.percentile(slope_values, [20, 40, 60, 80])
print(percentiles)
print(f"Max slope: {np.max(slope_values)}")
print(f"Min slope: {np.min(slope_values)}")
def categorize_slope_percentile(slope, percentiles):
    if slope <= percentiles[0]:
        return '2xBear'
    elif slope <= percentiles[1]:
        return 'Bear'
    elif slope <= percentiles[2]:
        return 'Flat'
    elif slope <= percentiles[3]:
        return 'Bull'
    else:
        return '2xBull'
    
data_for_df = []
for slope, n_chunk, chunk, t_chunk, h_chunk, l_chunk, o_chunk in zip(slopes, filtered_data_chunks, og_data_chunks, time_chunks, high_chunks, low_chunks, open_chunks):
    category = categorize_slope_percentile(slope, percentiles)
    data_for_df.append({
        # 'slope': slope,
        'category': category,
        # 'n_chunk': n_chunk,
        'o_chunk': o_chunk,
        # 't_chunk': t_chunk,
        'h_chunk': h_chunk,
        'l_chunk': l_chunk,
        'c_chunk': chunk
    })

df_slopes = pd.DataFrame(data_for_df)

category_counts = df_slopes['category'].value_counts()
print(category_counts)

vis_df = df_slopes[:16]
category_colors = {
    'Bull': 'green',
    'Bear': 'red',
    'Flat': 'gray',
    '2xBull': 'blue',
    '2xBear': 'darkred'
}

plt.figure(figsize=(15, 5))

x = 0  
prev_val = None

for _, row in vis_df.iterrows():
    chunk = row['c_chunk']
    category = row['category']
    color = category_colors.get(category, 'black')
    
    x_vals = list(range(x, x + len(chunk)))

    if prev_val is not None:
        plt.plot([x - 1, x], [prev_val, chunk[0]], color=color, linewidth=2)

    plt.plot(x_vals, chunk, color=color)

    x += len(chunk)
    prev_val = chunk[-1]

handles = [plt.Line2D([0], [0], color=color, label=cat) for cat, color in category_colors.items()]
# plt.legend(handles=handles, title="Category")
# plt.title('Sample Colored by Category')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Split df_slopes into a dictionary of DataFrames based on the 'category' column
category_dfs = {category: df for category, df in df_slopes.groupby('category')}
bear_df = category_dfs['Bear']
bull_df = category_dfs['Bull']
bearx2_df = category_dfs['2xBear']
bullx2_df = category_dfs['2xBull']
flat_df = category_dfs['Flat']

dfs = {
    'bear': bear_df,
    'bull': bull_df,
    'bearx2': bearx2_df,
    'bullx2': bullx2_df,
    'flat': flat_df
}

for name, df in dfs.items():
    chunk_lengths = df['c_chunk'].apply(len)
    min_length = chunk_lengths.min()
    max_length = chunk_lengths.max()
    
    print(f"{name}:")
    print(f"  Min chunk length: {min_length}")
    print(f"  Max chunk length: {max_length}\n")

def compute_relative_changes(chunk):
    return [(chunk[i+1] - chunk[i]) / chunk[i] for i in range(len(chunk)-1) if chunk[i] != 0]

groups = ['Bull', '2xBull', 'Bear', '2xBear', 'Flat']

mapping = {
    'Bull': bull_df,
    'Bear': bear_df,
    '2xBear': bearx2_df,
    '2xBull': bullx2_df,
    'Flat': flat_df
}

results = {}

for group in groups:
    temp = mapping[group]
    temp['relativeChange'] = temp['c_chunk'].apply(compute_relative_changes)

    nested_array = temp['relativeChange'].tolist()
    flat_array = [item for sublist in nested_array for item in sublist]
    
    mean = np.mean(flat_array)
    stdev = np.std(flat_array)
    
    results[group] = {'Mean': mean, 'Standard Deviation': stdev}

for group, stats in results.items():
    print(f'{group}')
    print(f"Mean: {stats['Mean']:.6f}")
    print(f"Standard Deviation: {stats['Standard Deviation']:.6f}")
    print()

# (min(O,C) - L) / C
def calculate_r1(row):
    l = np.array(row['l_chunk'])
    c = np.array(row['c_chunk'])
    o = np.array(row['o_chunk'])
    r1 = []
    for i in range(len(c)):
        r1.append((min(c[i],o[i]) - l[i])/c[i])
    return r1

# (H - max(O,C)) / C
def calculate_r2(row):
    h = np.array(row['h_chunk'])
    o = np.array(row['o_chunk'])
    c = np.array(row['c_chunk'])
    r2 = []
    for i in range(len(c)):
        r2.append((h[i] - max(o[i], c[i])) / c[i])
    return r2

# (O - PrevC) / C
def calculate_r3(row):
    c = np.array(row['c_chunk'])
    r3 = []
    for i in range(len(c)-1):
        r3.append(c[i] - c[i+1])
    return r3
    
def add_features(df):
    df['r1'] = df.apply(calculate_r1, axis=1)
    df['r2'] = df.apply(calculate_r2, axis=1)
    df['r3'] = df.apply(calculate_r3, axis=1)
    print(len(df['r1'].iloc[0]))
    print(len(df['r2'].iloc[0]))
    print(len(df['c_chunk'].iloc[0]))

dfs = [bullx2_df, bull_df, flat_df, bear_df, bearx2_df]
for item in dfs:
    add_features(item)

# Concatenate all r1 values from all categories (after all dfs have r1)
all_r1 = []
for df in dfs:
    all_r1.extend([item for sublist in df['r1'] for item in sublist])
    
bullx2_df.to_feather("./data/bullx2v7.feather")
bull_df.to_feather("./data/bullv7.feather")
bearx2_df.to_feather("./data/bearx2v7.feather")
bear_df.to_feather("./data/bearv7.feather")
flat_df.to_feather("./data/flatv7.feather")