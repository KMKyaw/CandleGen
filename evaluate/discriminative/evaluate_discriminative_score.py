import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

columns_to_keep = ['o_chunk', 'h_chunk', 'l_chunk', 'c_chunk']

def process_chunk(chunk):
    if isinstance(chunk, str):
        chunk = np.array(eval(chunk))
    return chunk.reshape(-1, 1) if chunk.ndim == 1 else chunk

def pad_or_truncate(chunk, target_length):
    if len(chunk) > target_length:
        return chunk[:target_length]
    elif len(chunk) < target_length:
        return np.pad(chunk, (0, target_length - len(chunk)), mode='constant')
    return chunk

# List of all fake and original file paths
fake_files = [
    # "./synthetic_bear_candles_len10.csv",
    # "./synthetic_bearx2_candles_10.csv",
    # "./synthetic_bull_candles_len10.csv",
    # "./synthetic_bullx2_candles_len10.csv",
    # "./synthetic_flat_candles_len10.csv"

    # "./synthetic_bear_candles_len16.csv",
    # "./synthetic_bearx2_candles_16.csv",
    # "./synthetic_bull_candles_len16.csv",
    # "./synthetic_bullx2_candles_len16.csv",
    # "./synthetic_flat_candles_len16.csv"

    "./synthetic_bear_candles_len30.csv",
    # "./synthetic_bearx2_candles_30.csv",
    # "./synthetic_bull_candles_len30.csv",
    # "./synthetic_bullx2_candles_len30.csv",
    # "./synthetic_flat_candles_len30.csv"
]
original_files = [
    "./bearv7.feather",
    # "./bearx2v7.feather",
    # "./bullv7.feather",
    # "./bullx2v7.feather",
    # "./flatv7.feather"
]

LENGTH = 29

# Print total length of each fake file
for fake_path in fake_files:
    fake_df = pd.read_csv(fake_path)
    print(f"Total length of {fake_path}: {len(fake_df)}")
# Combine all fake and original data
all_fake = []
all_original = []
for fake_path, orig_path in zip(fake_files, original_files):
    fake_df = pd.read_csv(fake_path)[columns_to_keep]
    orig_df = pd.read_feather(orig_path)[columns_to_keep]
    # Filter original data to only rows with o_chunk length LENGTH
    orig_df = orig_df[orig_df['o_chunk'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x) == LENGTH)].reset_index(drop=True)
    # Convert chunks to arrays
    fake_df = fake_df.applymap(process_chunk)
    orig_df = orig_df.applymap(process_chunk)
    # Pad/truncate to consistent length
    target_length = LENGTH
    fake_df = fake_df.applymap(lambda x: pad_or_truncate(process_chunk(x), target_length))
    orig_df = orig_df.applymap(lambda x: pad_or_truncate(process_chunk(x), target_length))
    # Stack rows
    all_fake.append(np.stack([np.hstack(row) for row in fake_df.values]))
    all_original.append(np.stack([np.hstack(row) for row in orig_df.values]))

# Concatenate all samples
all_fake_np = np.concatenate(all_fake, axis=0)
all_original_np = np.concatenate(all_original, axis=0)

print(f"fake_data_np shape: {all_fake_np.shape}")
print(f"original_data_np shape: {all_original_np.shape}")

# Downsize both to the smallest sample count
min_samples = min(all_fake_np.shape[0], all_original_np.shape[0])

fake_idx = np.random.choice(all_fake_np.shape[0], min_samples, replace=False)
orig_idx = np.random.choice(all_original_np.shape[0], min_samples, replace=False)

sampled_fake_np = all_fake_np[fake_idx]
sampled_original_np = all_original_np[orig_idx]

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.gru(x)
        logits = self.fc(h_n[-1])
        output = self.sigmoid(logits)
        return logits, output

def custom_discriminative_score_torch(ori_data, generated_data):
    """Custom implementation of the discriminative score using PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no, seq_len, dim = ori_data.shape
    hidden_dim = dim // 2
    iterations = 2000
    batch_size = 128

    # Convert data to PyTorch tensors
    ori_data = torch.tensor(ori_data, dtype=torch.float32).to(device)
    generated_data = torch.tensor(generated_data, dtype=torch.float32).to(device)

    # Create DataLoader for training and testing
    labels_ori = torch.ones((ori_data.size(0), 1), dtype=torch.float32).to(device)
    labels_gen = torch.zeros((generated_data.size(0), 1), dtype=torch.float32).to(device)

    dataset = TensorDataset(torch.cat([ori_data, generated_data]), torch.cat([labels_ori, labels_gen]))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss, and optimizer
    model = Discriminator(dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    model.train()
    for _ in range(iterations):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            _, preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    acc = accuracy_score(all_labels, (all_preds > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score

# Compute the discriminative score using the PyTorch implementation
score = custom_discriminative_score_torch(sampled_original_np, sampled_fake_np)
print(f"Combined Custom Discriminative Score (PyTorch): {score}")
