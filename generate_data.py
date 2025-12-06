"""
Generate data for all experiments.
"""

import torch
import numpy as np
import os

np.random.seed(0)
torch.manual_seed(0)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

print("Generating data for experiments...")

# Example 1: Sin + Cos function (20D and 40D versions)
print("\n1. Generating Example 1 data (sin + cos)...")
D = 40
M = 10000
p = np.random.randn(D)
q = np.random.randn(D)

x_data = np.random.uniform(0, 1, (M, D))
y_data = np.sin(x_data @ p) + np.cos(x_data @ q)
y_data = y_data.reshape(-1, 1)

# Split into train and test
split_idx = int(0.8 * M)
x_train_1 = torch.tensor(x_data[:split_idx], dtype=torch.float32)
y_train_1 = torch.tensor(y_data[:split_idx], dtype=torch.float32)
x_test_1 = torch.tensor(x_data[split_idx:], dtype=torch.float32)
y_test_1 = torch.tensor(y_data[split_idx:], dtype=torch.float32)

# Save
torch.save((x_train_1, y_train_1), 'data/Example1_train_data.pt')
torch.save((x_test_1, y_test_1), 'data/Example1_test_data.pt')
print(f"   Saved Example 1 data: {M} samples, {D} dimensions")

# Example 2: Quadratic function
print("\n2. Generating Example 2 data (quadratic)...")
D = 40
M = 10000
c = np.random.randn(D)

x_data = np.random.uniform(0, 5, (M, D))
y_data = np.sum(c * x_data**2, axis=1, keepdims=True)

# Split into train and test
split_idx = int(0.8 * M)
x_train_2 = torch.tensor(x_data[:split_idx], dtype=torch.float32)
y_train_2 = torch.tensor(y_data[:split_idx], dtype=torch.float32)
x_test_2 = torch.tensor(x_data[split_idx:], dtype=torch.float32)
y_test_2 = torch.tensor(y_data[split_idx:], dtype=torch.float32)

# Save
torch.save((x_train_2, y_train_2), 'data/Example2_train_data.pt')
torch.save((x_test_2, y_test_2), 'data/Example2_test_data.pt')
print(f"   Saved Example 2 data: {M} samples, {D} dimensions")

# Example 3: Gaussian function
print("\n3. Generating Example 3 data (Gaussian)...")
D = 40
M = 100000

# Use non-uniform distribution as mentioned in paper
x_data = np.random.normal(0, 0.2, (M, D))
y_data = np.exp(-10 * np.sum(x_data**2, axis=1, keepdims=True))

# Split into train and test
split_idx = int(0.8 * M)
x_train_3 = torch.tensor(x_data[:split_idx], dtype=torch.float32)
y_train_3 = torch.tensor(y_data[:split_idx], dtype=torch.float32)
x_test_3 = torch.tensor(x_data[split_idx:], dtype=torch.float32)
y_test_3 = torch.tensor(y_data[split_idx:], dtype=torch.float32)

# Save
torch.save((x_train_3, y_train_3), 'data/Example3_train_data.pt')
torch.save((x_test_3, y_test_3), 'data/Example3_test_data.pt')
print(f"   Saved Example 3 data: {M} samples, {D} dimensions")

# Also generate simple 1D Gaussian data for quick testing
print("\n4. Generating Gaussian 1D data (for compatibility)...")
x_data_1d = np.random.normal(0, 0.2, (1000, 1))
y_data_1d = np.exp(-x_data_1d**2)

split_idx = int(0.8 * 1000)
x_train_gauss = torch.tensor(x_data_1d[:split_idx], dtype=torch.float32)
y_train_gauss = torch.tensor(y_data_1d[:split_idx], dtype=torch.float32)
x_test_gauss = torch.tensor(x_data_1d[split_idx:], dtype=torch.float32)
y_test_gauss = torch.tensor(y_data_1d[split_idx:], dtype=torch.float32)

torch.save((x_train_gauss, y_train_gauss), 'data/Gaussian_train_data.pt')
torch.save((x_test_gauss, y_test_gauss), 'data/Gaussian_test_data.pt')
print(f"   Saved Gaussian 1D data: 1000 samples")

print("\n✓ All data generated successfully!")
print("\nData files created:")
print("  - data/Example1_train_data.pt, data/Example1_test_data.pt")
print("  - data/Example2_train_data.pt, data/Example2_test_data.pt")
print("  - data/Example3_train_data.pt, data/Example3_test_data.pt")
print("  - data/Gaussian_train_data.pt, data/Gaussian_test_data.pt")
