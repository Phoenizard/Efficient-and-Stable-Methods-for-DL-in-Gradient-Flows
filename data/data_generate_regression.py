"""
Generate regression data for Experiments 1, 2, and 3.

Running this script will generate:
- experiment_1_train_data.pt, experiment_1_test_data.pt (Sin + Cos function)
- experiment_2_train_data.pt, experiment_2_test_data.pt (Quadratic function)
- experiment_3_train_data.pt, experiment_3_test_data.pt (Gaussian function)
"""

import torch
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("Generating Regression Data for All Experiments")
print("="*60)

# ============================================================
# Experiment 1: Sin + Cos function
# Target: f*(x) = sin(Σ p_i x_i) + cos(Σ q_i x_i)
# ============================================================
print("\n1. Generating Experiment 1 data (Sin + Cos function)...")
D = 40  # Dimension
M = 10000  # Number of samples
p = np.random.randn(D)
q = np.random.randn(D)

x_data = np.random.uniform(0, 1, (M, D))
y_data = np.sin(x_data @ p) + np.cos(x_data @ q)
y_data = y_data.reshape(-1, 1)

# Split into train (80%) and test (20%)
split_idx = int(0.8 * M)
x_train_1 = torch.tensor(x_data[:split_idx], dtype=torch.float32)
y_train_1 = torch.tensor(y_data[:split_idx], dtype=torch.float32)
x_test_1 = torch.tensor(x_data[split_idx:], dtype=torch.float32)
y_test_1 = torch.tensor(y_data[split_idx:], dtype=torch.float32)

# Save
torch.save((x_train_1, y_train_1), os.path.join(script_dir, 'experiment_1_train_data.pt'))
torch.save((x_test_1, y_test_1), os.path.join(script_dir, 'experiment_1_test_data.pt'))
print(f"   ✓ Saved: {M} samples, {D} dimensions")
print(f"     Train: {x_train_1.shape[0]} samples")
print(f"     Test:  {x_test_1.shape[0]} samples")

# ============================================================
# Experiment 2: Quadratic function
# Target: f*(x) = Σ c_i x_i^2
# ============================================================
print("\n2. Generating Experiment 2 data (Quadratic function)...")
D = 40  # Dimension
M = 10000  # Number of samples
c = np.random.randn(D)

x_data = np.random.uniform(0, 5, (M, D))
y_data = np.sum(c * x_data**2, axis=1, keepdims=True)

# Split into train (80%) and test (20%)
split_idx = int(0.8 * M)
x_train_2 = torch.tensor(x_data[:split_idx], dtype=torch.float32)
y_train_2 = torch.tensor(y_data[:split_idx], dtype=torch.float32)
x_test_2 = torch.tensor(x_data[split_idx:], dtype=torch.float32)
y_test_2 = torch.tensor(y_data[split_idx:], dtype=torch.float32)

# Save
torch.save((x_train_2, y_train_2), os.path.join(script_dir, 'experiment_2_train_data.pt'))
torch.save((x_test_2, y_test_2), os.path.join(script_dir, 'experiment_2_test_data.pt'))
print(f"   ✓ Saved: {M} samples, {D} dimensions")
print(f"     Train: {x_train_2.shape[0]} samples")
print(f"     Test:  {x_test_2.shape[0]} samples")

# ============================================================
# Experiment 3: Gaussian function
# Target: f*(x) = exp(-10||x||²)
# ============================================================
print("\n3. Generating Experiment 3 data (Gaussian function)...")
D = 40  # Dimension
M = 1000  # Larger dataset for this challenging problem

# Use normal distribution as mentioned in paper
x_data = np.random.normal(0, 0.2, (M, D))
y_data = np.exp(-10 * np.sum(x_data**2, axis=1, keepdims=True))

# Split into train (80%) and test (20%)
split_idx = int(0.8 * M)
x_train_3 = torch.tensor(x_data[:split_idx], dtype=torch.float32)
y_train_3 = torch.tensor(y_data[:split_idx], dtype=torch.float32)
x_test_3 = torch.tensor(x_data[split_idx:], dtype=torch.float32)
y_test_3 = torch.tensor(y_data[split_idx:], dtype=torch.float32)

# Save
torch.save((x_train_3, y_train_3), os.path.join(script_dir, 'experiment_3_train_data.pt'))
torch.save((x_test_3, y_test_3), os.path.join(script_dir, 'experiment_3_test_data.pt'))
print(f"   ✓ Saved: {M} samples, {D} dimensions")
print(f"     Train: {x_train_3.shape[0]} samples")
print(f"     Test:  {x_test_3.shape[0]} samples")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("✓ All regression data generated successfully!")
print("="*60)
print("\nData files created in data/ folder:")
print("  - experiment_1_train_data.pt, experiment_1_test_data.pt")
print("  - experiment_2_train_data.pt, experiment_2_test_data.pt")
print("  - experiment_3_train_data.pt, experiment_3_test_data.pt")
print("\nYou can now run the experiment scripts:")
print("  - experiment_regression_1.py")
print("  - experiment_regression_2.py")
print("  - experiment_regression_3.py")
print("="*60)
