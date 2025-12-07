"""
Experiment: Regression Task 1
Target function: f*(x) = sin(Σ p_i x_i) + cos(Σ q_i x_i)

This experiment compares different optimization algorithms for regression:
- SGD
- Adam
- SAV
- ExpSAV
- IEQ
- IEQ Adaptive
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from algorithms import (sgd_regression, adam_regression, sav_regression,
                        esav_regression, ieq_regression, ieq_adaptive_regression)

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

# Create results directory
os.makedirs('results/experiment_1', exist_ok=True)

print("="*60)
print("Experiment: Regression Task 1 (Sin + Cos Function)")
print("="*60)

# Load data
print("\nLoading data...")
(x_train, y_train) = torch.load('data/experiment_1_train_data.pt')
(x_test, y_test) = torch.load('data/experiment_1_train_data.pt')
print(f"Data loaded: Train={x_train.shape[0]}, Test={x_test.shape[0]}, Dim={x_train.shape[1]}")

# Configuration
device = 'cuda' if torch.cuda.is_available() else \
         ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
m = 1000  # Number of neurons
num_epochs_standard = 1000
num_epochs_sav = 1000

# Store all results
results = {}

# Run experiments
print("\n" + "="*60)
print("Running SGD...")
print("="*60)
hist_sgd = sgd_regression(x_train, y_train, x_test, y_test,
                          m=m, batch_size=256, learning_rate=0.01,
                          num_epochs=num_epochs_standard, device=device)
results['SGD'] = hist_sgd

print("\n" + "="*60)
print("Running Adam...")
print("="*60)
hist_adam = adam_regression(x_train, y_train, x_test, y_test,
                            m=m, batch_size=64, learning_rate=0.001,
                            num_epochs=num_epochs_standard, device=device)
results['Adam'] = hist_adam

print("\n" + "="*60)
print("Running SAV...")
print("="*60)
hist_sav = sav_regression(x_train, y_train, x_test, y_test,
                          m=m, batch_size=256, C=100, lambda_=4, dt=0.5,
                          num_epochs=num_epochs_standard, device=device)
results['SAV'] = hist_sav

print("\n" + "="*60)
print("Running ExpSAV...")
print("="*60)
hist_esav = esav_regression(x_train, y_train, x_test, y_test,
                            m=m, batch_size=256, C=1, lambda_=1, dt=0.1,
                            num_epochs=num_epochs_standard, device=device)
results['ExpSAV'] = hist_esav

print("\n" + "="*60)
print("Running IEQ (Full Jacobian)...")
print("="*60)
hist_ieq = ieq_regression(x_train, y_train, x_test, y_test,
                          m=m, batch_size=64, dt=0.1,
                          num_epochs=num_epochs_standard, device=device)
results['IEQ'] = hist_ieq

print("\n" + "="*60)
print("Running IEQ Adaptive...")
print("="*60)
hist_ieq_adaptive = ieq_adaptive_regression(x_train, y_train, x_test, y_test,
                                            m=m, batch_size=256, dt=0.1,
                                            num_epochs=num_epochs_standard, device=device)
results['IEQ_Adaptive'] = hist_ieq_adaptive

# Save results
print("\n" + "="*60)
print("Saving results...")
print("="*60)
torch.save(results, 'results/experiment_1/results.pt')
print("Results saved to: results/experiment_1/results.pt")

# Plot results
print("\nPlotting results...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot train loss
for name, hist in results.items():
    ax1.plot(hist['train_loss'], label=name)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss')
ax1.set_yscale('log')
ax1.set_title('Training Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot test loss
for name, hist in results.items():
    ax2.plot(hist['test_loss'], label=name)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Loss')
ax2.set_yscale('log')
ax2.set_title('Test Loss Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/experiment_1/loss_comparison.png', dpi=300, bbox_inches='tight')
print("Loss comparison plot saved to: results/experiment_1/loss_comparison.png")

# Print final results
print("\n" + "="*60)
print("Final Test Loss Results:")
print("="*60)
for name, hist in results.items():
    final_loss = hist['test_loss'][-1]
    print(f"{name:15s}: {final_loss:.6e}")

print("\n" + "="*60)
print("Experiment completed successfully!")
print("="*60)
