"""
Experiment: Classification Task (MNIST)
Dataset: MNIST handwritten digits (0-9)

This experiment compares different optimization algorithms for classification:
- SGD
- SAV
- ExpSAV
- IEQ Adaptive
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from algorithms import (sgd_classification,sav_classification,
                        esav_classification, ieq_adaptive_classification, ieq_classification)

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

# Create results directory
os.makedirs('results/experiment_mnist', exist_ok=True)

print("="*60)
print("Experiment: Classification Task (MNIST)")
print("="*60)

# Load data
print("\nLoading MNIST data...")
try:
    (x_train, y_train) = torch.load('data/MNIST_train_data.pt')
    (x_test, y_test) = torch.load('data/MNIST_test_data.pt')
except:
    print("Warning: MNIST data not found in data/ directory")
    print("Trying alternative path: data/MNIST/data/")
    (x_train, y_train) = torch.load('data/MNIST/data/MNIST_train_data.pt')
    (x_test, y_test) = torch.load('data/MNIST/data/MNIST_test_data.pt')

print(f"Data loaded: Train={x_train.shape[0]}, Test={x_test.shape[0]}")
print(f"Image size: {x_train.shape[1]} features")

# Configuration
device = 'cuda' if torch.cuda.is_available() else \
         ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
m = 100  # Number of neurons
inputs = 784  # 28x28 images flattened
outputs = 10  # 10 classes
num_epochs = 50

# Store all results
results = {}

# Run experiments
print("\n" + "="*60)
print("Running SGD...")
print("="*60)
hist_sgd = sgd_classification(x_train, y_train, x_test, y_test,
                              m=m, batch_size=256, learning_rate=0.1,
                              num_epochs=num_epochs, inputs=inputs, outputs=outputs,
                              device=device)
results['SGD'] = hist_sgd

print("\n" + "="*60)
print("Running SAV...")
print("="*60)
hist_sav = sav_classification(x_train, y_train, x_test, y_test,
                              m=m, batch_size=256, C=100, lambda_=4, dt=0.1,
                              num_epochs=num_epochs, inputs=inputs, outputs=outputs,
                              device=device)
results['SAV'] = hist_sav

print("\n" + "="*60)
print("Running ExpSAV...")
print("="*60)
hist_esav = esav_classification(x_train, y_train, x_test, y_test,
                                m=m, batch_size=256, C=1, lambda_=0, dt=0.1,
                                num_epochs=num_epochs, inputs=inputs, outputs=outputs,
                                device=device)
results['ExpSAV'] = hist_esav


print("\n" + "="*60)
print("Running IEQ...")
print("="*60)
hist_ieq = ieq_classification(x_train, y_train, x_test, y_test,
                              m=m, batch_size=256, dt=0.1,
                              num_epochs=num_epochs, inputs=inputs, outputs=outputs,
                              device=device)
results['IEQ'] = hist_ieq


print("\n" + "="*60)
print("Running IEQ Adaptive...")
print("="*60)
hist_ieq_adaptive = ieq_adaptive_classification(x_train, y_train, x_test, y_test,
                                                m=m, batch_size=256, dt=0.1,
                                                num_epochs=num_epochs, inputs=inputs,
                                                outputs=outputs, device=device)
results['IEQ_Adaptive'] = hist_ieq_adaptive

# Save results
print("\n" + "="*60)
print("Saving results...")
print("="*60)
torch.save(results, 'results/experiment_mnist/results.pt')
print("Results saved to: results/experiment_mnist/results.pt")

# Plot results
print("\nPlotting results...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot train loss
for name, hist in results.items():
    ax1.plot(hist['train_loss'], label=name, marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss')
ax1.set_yscale('log')
ax1.set_title('Training Loss Comparison (MNIST)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot test loss
for name, hist in results.items():
    ax2.plot(hist['test_loss'], label=name, marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Loss')
ax2.set_yscale('log')
ax2.set_title('Test Loss Comparison (MNIST)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot test accuracy
for name, hist in results.items():
    ax3.plot(hist['test_accuracy'], label=name, marker='o')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Test Accuracy (%)')
ax3.set_title('Test Accuracy Comparison (MNIST)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/experiment_mnist/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("Metrics comparison plot saved to: results/experiment_mnist/metrics_comparison.png")

# Print final results
print("\n" + "="*60)
print("Final Results:")
print("="*60)
print(f"{'Algorithm':<15} {'Test Loss':<12} {'Test Accuracy':<12}")
print("-" * 60)
for name, hist in results.items():
    final_loss = hist['test_loss'][-1]
    final_acc = hist['test_accuracy'][-1]
    print(f"{name:<15} {final_loss:<12.6f} {final_acc:<12.2f}%")

print("\n" + "="*60)
print("Experiment completed successfully!")
print("="*60)
