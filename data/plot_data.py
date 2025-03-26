import numpy as np
import torch
import matplotlib.pyplot as plt

# Load data
(x_train, y_train) = torch.load('data/Gaussian_train_data.pt')
(x_test, y_test) = torch.load('data/Gaussian_test_data.pt')

# Plot data
plt.figure()
plt.plot(x_train, y_train, 'o', label='Training data')
plt.plot(x_test, y_test, 'x', label='Test data')
plt.legend()
plt.show()
