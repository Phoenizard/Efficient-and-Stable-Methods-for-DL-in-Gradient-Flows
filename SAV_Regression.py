from model import LinearModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import matplotlib.pyplot as plt
import numpy as np
from utilize import flatten_params, unflatten_params, flatten_grad
np.random.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else \
         ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using {device}")
#=============================Load Data=========================================
(x_train, y_train) = torch.load('data/Gaussian_train_data.pt')
(x_test, y_test) = torch.load('data/Gaussian_test_data.pt')

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#=============================Train Config======================================
m = 100 # Number of neurons
model = LinearModel.SinCosModel(m=m)
model.to(device)
criterion = nn.MSELoss()
num_epochs = 50000
C = 100
lambda_ = 4
dt = 0.1 # Δt
train_losses = []
test_losses = []
# Initialize auxiliary variable r with initial loss on full training data
with torch.no_grad():
    initial_loss = criterion(model(x_train), y_train).item()
    r = math.sqrt(initial_loss + C)
    print(f"Initial loss: {initial_loss:.8f}, Initial r: {r:.8f}")
#=============================Train=============================================
for epoch in range(num_epochs):
    for X, Y in train_loader:
        pred = model(X)
        loss = criterion(pred, Y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            theta_n = flatten_params(model.W, model.a)
            grad_n = flatten_grad(model)
            inv_operator = 1.0 / (1.0 + dt * lambda_)
            grad_scaled = grad_n * inv_operator
            
            alpha = dt / math.sqrt(loss.item() + C)
            theta_n_2 = - alpha * grad_scaled

            dot_val = torch.dot(grad_n, grad_scaled)
            denom = 1.0 + dt * dot_val / (2.0 * (loss.item() + C))
            r = r / denom
            # \theta^{n+1} = \theta^{n+1,1} + r^{n+1} * \theta^{n+1,2}
            theta_n_plus_1 = theta_n + r * theta_n_2

            W_new, a_new = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
            model.W.copy_(W_new)
            model.a.copy_(a_new)
    with torch.no_grad():
        model.eval()
        train_loss = criterion(model(x_train), y_train).item()
        test_loss = criterion(model(x_test), y_test).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
#=============================Test==============================================
model.eval()
with torch.no_grad():
    y_predict = model(x_test)
#=============================Plot==============================================
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x_test.numpy(), y_test.numpy(), label='Original Data')

plt.scatter(x_test.numpy(), y_predict.numpy(), label='Fitted Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()