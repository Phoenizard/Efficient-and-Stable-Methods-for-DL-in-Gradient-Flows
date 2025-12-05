from model import LinearModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from utilize import flatten_params, unflatten_params, flatten_grad
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
model = LinearModel.SinCosModel(m=100)
model.to(device)
criterion = nn.MSELoss()
num_epochs = 50000
dt = 0.1 # Δt
epsilon = 1e-8 # Regularization parameter
train_losses = []
test_losses = []
q = None
#=============================Train=============================================
for epoch in range(num_epochs):
    for X, Y in train_loader:
        pred = model(X)
        loss = criterion(pred, Y)

        # Initialize auxiliary variable q = f(w) - y
        if q is None:
            q = pred.detach() - Y

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            theta_n = flatten_params(model.W, model.a)
            grad_n = flatten_grad(model)

            # Compute adaptive scaling factor α^n
            grad_norm_sq = torch.norm(grad_n) ** 2
            q_norm_sq = torch.norm(q) ** 2
            alpha = 1.0 / (1.0 + dt * grad_norm_sq / (q_norm_sq + epsilon))

            # Update parameters: w^{n+1} = w^n - Δt α^n ∇L(w^n)
            theta_n_plus_1 = theta_n - dt * alpha * grad_n

            W_new, a_new = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
            model.W.copy_(W_new)
            model.a.copy_(a_new)

            # Update auxiliary variable: q^{n+1} = q^n / (1 + Δt ||∇L||^2 / (||q||^2 + ε))
            q = q / (1.0 + dt * grad_norm_sq / (q_norm_sq + epsilon))

    with torch.no_grad():
        model.eval()
        train_loss = criterion(model(x_train), y_train).item()
        test_loss = criterion(model(x_test), y_test).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)    if (epoch + 1) % 1000 == 0:
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
plt.scatter(x_test.cpu().numpy(), y_test.cpu().numpy(), label='Original Data')
plt.scatter(x_test.cpu().numpy(), y_predict.cpu().numpy(), label='Fitted Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

