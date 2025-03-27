from model import LinearModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import matplotlib.pyplot as plt
import numpy as np
import wandb
from utilize import flatten_params, unflatten_params, flatten_grad
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
#=============================Load Data=========================================
(x_train, y_train) = torch.load('data/MNIST_train_data.pt')
(x_test, y_test) = torch.load('data/MNIST_test_data.pt')

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
# model = LinearModel.SinCosModel(m=m, outputs=10)
model = LinearModel.ClassificationModel(m=m, inputs=784, outputs=10)
model.to(device)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
num_epochs = 100
C = 100
lambda_ = 4
dt = 0.1 # Δt
train_losses = []
test_losses = []
r = None
isRecord = False
#=============================Wandb Config======================================
if isRecord:
    run = wandb.init(
        entity="pheonizard-university-of-nottingham",
        project="SAV-base-Optimization",
        name="1D-SAV-Gaussian-Mar26",
        config={
            "C": C,
            "lambda": lambda_,
            "learning_rate": dt,
            "architecture": "[x, 1]->[W, a] with ReLU, m = 100",
            "dataset": "y = exp(-x^2), x in N(0, 0.2)",
            "optimizer": "SAV",
            "epochs": num_epochs,
        },
    )
#=============================Train=============================================
for epoch in range(num_epochs):
    r = None
    for X, Y in train_loader:
        pred = model(X)
        loss = criterion(pred, Y)  
        if r is None:
            r = math.sqrt(loss.item() + C)
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
        if isRecord:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss})
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
#=============================Test==============================================
# Test the accuracy for classification
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
if isRecord:
    run.log({"accuracy": 100 * correct / total})

#=============================Plot==============================================
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.show()

# plt.figure(figsize=(8, 6))
# plt.scatter(x_test.numpy(), y_test.numpy(), label='Original Data')
# model.eval()
# with torch.no_grad():
#     y_predict = model(x_test)
# plt.scatter(x_test.numpy(), y_predict.numpy(), label='Fitted Data')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# if isRecord:
#     run.log({
#         "x_test": x_test,
#         "y_Test": y_test,
#         "y_hat": y_predict
#         })
#     run.finish()