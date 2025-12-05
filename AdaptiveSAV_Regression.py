"""
Adaptive SAV Method for Regression - Algorithm 5 from the paper
"Efficient and stable SAV-based methods for gradient flows arising from deep learning"
Combines SAV with Adam's adaptive learning rate strategy
"""
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
m = 100  # Number of neurons
model = LinearModel.SinCosModel(m=m, outputs=10)
model.to(device)
criterion = nn.MSELoss()
num_epochs = 50000
C = 100
lambda_ = 4
dt = 0.1  # Initial Δt (initial learning rate)

# Adam parameters from Algorithm 5
epsilon = 1e-8
beta1 = 0.9
beta2 = 0.999
m_adam = 0  # First moment (momentum)
v_adam = 0  # Second moment (variance)

train_losses = []
test_losses = []
r = None
isRecord = False

#=============================Wandb Config======================================
if isRecord:
    run = wandb.init(
        entity="pheonizard-university-of-nottingham",
        project="SAV-base-Optimization",
        name="AdaptiveSAV-Gaussian-Regression",
        config={
            "C": C,
            "lambda": lambda_,
            "initial_learning_rate": dt,
            "beta1": beta1,
            "beta2": beta2,
            "architecture": "[x, 1]->[W, a] with ReLU, m = 100",
            "dataset": "y = exp(-x^2), x in N(0, 0.2)",
            "optimizer": "Adaptive-SAV (Algorithm 5)",
            "epochs": num_epochs,
        },
    )

#=============================Train=============================================
print("Training with Adaptive SAV method (Algorithm 5)...")
for epoch in range(num_epochs):
    r = None
    n_step = 0
    for X, Y in train_loader:
        n_step += 1
        pred = model(X)
        loss = criterion(pred, Y)

        if r is None:
            r = math.sqrt(loss.item() + C)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Get current parameters and gradients
            theta_n = flatten_params(model.W, model.a)
            grad_n = flatten_grad(model)

            # Algorithm 5: Steps 2-5 - Adam adaptive strategy
            # Step 2: m^(n+1) = β₁m^n + (1-β₁)∇I(θ^n)
            if isinstance(m_adam, int):
                m_adam = torch.zeros_like(grad_n)
                v_adam = torch.zeros_like(grad_n)

            m_adam = beta1 * m_adam + (1 - beta1) * grad_n

            # Step 3: v^(n+1) = β₂v^n + (1-β₂)||∇I(θ^n)||²
            v_adam = beta2 * v_adam + (1 - beta2) * (grad_n ** 2)

            # Step 4: m̂^(n+1) = m^(n+1) / (1 - β₁^(n+1))
            m_hat = m_adam / (1 - beta1 ** n_step)

            # Step 5: v̂^(n+1) = v^(n+1) / (1 - β₂^(n+1))
            v_hat = v_adam / (1 - beta2 ** n_step)

            # Step 6: N̂'(θ^n) = m̂^(n+1) (modified gradient)
            grad_modified = m_hat

            # Step 7: Δ̂t = Δt / √(v̂^(n+1) + ε) (adaptive time step)
            dt_adaptive = dt / torch.sqrt(v_hat + epsilon)

            # Step 8: Update θ^(n+1) using Algorithm 2, 3, or 4
            # Here we use vanilla SAV (Algorithm 2) with adaptive terms
            inv_operator = 1.0 / (1.0 + dt_adaptive * lambda_)
            grad_scaled = grad_modified * inv_operator

            alpha = dt_adaptive / math.sqrt(loss.item() + C)
            theta_n_2 = -alpha * grad_scaled

            dot_val = torch.dot(grad_modified, grad_scaled)
            denom = 1.0 + torch.mean(dt_adaptive * dot_val / (2.0 * (loss.item() + C)))
            r = r / denom.item()

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
plt.title('Adaptive SAV Method (Algorithm 5)')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x_test.cpu().numpy(), y_test.cpu().numpy(), label='Original Data')
plt.scatter(x_test.cpu().numpy(), y_predict.cpu().numpy(), label='Fitted Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Adaptive SAV Regression Results')
plt.show()

if isRecord:
    run.log({
        "x_test": x_test,
        "y_Test": y_test,
        "y_hat": y_predict
    })
    run.finish()
