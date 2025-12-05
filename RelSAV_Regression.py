"""
Relaxed SAV Method for Regression - Algorithm 4 from the paper
"Efficient and stable SAV-based methods for gradient flows arising from deep learning"
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
dt = 0.1  # Δt
eta = 0.99  # Relaxation parameter η from paper (default 0.99)
train_losses = []
test_losses = []
r = None
isRecord = False

#=============================Wandb Config======================================
if isRecord:
    run = wandb.init(
        entity="pheonizard-university-of-nottingham",
        project="SAV-base-Optimization",
        name="RelSAV-Gaussian-Regression",
        config={
            "C": C,
            "lambda": lambda_,
            "learning_rate": dt,
            "eta": eta,
            "architecture": "[x, 1]->[W, a] with ReLU, m = 100",
            "dataset": "y = exp(-x^2), x in N(0, 0.2)",
            "optimizer": "Relaxed-SAV",
            "epochs": num_epochs,
        },
    )

#=============================Train=============================================
print("Training with Relaxed SAV method (Algorithm 4)...")
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
            # Get current parameters and gradients
            theta_n = flatten_params(model.W, model.a)
            grad_n = flatten_grad(model)

            # Algorithm 4: Step 2 - Obtain θ^(n+1) and r̃^(n+1) by using Algorithm 2 (vanilla SAV)
            inv_operator = 1.0 / (1.0 + dt * lambda_)
            grad_scaled = grad_n * inv_operator
            alpha = dt / math.sqrt(loss.item() + C)
            theta_n_2 = -alpha * grad_scaled

            dot_val = torch.dot(grad_n, grad_scaled)
            denom = 1.0 + dt * dot_val / (2.0 * (loss.item() + C))
            r_tilde_n_plus_1 = r / denom

            theta_n_plus_1 = theta_n + r_tilde_n_plus_1 * theta_n_2

            # Update model to compute I(θ^(n+1))
            W_temp, a_temp = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
            model.W.copy_(W_temp)
            model.a.copy_(a_temp)

        # Algorithm 4: Step 3 - r̂^(n+1) = √(I(θ^(n+1)) + C)
        pred_new = model(X)
        loss_new = criterion(pred_new, Y)
        r_hat_n_plus_1 = math.sqrt(loss_new.item() + C)

        with torch.no_grad():
            # Algorithm 4: Steps 4-7 - Compute relaxation parameter ξ₀
            # Equation (24): ξ₀ = max{0, (-b - √(b²-4ac)) / (2a)}

            a_coef = (r_tilde_n_plus_1 - r_hat_n_plus_1) ** 2
            b_coef = 2 * r_hat_n_plus_1 * (r_tilde_n_plus_1 - r_hat_n_plus_1)

            # Compute ||θ^(n+1) - θ^n||²
            theta_diff_norm_sq = torch.sum((theta_n_plus_1 - theta_n) ** 2).item()
            c_coef = r_hat_n_plus_1**2 - r_tilde_n_plus_1**2 - eta * theta_diff_norm_sq / dt

            # Compute ξ₀
            discriminant = b_coef**2 - 4*a_coef*c_coef
            if discriminant >= 0 and a_coef != 0:
                xi_0 = max(0, (-b_coef - math.sqrt(discriminant)) / (2*a_coef))
            else:
                xi_0 = 0.0

            # Algorithm 4: Step 8 - r^(n+1) = ξ₀r̃^(n+1) + (1-ξ₀)r̂^(n+1)
            r = xi_0 * r_tilde_n_plus_1 + (1 - xi_0) * r_hat_n_plus_1

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
plt.title('Relaxed SAV Method (Algorithm 4)')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x_test.cpu().numpy(), y_test.cpu().numpy(), label='Original Data')
plt.scatter(x_test.cpu().numpy(), y_predict.cpu().numpy(), label='Fitted Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Relaxed SAV Regression Results')
plt.show()

if isRecord:
    run.log({
        "x_test": x_test,
        "y_Test": y_test,
        "y_hat": y_predict
    })
    run.finish()
