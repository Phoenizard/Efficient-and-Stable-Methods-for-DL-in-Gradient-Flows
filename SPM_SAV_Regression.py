"""
Smoothed Particle Method (SPM) with SAV - Algorithm 1 + Algorithm 2 from the paper
"Efficient and stable SAV-based methods for gradient flows arising from deep learning"

SPM uses smooth kernels instead of Dirac delta functions for better accuracy.
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

# SPM parameters
h = 0.0001  # Smoothing parameter (default from paper)
J_samples = 10  # Number of Monte Carlo samples (default from paper)

train_losses = []
test_losses = []
r = None
isRecord = False

#=============================Wandb Config======================================
if isRecord:
    run = wandb.init(
        entity="pheonizard-university-of-nottingham",
        project="SAV-base-Optimization",
        name="SPM-SAV-Gaussian-Regression",
        config={
            "C": C,
            "lambda": lambda_,
            "learning_rate": dt,
            "h": h,
            "J_samples": J_samples,
            "architecture": "[x, 1]->[W, a] with ReLU, m = 100",
            "dataset": "y = exp(-x^2), x in N(0, 0.2)",
            "optimizer": "SPM-SAV (Algorithm 1 + 2)",
            "epochs": num_epochs,
        },
    )

#=============================Train=============================================
print(f"Training with Smoothed Particle Method + SAV (Algorithm 1 + 2)...")
print(f"SPM parameters: h={h}, J={J_samples}")

for epoch in range(num_epochs):
    r = None
    for X, Y in train_loader:
        # Algorithm 1: SPM with Monte Carlo integration
        # Generate {ξⱼ}ⱼ₌₁ᴶ ~ N(0, I_{D+1})
        loss_accumulator = 0.0
        grad_W_accumulator = torch.zeros_like(model.W)
        grad_a_accumulator = torch.zeros_like(model.a)

        # Algorithm 1: Steps 2-6 - Monte Carlo loop
        for j in range(J_samples):
            # Step 3: Generate ξʲ from standard normal
            # For W: shape (D+1, m), for a: shape (m, outputs)
            xi_W = torch.randn_like(model.W) * h  # ξ ~ N(0, h²I)
            xi_a = torch.randn_like(model.a) * h

            # Step 4: Calculate Lⱼ = I(x, θⁿ, ξʲ)
            # Forward pass with perturbed parameters
            W_perturbed = model.W + xi_W
            a_perturbed = model.a + xi_a

            # Compute output with perturbed parameters
            ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
            x_aug = torch.cat([X, ones], dim=1)
            hidden = torch.relu(x_aug @ W_perturbed)
            pred_perturbed = hidden @ a_perturbed

            # Compute loss for this sample
            loss_j = criterion(pred_perturbed, Y)
            loss_accumulator += loss_j.item()

            # Compute gradients w.r.t. perturbed parameters
            loss_j.backward()
            if model.W.grad is not None:
                grad_W_accumulator += model.W.grad
                grad_a_accumulator += model.a.grad
                model.zero_grad()

        # Step 6: Loss = (1/J) Σⱼ Lⱼ
        avg_loss = loss_accumulator / J_samples
        avg_grad_W = grad_W_accumulator / J_samples
        avg_grad_a = grad_a_accumulator / J_samples

        if r is None:
            r = math.sqrt(avg_loss + C)

        # Step 7: Use the Loss to renew θⁿ⁺¹ with SAV scheme
        with torch.no_grad():
            theta_n = flatten_params(model.W, model.a)
            grad_n = torch.cat([avg_grad_W.view(-1), avg_grad_a.view(-1)])

            # SAV update (Algorithm 2)
            inv_operator = 1.0 / (1.0 + dt * lambda_)
            grad_scaled = grad_n * inv_operator

            alpha = dt / math.sqrt(avg_loss + C)
            theta_n_2 = -alpha * grad_scaled

            dot_val = torch.dot(grad_n, grad_scaled)
            denom = 1.0 + dt * dot_val / (2.0 * (avg_loss + C))
            r = r / denom

            theta_n_plus_1 = theta_n + r * theta_n_2

            W_new, a_new = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
            model.W.copy_(W_new)
            model.a.copy_(a_new)

    # Evaluate on full datasets
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
plt.title('SPM with SAV Method (Algorithm 1 + 2)')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(x_test.cpu().numpy(), y_test.cpu().numpy(), label='Original Data')
plt.scatter(x_test.cpu().numpy(), y_predict.cpu().numpy(), label='Fitted Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('SPM-SAV Regression Results')
plt.show()

if isRecord:
    run.log({
        "x_test": x_test,
        "y_Test": y_test,
        "y_hat": y_predict
    })
    run.finish()

print("\nSPM Training complete!")
print(f"Final Train Loss: {train_losses[-1]:.8f}")
print(f"Final Test Loss: {test_losses[-1]:.8f}")
