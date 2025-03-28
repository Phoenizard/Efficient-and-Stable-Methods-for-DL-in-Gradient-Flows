from model import LinearModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import wandb
from utilize import flatten_params, unflatten_params, flatten_grad, compute_jacobian, compute_jacobian_C
np.random.seed(0)
torch.manual_seed(0)
#=============================Load Data=========================================
(x_train, y_train) = torch.load('data/Gaussian_train_data.pt')
(x_test, y_test) = torch.load('data/Gaussian_test_data.pt')
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#=============================Train Config======================================
model = LinearModel.SinCosModel(m=100)
criterion = nn.MSELoss()
num_epochs = 50000
dt = 0.1 # Δt
train_losses = []
test_losses = []
isRecord = True
#=============================Wandb Config======================================
if isRecord:
    run = wandb.init(
        entity="pheonizard-university-of-nottingham",
        project="SAV-base-Optimization",
        name="1D-IEQ-Gaussian-Mar27",
        config={
            "learning_rate": dt,
            "architecture": "[x, 1]->[W, a] with ReLU, m = 100",
            "dataset": "y = exp(-x^2), x in N(0, 0.2)",
            "optimizer": "IEQ",
            "epochs": num_epochs,
        },
    )
#=============================Train=============================================
for epoch in range(num_epochs):
    for X, Y in train_loader:
        theta_n = flatten_params(model.W, model.a)
        theta_n = theta_n.clone().detach().requires_grad_(True)
        
        f_theta = model(X).squeeze(-1)  # 模型输出，形状: (batch_size,)

        q_n = f_theta - Y.squeeze(-1)  # 形状: (batch_size,)

        # 计算雅各比矩阵 J_f(theta)，形状为 (batch_size, num_parameters)
        J = compute_jacobian(theta_n, X, model)
        # 离散更新辅助变量： q^{n+1} = (I + dt * (J J^T))^{-1} * q^n
        batch_size = X.size(0)
        I = torch.eye(batch_size, device=X.device, dtype=X.dtype)
        JJt = J @ J.t()  # 形状: (batch_size, batch_size)
        A_mat = I + dt * JJt
        # 解线性系统 A_mat * q_np1 = q_n
        q_np1 = torch.linalg.solve(A_mat, q_n)
        
        theta_np1 = theta_n - dt * (J.t() @ q_np1)
        
        with torch.no_grad():
            W_new, a_new = unflatten_params(theta_np1, model.W.shape, model.a.shape)
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
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
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
model.eval()
with torch.no_grad():
    y_predict = model(x_test)
plt.scatter(x_test.numpy(), y_predict.numpy(), label='Fitted Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

if isRecord:
    run.log({
        "x_test": x_test,
        "y_Test": y_test,
        "y_hat": y_predict
        })
    run.finish()