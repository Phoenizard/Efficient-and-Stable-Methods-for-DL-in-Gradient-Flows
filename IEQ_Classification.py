from model import LinearModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from utilize import flatten_params, unflatten_params, flatten_grad, compute_jacobian_C
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

# Y 修改为one-hot编码
y_train = nn.functional.one_hot(y_train, num_classes=10)
y_test = nn.functional.one_hot(y_test, num_classes=10)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#=============================Train Config======================================
m = 100 # Number of neurons
outputs = 10
model = LinearModel.ClassificationModel(m=m, inputs=784, outputs=outputs)
model.to(device)
criterion = nn.CrossEntropyLoss()
num_epochs = 100
dt = 0.1 # Δt
train_losses = []
test_losses = []
#=============================Train=============================================
for epoch in range(num_epochs):
    for X, Y in train_loader:
        # 为了计算雅各比需要开启梯度计算
        theta_n = flatten_params(model.W, model.a)
        theta_n = theta_n.clone().detach().requires_grad_(True)
        
        # 使用 model(X) 计算预测输出 f(theta)，形状: (batch_size, k)
        f_theta = model(X)  # shape: (batch_size, k)
        # 定义辅助变量 q = f(theta) - Y，并展平为 (batch_size*k,)
        q_n = (f_theta - Y).reshape(-1)
        
        # 计算雅各比矩阵 J，形状为 (batch_size*k, num_parameters)
        J = compute_jacobian_C(theta_n, X, model)
        # 离散更新辅助变量： q^{n+1} = (I + dt * (J J^T))^{-1} * q^n
        batch_size, _ = f_theta.shape  # batch_size 和 k 已知
        total_dim = batch_size * outputs
        I = torch.eye(total_dim, device=X.device, dtype=X.dtype)
        JJt = J @ J.t()  # shape: (total_dim, total_dim)
        A_mat = I + dt * JJt
        q_np1 = torch.linalg.solve(A_mat, q_n)
        
        # 离散更新参数： theta^{n+1} = theta^n - dt * J^T * q^{n+1}
        theta_np1 = theta_n - dt * (J.t() @ q_np1)
        
        with torch.no_grad():
            W_new, a_new = unflatten_params(theta_np1, model.W.shape, model.a.shape)
            model.W.copy_(W_new)
            model.a.copy_(a_new)
    model.eval()
    test_loss = 0.0
    accuracy = 0
    total = 0
    with torch.no_grad():
        train_loss = criterion(model(x_train), y_train).item()
        train_losses.append(train_loss)

        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            accuracy += (predicted == batch_y).sum().item()
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test Accuracy: {100 * accuracy / total}%")
#=============================Plot==============================================
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.show()

