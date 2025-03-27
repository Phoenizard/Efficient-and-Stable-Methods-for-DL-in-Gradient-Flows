from model import LinearModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import wandb
np.random.seed(0)
torch.manual_seed(0)
#=============================Load Data=========================================
(x_train, y_train) = torch.load('data/MNIST_train_data.pt')
(x_test, y_test) = torch.load('data/MNIST_test_data.pt')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#=============================Train Config======================================
m = 1000 # Number of neurons
# model = LinearModel.SinCosModel(m=m, outputs=10)
model = LinearModel.ClassificationModel(m=m, inputs=784, outputs=10)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 1000
isRecord = False # Change to True if you want to record the training process
train_losses = []
test_losses = []
#=============================Wandb Config======================================
if isRecord:
    run = wandb.init(
        entity="pheonizard-university-of-nottingham",
        project="SAV-base-Optimization",
        name="SGD-MNIST-Mar26",
        config={
            "learning_rate": learning_rate,
            "architecture": f"[x, 784]->[W, a] with ReLU, m = {m}",
            "dataset": "MNIST",
            "optimizer": "SGD",
            "epochs": num_epochs,
        },
    )
#=============================Train=============================================
for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_x.size(0)
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)
        if isRecord:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss})
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
#=============================Plot==============================================
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
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

if isRecord:
    run.log({
        "x_test": x_test,
        "y_Test": y_test,
        "y_hat": y_predict
        })
    run.finish()
