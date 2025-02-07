import torch
import torch.nn as nn
import torch.optim as optim
from model.LinearRegressionModel import LinearRegressionModel

from data_loader import train_loader, x_test, y_test
#================================================================================================
learning_rate = 0.2
epoch_num = 10000
isTest = False
#================================================================================================
model = LinearRegressionModel()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
epochs = epoch_num
#===================================Training Loop=======================================
for epoch in range(epochs):
    for inputs, targets in train_loader:
        model.train()
        optimizer.zero_grad() 
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
#===================================Model Saving=======================================
torch.save(model.state_dict(), 'model_SGD.pth')
#===================================Testing============================================
if isTest:
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        loss = criterion(y_pred, y_test)
        print(f'Test Loss: {loss.item():.4f}')