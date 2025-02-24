import torch
import torch.nn as nn
import torch.optim as optim
from criterion.RE_Loss import RelativeErrorLoss
from model.LinearRegressionModel import LinearRegressionModel
import matplotlib.pyplot as plt
from data_loader import train_loader, x_test, y_test, x_train, y_train
#================================================================================================
learning_rate = 0.2
epoch_num = 10000
isTest = False
#================================================================================================
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
epochs = epoch_num
train_losses = []
test_losses = []
#===================================Training Loop=======================================
for epoch in range(epochs):
    for inputs, targets in train_loader:
        model.train()
        optimizer.zero_grad() 
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_trian_pred = model(x_train)
        y_test_pred = model(x_test)
        train_loss = criterion(y_trian_pred, y_train)
        test_loss = criterion(y_test_pred, y_test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#===================================Plotting=======================================
plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), test_losses, label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()
#===================================Model Saving=======================================
torch.save(model.state_dict(), 'model_SGD_RE.pth')
#===================================Testing============================================
if isTest:
    model.eval()
    with torch.no_grad():
        y_train_pred = model()
        y_pred = model(x_test)
        loss = criterion(y_pred, y_test)
        print(f'Test Loss: {loss.item():.4f}')
