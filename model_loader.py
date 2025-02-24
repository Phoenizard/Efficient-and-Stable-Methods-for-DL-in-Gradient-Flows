import torch
from model.LinearRegressionModel import LinearRegressionModel
from data_loader import x_train, y_train, x_test, y_test

def RE(y, y_pred):
    return torch.mean(torch.abs(y-y_pred)/torch.abs(y))

def L2(y, y_pred):
    return torch.mean((y-y_pred)**2)

model = LinearRegressionModel()
model.load_state_dict(torch.load('model_SGD.pth'))
model.eval()

y_train_pred = model(x_train)
y_test_pred = model(x_test)

train_RE_loss = torch.log10(RE(y_train, y_train_pred))
test_RE_loss = torch.log10(RE(y_test, y_test_pred))

train_L2_loss = L2(y_train, y_train_pred)
test_L2_loss = L2(y_test, y_test_pred)

print(f'Train RE Loss: {train_RE_loss:.8f}')
print(f'Test RE Loss: {test_RE_loss:.8f}')
print(f'Train Loss: {train_L2_loss:.8f}')
print(f'Test Loss: {test_L2_loss:.8f}')